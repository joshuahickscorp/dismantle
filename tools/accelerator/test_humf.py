"""HUMF pins. FRONT H (G050)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
from humf import (Domain, Humf, HumfError, HumfObject, Materialization,  # noqa: E402
                  MockExternalMemoryProvider, State)

N = 1 << 20
NB = N * 4


def fabric():
    mp = MockExternalMemoryProvider(capacity_bytes=1 << 30, bandwidth_gb_s=5.0,
                                    latency_s=1e-4)
    doms = {"APPLE_UM": Domain("APPLE_UM", 96 << 30, 589.73, physical=True),
            mp.domain.name: mp.domain}
    h = Humf(doms)
    o = HumfObject("W42", "tensor", N, "f32", recompute_cost_s=0.05)
    o.place(Materialization("APPLE_UM", "dense_f32", "row_major", NB, State.CLEAN))
    h.register(o)
    return h, o, mp


def test_illegal_transition_fails_closed():
    m = Materialization("APPLE_UM", "dense_f32", "row_major", NB, State.ABSENT)
    with pytest.raises(HumfError, match="illegal transition"):
        m.transition(State.CLEAN)          # must pass through MATERIALIZING


def test_evicted_cannot_become_clean_directly():
    m = Materialization("D", "r", "l", 8, State.EVICTED)
    with pytest.raises(HumfError):
        m.transition(State.CLEAN)


def test_a_write_makes_every_other_copy_stale():
    h, o, _ = fabric()
    p = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    h.execute("W42", p, "MOCK_EXTERNAL_VRAM")
    assert sorted(o.valid_copies()) == ["APPLE_UM", "MOCK_EXTERNAL_VRAM"]
    o.mark_written("MOCK_EXTERNAL_VRAM")
    assert o.materializations["APPLE_UM"].state is State.STALE
    assert o.is_dirty() and o.owner == "MOCK_EXTERNAL_VRAM"
    assert o.valid_copies() == []          # a dirty copy is not a valid copy


def test_a_stale_copy_is_never_offered_as_a_transfer_source():
    h, o, _ = fabric()
    h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    o.mark_written("MOCK_EXTERNAL_VRAM")
    plan = h.plan_acquire("W42", "APPLE_UM")
    assert all(opt.get("from") != "APPLE_UM" for opt in plan.options)
    # the only CLEAN-copy source is gone, so recompute is the honest answer
    assert plan.action == "RECOMPUTE"


def test_planner_prefers_the_cheaper_option():
    h, o, _ = fabric()
    plan = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    assert plan.action == "TRANSFER"       # 3.4ms transfer beats 50ms recompute
    o.recompute_cost_s = 1e-6              # now recompute is far cheaper
    assert h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM").action == "RECOMPUTE"


def test_a_plan_across_a_mock_domain_is_flagged_simulated():
    """The steer forbids treating a simulated transport number as physical
    evidence, so the plan must carry that on its face."""
    h, _, _ = fabric()
    plan = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    assert plan.cost_provenance == "SIMULATED"
    assert plan.rests_on_simulated_numbers is True


def test_recompute_cost_is_not_laundered_into_measured_transport():
    h, o, _ = fabric()
    o.recompute_cost_s = 1e-9
    plan = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    assert plan.action == "RECOMPUTE" and plan.cost_provenance == "MEASURED"


def test_impossible_when_no_clean_copy_and_no_recipe():
    h, o, _ = fabric()
    o.recompute_cost_s = None
    o.materializations["APPLE_UM"].transition(State.INVALID)
    plan = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    assert plan.action == "IMPOSSIBLE"
    with pytest.raises(HumfError):
        h.execute("W42", plan, "MOCK_EXTERNAL_VRAM")


def test_failure_injection_allocate():
    _, _, mp = fabric()
    mp.fail_next = "allocate"
    with pytest.raises(HumfError, match="injected failure: allocate"):
        mp.allocate(NB)
    mp.allocate(NB)                        # injection is one-shot


def test_capacity_is_enforced():
    mp = MockExternalMemoryProvider(capacity_bytes=1024, bandwidth_gb_s=1.0)
    with pytest.raises(HumfError, match="out of capacity"):
        mp.allocate(2048)


def test_mock_domain_is_never_marked_physical():
    mp = MockExternalMemoryProvider(capacity_bytes=1 << 20, bandwidth_gb_s=999.0)
    assert mp.domain.physical is False
    assert mp.domain.provenance == "SIMULATED"


def test_already_resident_costs_nothing():
    h, _, _ = fabric()
    plan = h.plan_acquire("W42", "APPLE_UM")
    assert plan.action == "ALREADY_RESIDENT" and plan.cost_s == 0.0
