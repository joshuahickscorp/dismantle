"""HUMF pins. FRONT H (G050)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
from humf import (Domain, Humf, HumfError, HumfObject, Materialization,  # noqa: E402
                  MockExternalMemoryProvider, State)

N = 1 << 20
NB = N * 4


PAYLOAD = bytes(range(256)) * (NB // 256)


def fabric():
    mp = MockExternalMemoryProvider(capacity_bytes=1 << 30, bandwidth_gb_s=5.0,
                                    latency_s=1e-4)
    doms = {"APPLE_UM": Domain("APPLE_UM", 96 << 30, 589.73, physical=True),
            mp.domain.name: mp.domain}
    # the fabric knows the mock domain is provider-backed, so a transfer into it
    # really moves bytes and can really fail
    h = Humf(doms, providers={mp.domain.name: mp})
    o = HumfObject("W42", "tensor", N, "f32", recompute_cost_s=0.05)
    # a CLEAN copy carries its bytes. Before this the fixture had a copy marked
    # CLEAN that held nothing, which is the fiction the executor's bookkeeping-only
    # transfer allowed.
    o.place(Materialization("APPLE_UM", "dense_f32", "row_major", NB, State.CLEAN,
                            payload=PAYLOAD))
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


# ------------------------------------------ failure injection through the executor

def test_a_failed_transfer_leaves_the_destination_INVALID_not_CLEAN():
    """The defect this closes: execute() transitioned TRANSFERRING -> CLEAN on
    bookkeeping alone, never calling the provider. A transfer that cannot fail is
    not a transfer, it is an assumption -- and a copy marked CLEAN that holds
    nothing is exactly the silent corruption this module claims to fail closed on."""
    h, o, mp = fabric()
    mp.fail_next = "copy"
    plan = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    with pytest.raises(HumfError, match="injected failure"):
        h.execute("W42", plan, "MOCK_EXTERNAL_VRAM")
    dst = o.materializations["MOCK_EXTERNAL_VRAM"]
    assert dst.state is State.INVALID
    assert "MOCK_EXTERNAL_VRAM" not in o.valid_copies()
    assert dst.payload is None


def test_a_failed_outbound_transfer_does_not_damage_the_source():
    """The source was good before and must be good after. Losing the only valid
    copy because a transfer failed would turn a recoverable error into data loss."""
    h, o, mp = fabric()
    mp.fail_next = "copy"
    with pytest.raises(HumfError):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"),
                  "MOCK_EXTERNAL_VRAM")
    src = o.materializations["APPLE_UM"]
    assert src.state is State.CLEAN and src.payload == PAYLOAD
    assert o.valid_copies() == ["APPLE_UM"]


def test_the_failure_is_recorded_in_the_log_as_FAILED():
    h, o, mp = fabric()
    mp.fail_next = "copy"
    with pytest.raises(HumfError):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"),
                  "MOCK_EXTERNAL_VRAM")
    assert h.log and h.log[-1]["outcome"] == "FAILED"


def test_a_successful_transfer_moves_the_bytes_intact():
    """The other half: the same path that can fail must, when it succeeds, deliver
    the actual data. Otherwise the failure test only proves an exception path."""
    h, o, mp = fabric()
    h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    dst = o.materializations["MOCK_EXTERNAL_VRAM"]
    assert dst.state is State.CLEAN
    assert dst.payload == PAYLOAD                       # bit-identical through the provider
    assert sorted(o.valid_copies()) == ["APPLE_UM", "MOCK_EXTERNAL_VRAM"]


def test_recovery_after_a_failed_transfer_is_possible():
    """INVALID must not be a dead end -- a retry has to be able to succeed, or a
    single transport hiccup would permanently strand the object."""
    h, o, mp = fabric()
    mp.fail_next = "copy"
    with pytest.raises(HumfError):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    o.materializations["MOCK_EXTERNAL_VRAM"].transition(State.ABSENT)
    h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    assert o.materializations["MOCK_EXTERNAL_VRAM"].payload == PAYLOAD


def test_a_transfer_with_no_valid_source_payload_is_refused():
    """A domain with no provider must still never conjure bytes from nothing."""
    h, o, mp = fabric()
    o.materializations["APPLE_UM"].payload = None
    with pytest.raises(HumfError, match="no valid source payload"):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    assert o.materializations["MOCK_EXTERNAL_VRAM"].state is State.INVALID


def test_recompute_actually_runs_the_recipe():
    h, o, mp = fabric()
    o.materializations["APPLE_UM"].transition(State.STALE)
    o.recompute = lambda: b"\xAB" * NB
    plan = h.plan_acquire("W42", "APPLE_UM")
    assert plan.action == "RECOMPUTE"
    m = h.execute("W42", plan, "APPLE_UM")
    assert m.state is State.CLEAN and m.payload == b"\xAB" * NB
