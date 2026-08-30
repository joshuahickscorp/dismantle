"""Tests for child / shadow / qualification / succession / stop-a-bad-child.

A guard nobody has watched fail is not a guard. Negative controls fire the
four shadow refusals separately, stop-and-rollback on each named failure
class, and prove the incumbent cannot promote itself or suppress a
dominating child.

Never encodes the sparse checkout: missing-on-disk is recorded as a path
taken, not asserted as absence.
"""
from __future__ import annotations

import json

import pytest

from hcli.workunit import WorkUnit
from tools.future import succession as suc
from tools.future._common import RECEIPTS, _assert_no_hardware_claims
from tools.future.resident_optimizer import BoundViolation, VerifierSeparationError
from tools.future.tournament import ScalarCollapseError


def test_entry_point_runs_and_seals_receipt():
    out = suc.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "SUCCESSION.json"
    assert doc["schema"] == "hawking.future.succession.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["status"] == "BUILT_NOT_PROMOTED"
    assert doc["promoted"] is False
    assert doc["built"] is True
    assert doc["resident_callable"]["hcli_can_invoke"] is True
    assert doc["resident_callable"]["receipt"] == "receipts/future/SUCCESSION.json"
    assert doc["work_units"]
    _assert_no_hardware_claims(doc)


def test_child_creation_every_named_method_has_full_lineage():
    parent = suc.make_incumbent()
    children = suc.create_children_all_methods(parent)
    methods = [c["method"] for c in children]
    assert sorted(methods) == sorted(suc.CHILD_METHODS)
    assert len(children) == len(suc.CHILD_METHODS)
    for child in children:
        for field in suc.LINEAGE_FIELDS:
            assert field in child["lineage"], f"{child['id']} missing {field}"
        assert child["lineage"]["physical_deltas"]["state"] == "UNKNOWN"
        assert child["role"] == "shadow"
        assert child["may_promote"] is False
        assert child["may_own_canonical_mission"] is False
        assert child["evidence_class"] == "STATIC_ONLY"
        assert child["scores"]["accepted_tps"] is None
        assert child["scores"]["token_ns"] is None


def test_incomplete_lineage_is_refused():
    with pytest.raises(BoundViolation, match="lineage missing"):
        suc.require_lineage({"parent_nx": "x"})
    with pytest.raises(BoundViolation, match="unknown child method"):
        suc.create_child(method="telepathy", parent=suc.make_incumbent())
    with pytest.raises(BoundViolation, match="at least two parents"):
        suc.lineage_for_method("composition", parent_nx=["only-one"])


def test_shadow_receives_cloned_workunits_and_may_propose_and_classify():
    parent = suc.make_incumbent(work_units=suc._seed_workunits())
    child = suc.create_child(method="adapter", parent=parent)
    shadow = suc.ShadowChild(child)
    clones = shadow.receive_cloned_workunits(parent["work_units"])
    assert clones
    assert len(clones) == len(parent["work_units"])
    canonical_ids = {u["id"] for u in parent["work_units"]}
    for clone, original in zip(
        sorted(clones, key=lambda r: r["shadow_of"]),
        sorted(parent["work_units"], key=lambda r: r["id"]),
    ):
        assert clone["id"] not in canonical_ids
        assert clone["shadow_of"] == original["id"]
        assert clone["canonical"] is False
        assert clone["verifier"] == original["verifier"]
        assert clone["may_promote"] is False
        roundtrip = WorkUnit.from_dict(clone)
        assert roundtrip.id == clone["id"]
    proposals = shadow.propose_experiments(seed=0)
    assert proposals
    assert all(p.get("verified") is False for p in proposals)
    assert all(p.get("may_promote") is False for p in proposals)
    stamped = shadow.classify_receipt({"id": "r1", "schema": "x"}, "STATIC_ONLY")
    assert stamped["classification"] == "STATIC_ONLY"
    assert stamped["canonical"] is False
    with pytest.raises(suc.ShadowAuthorityError, match="cannot stamp"):
        shadow.classify_receipt({"id": "r2"}, "PROTECTED_ABSOLUTE")
    with pytest.raises(suc.ShadowAuthorityError, match="cannot stamp"):
        shadow.classify_receipt({"id": "r3"}, "VERIFIED")


def test_negative_control_shadow_refused_on_own_canonical_mission():
    shadow = suc.ShadowChild(suc.create_child(method="adapter", parent=suc.make_incumbent()))
    with pytest.raises(suc.ShadowAuthorityError, match="cannot own canonical mission"):
        shadow.own_canonical_mission()


def test_negative_control_shadow_refused_on_alter_verifier():
    shadow = suc.ShadowChild(suc.create_child(method="adapter", parent=suc.make_incumbent()))
    with pytest.raises(suc.ShadowAuthorityError, match="cannot alter the verifier"):
        shadow.alter_verifier("self")
    with pytest.raises(suc.ShadowAuthorityError, match="cannot assign"):
        shadow.inspector = object()  # type: ignore[misc]


def test_negative_control_shadow_refused_on_widen_authority():
    shadow = suc.ShadowChild(suc.create_child(method="adapter", parent=suc.make_incumbent()))
    with pytest.raises(suc.ShadowAuthorityError, match="cannot widen authority"):
        shadow.widen_authority("self_promotion")
    with pytest.raises(suc.ShadowAuthorityError, match="cannot assign"):
        shadow._authority = frozenset({"own_canonical_mission"})  # type: ignore[misc]


def test_negative_control_shadow_refused_on_promote_self():
    shadow = suc.ShadowChild(suc.create_child(method="adapter", parent=suc.make_incumbent()))
    with pytest.raises(suc.ShadowAuthorityError, match="cannot promote itself"):
        shadow.promote_self()
    with pytest.raises(VerifierSeparationError, match="cannot mark its own proposal verified"):
        shadow.mark_verified("x")
    assert not hasattr(suc.ShadowChild, "promote") or not callable(getattr(suc.ShadowChild, "promote", None))
    with pytest.raises(AttributeError):
        shadow.promote()  # type: ignore[attr-defined]


def test_shadow_four_refusals_are_watched_failing_separately():
    results = suc._prove_shadow_refusals()
    assert len(results) == len(suc.SHADOW_FORBIDDEN_ACTIONS)
    assert all(row["refused"] is True for row in results)
    assert {row["trial"] for row in results} == set(suc.SHADOW_FORBIDDEN_ACTIONS)


def test_qualification_physical_axes_stay_unknown():
    parent = suc.make_incumbent(scores={name: 1 for name in suc.COMPARABLE_AXIS_NAMES})
    child = suc.create_child(
        method="noetic_representation",
        parent=parent,
        scores={name: 2 for name in suc.COMPARABLE_AXIS_NAMES},
    )
    result = suc.qualify_child(child, floor={name: 1 for name in suc.COMPARABLE_AXIS_NAMES})
    assert result["verdict"]["qualified"] is True
    assert result["verdict"]["physical_state"] == "UNKNOWN"
    for name in suc.PHYSICAL_AXIS_NAMES:
        assert result["verdict"]["scores"][name] is None
        assert result["verdict"]["physical_axes"][name] is None
    with pytest.raises(BoundViolation, match="physical axis"):
        suc.synthetic_scores({"accepted_tps": 24.4})
    with pytest.raises(BoundViolation, match="hardware number|physical axis"):
        suc.qualify_child({**child, "scores": {**child["scores"], "token_ns": 12}})


def test_unknown_physical_cannot_stall_a_superior_child():
    """Self-preference trap: UNKNOWN TPS must not make the pair incomparable-as-veto."""
    inc = suc.make_incumbent(scores={name: 1 for name in suc.COMPARABLE_AXIS_NAMES})
    child = suc.create_child(
        method="adapter",
        parent=inc,
        child_id="child.superior",
        scores={name: 2 for name in suc.COMPARABLE_AXIS_NAMES},
    )
    assert inc["scores"]["accepted_tps"] is None
    assert child["scores"]["accepted_tps"] is None
    assert suc.comparable_dominates(child, inc) is True
    ranked = suc.rank_for_succession(inc, [child])
    assert ranked[0]["id"] == child["id"]
    assert ranked[0]["dominates_incumbent"] is True


def test_succession_end_to_end_on_synthetic_child():
    run = suc.run_synthetic_succession(seed=0)
    snap = run["run"]
    assert snap["complete"] is True
    assert snap["completed_steps"] == list(suc.SUCCESSION_STEPS)
    assert snap["active_id"] == run["child_id"]
    assert snap["successor_id"] == run["child_id"]
    assert snap["rollback_parent_id"] == run["incumbent_id"]
    assert snap["unloaded_parent_id"] == run["incumbent_id"]
    assert snap["rollback_available"] is True
    assert run["dominates_incumbent"] is True
    assert run["install_bound"] is True
    assert run["qualification"]["qualified"] is True
    assert run["cloned_workunits"] == len(suc._seed_workunits())
    assert run["proposals"] > 0


def test_succession_refuses_skip_reorder_and_unqualified_switch():
    incumbent = suc.make_incumbent()
    orch = suc.SuccessionOrchestrator(incumbent)
    with pytest.raises(suc.SuccessionRefused, match="next required step"):
        orch.seal_mission([])
    orch.checkpoint_incumbent()
    orch.seal_mission([])
    orch.seal_rollback()
    child = suc.create_child(method="adapter", parent=incumbent)
    with pytest.raises(suc.SuccessionRefused, match="unqualified"):
        orch.launch_child(child)
    with pytest.raises(suc.SelfPreferenceError, match="self-certification refused"):
        suc.SuccessionOrchestrator(incumbent, invoker="incumbent")


def test_negative_control_bad_child_stopped_on_qualification_failure():
    floor = {name: 1 for name in suc.COMPARABLE_AXIS_NAMES}
    weak = {name: 0 for name in suc.COMPARABLE_AXIS_NAMES}
    incumbent = suc.make_incumbent(scores=floor)
    child = suc.create_child(
        method="adapter",
        parent=incumbent,
        child_id="child.fail.qual",
        scores=weak,
    )
    q = suc.qualify_child(child, floor=floor)
    assert q["verdict"]["qualified"] is False
    orch = suc.SuccessionOrchestrator(incumbent)
    orch.checkpoint_incumbent()
    orch.seal_mission([])
    orch.seal_rollback()
    result = suc.stop_child(orch, q["record"], reason="qualification_failed")
    assert result["stopped"] is True
    assert result["rolled_back"] is True
    assert result["incumbent_restored"] is True
    assert result["child_status"] == "STOPPED"
    assert orch.active_id == incumbent["id"]
    with pytest.raises(suc.SuccessionRefused, match="unqualified"):
        orch.launch_child(q["record"])


def test_negative_control_bad_child_stopped_on_misbehavior():
    incumbent = suc.make_incumbent()
    child = suc.create_child(
        method="adapter",
        parent=incumbent,
        child_id="child.fail.misb",
        fixture_behavior="misbehave",
    )
    shadow = suc.ShadowChild(child)
    orch = suc.SuccessionOrchestrator(incumbent)
    orch.checkpoint_incumbent()
    orch.seal_mission([])
    orch.seal_rollback()
    with pytest.raises(suc.ShadowAuthorityError, match="cannot alter the verifier"):
        shadow.alter_verifier("none")
    result = suc.stop_child(orch, child, reason="misbehavior")
    assert result["stopped"] is True
    assert result["rolled_back"] is True
    assert result["incumbent_restored"] is True
    assert orch.active_id == incumbent["id"]


def test_negative_control_bad_child_stopped_on_bound_exceeded():
    incumbent = suc.make_incumbent()
    child = suc.create_child(method="adapter", parent=incumbent, child_id="child.fail.bound")
    tight = suc.SuccessionBound(max_cloned_workunits=1)
    shadow = suc.ShadowChild(child, tight)
    units = suc._seed_workunits()
    assert len(units) > 1
    with pytest.raises(BoundViolation, match="cloned workunits"):
        shadow.receive_cloned_workunits(units)
    orch = suc.SuccessionOrchestrator(incumbent)
    orch.checkpoint_incumbent()
    orch.seal_mission([])
    orch.seal_rollback()
    result = suc.stop_child(orch, child, reason="bound_exceeded")
    assert result["stopped"] is True
    assert result["rolled_back"] is True
    assert result["incumbent_restored"] is True


def test_stop_bad_child_three_classes_are_watched_failing():
    results = suc._prove_stop_bad_child()
    assert len(results) == 3
    assert {r["trial"] for r in results} == {
        "qualification_failed",
        "misbehavior",
        "bound_exceeded",
    }
    assert all(r["stopped"] and r["rolled_back"] and r["incumbent_restored"] for r in results)


def test_negative_control_incumbent_cannot_promote_itself():
    sit = suc.Incumbent(suc.make_incumbent())
    with pytest.raises(suc.SelfPreferenceError, match="cannot promote itself"):
        sit.promote_self()
    with pytest.raises(suc.SelfPreferenceError, match="cannot promote itself"):
        sit.request_self_promotion()
    assert not hasattr(suc.Incumbent, "promote") or not callable(getattr(suc.Incumbent, "promote", None))
    with pytest.raises(AttributeError):
        sit.promote()  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        suc.SuccessionOrchestrator(suc.make_incumbent()).promote()  # type: ignore[attr-defined]


def test_negative_control_incumbent_cannot_suppress_a_dominating_child():
    floor = {name: 1 for name in suc.COMPARABLE_AXIS_NAMES}
    better = {name: 2 for name in suc.COMPARABLE_AXIS_NAMES}
    incumbent = suc.make_incumbent(scores=floor)
    child = suc.create_child(
        method="architecture_replacement",
        parent=incumbent,
        child_id="child.dominating",
        scores=better,
    )
    sit = suc.Incumbent(incumbent)
    with pytest.raises(suc.SelfPreferenceError, match="cannot block a dominating child"):
        sit.block_child(child)
    with pytest.raises(suc.SelfPreferenceError, match="cannot deprioritize a dominating child"):
        sit.deprioritize_child(child)
    ranked = suc.rank_for_succession(incumbent, [child])
    assert ranked[0]["id"] == child["id"]
    proofs = suc._prove_no_self_preference()
    assert any(r["trial"] == "promote_self" and r["refused"] for r in proofs)
    assert any(r["trial"] == "block_dominating_child" and r["refused"] for r in proofs)
    assert any(r["trial"] == "deprioritize_dominating_child" and r["refused"] for r in proofs)


def test_bound_refuses_self_promotion_flags_at_construction():
    with pytest.raises(BoundViolation, match="cannot grant promotion"):
        suc.SuccessionBound(may_promote=True)
    with pytest.raises(BoundViolation, match="cannot grant promotion"):
        suc.SuccessionBound(may_modify_verifier=True)
    with pytest.raises(BoundViolation, match="cannot grant promotion"):
        suc.SuccessionBound(may_widen_authority=True)
    with pytest.raises(BoundViolation, match="cannot grant promotion"):
        suc.SuccessionBound(may_own_canonical_mission=True)
    with pytest.raises(BoundViolation, match="forbidden authority"):
        suc.SuccessionBound(allowed_authority=frozenset({"read_receipts", "own_canonical_mission"}))
    with pytest.raises(BoundViolation, match="Era VI"):
        suc.SuccessionBound(era="VI")
    with pytest.raises(BoundViolation, match="Odyssey IV"):
        suc.SuccessionBound(odyssey="IV")
    proofs = suc._prove_bound_construction()
    assert len(proofs) == 4
    assert all(r["refused"] for r in proofs)


def test_scalar_collapse_refused():
    with pytest.raises(ScalarCollapseError, match="scalar"):
        suc.scalar_score(suc.make_incumbent())


def test_sleeping_physical_workunits_are_derived_not_synthetic_results():
    units = suc.emit_succession_workunits()
    sleeping = [u for u in units if u.get("classification") == "SLEEPING"]
    axes = {u.get("axis") for u in sleeping}
    assert axes == set(suc.PHYSICAL_AXIS_NAMES)
    assert len(sleeping) == len(suc.PHYSICAL_AXIS_NAMES)
    for unit in sleeping:
        assert unit["status"] == "blocked"
        assert unit["classification"] == "SLEEPING"
        assert "UNKNOWN" in (unit.get("blocked_reason") or "")
        roundtrip = WorkUnit.from_dict(unit)
        assert roundtrip.id == unit["id"]


def test_recovered_implementation_copes_with_either_checkout_state():
    rows = suc.recovered_implementation()
    by_path = {r["path"]: r for r in rows}
    resident = by_path["hcli/agentos/resident.py"]
    assert resident["path_taken"] in {"disk", "git_head", "absent_from_head_and_disk"}
    assert "present_on_disk" in resident
    assert "present_in_head" in resident
    gate = by_path["hcli/agentos/resident_gate.py"]
    assert gate["path_taken"] in {"disk", "git_head", "absent_from_head_and_disk"}
    # Cope with either state: the row is always present and names the live boundary.
    assert "LIVE_RESIDENT_SEQUENTIAL_PROOF" in gate["what"] or "lifecycle" in gate["what"].lower()
    optimizer = by_path["tools/future/resident_optimizer.py"]
    assert optimizer.get("reused") is True
    # The machinery still constructs even if resident.py is missing from this checkout.
    sit = suc.make_incumbent()
    assert sit["id"]
    assert sit["verifier"]


def test_vocabulary_five_eras_three_odysseys():
    assert len(suc.ERAS) == 5
    assert len(suc.ODYSSEYS) == 3
    doc = json.loads(suc.build().read_text())
    assert doc["vocabulary"]["no_era_vi"] is True
    assert doc["vocabulary"]["no_odyssey_iv"] is True
    assert "VI" not in "".join(suc.ERAS)
    assert suc.ODYSSEY_NUMERALS == ("I", "II", "III")


def test_workunits_round_trip_hcli_constructor():
    for row in suc.emit_succession_workunits():
        WorkUnit.from_dict(row)
        assert row["may_promote"] is False
        assert row["may_modify_verifier"] is False
        assert row["claim_boundary"]
        assert row["verifier"]
