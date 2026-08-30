"""Negative controls for tools/future/evidence_dag.py.

A guard nobody has watched fail is not a guard. The three required refusals:

  * diamond graph: mutating one branch invalidates only its descendants; the
    sibling branch's proof survives and is reused without work
  * a promotion-adjacent candidate cannot be admitted below its required level
  * requesting V8 on this host RAISES rather than silently downgrading
"""
from __future__ import annotations

import json

import pytest

from tools.future import evidence_dag as ed
from tools.future import repro_science as rs
from tools.future._common import RECEIPTS, HardwareClaimError, _assert_no_hardware_claims, write_receipt


def test_build_emits_sealed_receipt():
    out = ed.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "EVIDENCE_DAG.json"
    assert doc["schema"] == ed.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert rs.seal_is_valid(doc)
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert "resident_callable" in doc
    assert doc["resident_callable"]["hcli_can_invoke"] is True
    assert doc["resident_callable"]["receipt"] == "receipts/future/EVIDENCE_DAG.json"
    assert len(doc["eras"]) == len(ed.ERAS)
    assert len(doc["odysseys"]) == len(ed.ODYSSEYS)
    assert "VI" not in "".join(doc["eras"])
    assert "Odyssey IV" not in "".join(doc["odysseys"])
    assert "IV" not in "".join(doc["odysseys"])
    assert doc["n_levels"] == len(ed.levels())
    assert doc["n_work_units"] == len(doc["work_units"])
    assert doc["reuse"]["skipped_work_on_identical_inputs"] is True
    assert doc["diamond_invalidation"]["holds"] is True
    assert doc["adaptive_depth"]["proof"]["holds"] is True
    assert doc["unavailable_levels"]["holds"] is True
    assert doc["claim_downgrade"]["holds"] is True
    _assert_no_hardware_claims(doc)


def test_selftest_emits_the_same_receipt():
    out = ed.selftest()
    assert out.name == "EVIDENCE_DAG.json"
    assert rs.seal_is_valid(json.loads(out.read_text()))


def test_ten_levels_declare_what_they_prove_and_do_not():
    rows = ed.levels()
    ids = [r["id"] for r in rows]
    assert ids == [f"V{i}" for i in range(len(rows))]
    assert ids[0] == "V0" and ids[-1] == "V9"
    for row in rows:
        assert row["proves"].strip()
        assert row["does_not_prove"].strip()
        assert row["proves"] != row["does_not_prove"]
        assert row["emits"] == "STATIC_ONLY"
    assert ed.LEVEL_BY_ID["V0"]["name"] == "schema/identity"
    assert ed.LEVEL_BY_ID["V1"]["name"] == "tiny numerical"
    assert ed.LEVEL_BY_ID["V2"]["name"] == "organ"
    assert ed.LEVEL_BY_ID["V3"]["name"] == "held-out organ"
    assert ed.LEVEL_BY_ID["V4"]["name"] == "short-chain"
    assert ed.LEVEL_BY_ID["V5"]["name"] == "deep-chain"
    assert ed.LEVEL_BY_ID["V6"]["name"] == "complete token"
    assert ed.LEVEL_BY_ID["V7"]["name"] == "capability subset"
    assert ed.LEVEL_BY_ID["V8"]["name"] == "protected capability/performance"
    assert ed.LEVEL_BY_ID["V9"]["name"] == "promotion/tournament"
    assert ed.LEVEL_BY_ID["V8"]["availability"] == "UNAVAILABLE"
    assert ed.LEVEL_BY_ID["V9"]["availability"] == "UNAVAILABLE"
    assert ed.LEVEL_BY_ID["V8"]["requires_gpu"] is True
    assert ed.LEVEL_BY_ID["V7"]["requires_gpu"] is False
    assert ed.LEVEL_BY_ID["V7"]["availability"] == "AVAILABLE"


def test_static_ladder_executes_every_available_level():
    proof = ed.static_ladder_proof()
    available = [s["id"] for s in ed.levels() if s["availability"] == "AVAILABLE"]
    assert proof["proved"] == available
    assert proof["work_count"] == len(available)
    assert proof["reused_top"] is True
    assert "V8" not in proof["proved"]
    assert "V9" not in proof["proved"]


def test_reuse_skips_work_on_byte_identical_inputs():
    proof = ed.reuse_skips_work_proof()
    assert proof["skipped_work_on_identical_inputs"] is True
    assert proof["changed_inputs_did_not_reuse"] is True
    assert proof["work_after_identical_reuse"] == proof["work_first_pass"]
    assert proof["work_after_input_change"] == proof["work_first_pass"] + 1
    assert proof["reuse_count_on_C"] >= 2


def test_identity_sensitive_to_inputs_code_genome_level():
    proof = ed.identity_sensitivity_proof()
    assert proof["reproduces"] is True
    assert proof["input_sensitive"] is True
    assert proof["code_sensitive"] is True
    assert proof["machine_genome_sensitive"] is True
    assert proof["level_sensitive"] is True


def test_negative_control_diamond_invalidation_sibling_survives():
    """Mutating one branch invalidates only its descendants.

    Sibling C's proof stays VALID and is reused without counting work.
    D refuses on a stale parent rather than silently passing.
    """
    dag = ed.make_diamond()
    ed.prove_diamond(dag)
    assert dag.statuses() == {"A": "VALID", "B": "VALID", "C": "VALID", "D": "VALID"}
    work_before = dag.work_count
    c_ident = dag.nodes["C"].proof_identity
    assert c_ident is not None

    affected = dag.mutate(
        "B",
        {**dag.nodes["B"].inputs, "payload": ed.payload_hash("left-branch-mutated")},
    )
    after = dag.statuses()
    assert set(affected) == {"B", "D"}
    assert after["B"] == "INVALID"
    assert after["D"] == "STALE"
    assert after["A"] == "VALID"
    assert after["C"] == "VALID"
    assert dag.nodes["C"].proof_identity == c_ident

    dag.prove("C")
    assert dag.work_count == work_before
    assert dag.proofs[c_ident].status == "VALID"
    assert dag.proofs[c_ident].reuse_count >= 1

    with pytest.raises(rs.FailClosed) as ei:
        dag.prove("D")
    assert ei.value.fault == "stale_parent"

    claims = dag.claim_statuses()
    assert claims["CLAIM_C"] == "VALID"
    assert claims["CLAIM_B"] == "DOWNGRADED"
    assert claims["CLAIM_D"] == "DOWNGRADED"
    assert claims["CLAIM_JOIN"] == "DOWNGRADED"

    sealed = ed.diamond_invalidation_proof()
    assert sealed["holds"] is True
    assert sealed["sibling_C_survived"] is True
    assert sealed["sibling_C_reused_without_work"] is True
    assert sealed["D_refused_on_stale_parent"] is True
    assert sealed["precise"] is True


def test_negative_control_promotion_adjacent_cannot_admit_below_required():
    """Both directions: cheap stays below V8; promotion-adjacent cannot enter at V7.

    Only enforcing one of those is how bars get quietly lowered.
    """
    cheap = ed.required_level(
        mutation_scope="tiny_numerical",
        uncertainty=0.05,
        risk=0.05,
        upside=0.10,
        promotion_proximity=0.0,
    )
    assert ed.level_ordinal(cheap) < 8
    assert (
        ed.admit_candidate(
            mutation_scope="tiny_numerical",
            uncertainty=0.05,
            risk=0.05,
            upside=0.10,
            promotion_proximity=0.0,
            achieved_level=cheap,
        )
        == "ADMITTED"
    )

    required = ed.required_level(
        mutation_scope="organ",
        uncertainty=0.05,
        risk=0.05,
        upside=0.10,
        promotion_proximity=0.90,
    )
    assert required == "V8"
    assert ed.LEVEL_BY_ID[required]["availability"] == "UNAVAILABLE"

    with pytest.raises(ed.BelowRequiredLevelError) as ei:
        ed.admit_candidate(
            mutation_scope="organ",
            uncertainty=0.05,
            risk=0.05,
            upside=0.10,
            promotion_proximity=0.90,
            achieved_level="V7",
        )
    assert ei.value.required == "V8"
    assert ei.value.achieved == "V7"
    assert ei.value.fault == "below_required_level"

    # Meeting the bar at V8 still refuses: this host cannot mint V8.
    with pytest.raises(ed.UnavailableLevelError) as e2:
        ed.admit_candidate(
            mutation_scope="organ",
            uncertainty=0.05,
            risk=0.05,
            upside=0.10,
            promotion_proximity=0.90,
            achieved_level="V8",
        )
    assert e2.value.level == "V8"
    assert e2.value.fault == "unavailable_level"


def test_negative_control_request_v8_raises_rather_than_downgrading():
    """request_level('V8') must RAISE. Returning V7 is the silent-downgrade defect."""
    with pytest.raises(ed.UnavailableLevelError) as ei:
        returned = ed.request_level("V8")
        raise AssertionError(f"V8 silently returned {returned!r} instead of raising")
    assert ei.value.level == "V8"
    assert ei.value.fault == "unavailable_level"

    with pytest.raises(ed.UnavailableLevelError) as e9:
        ed.request_level("V9")
    assert e9.value.level == "V9"

    v7 = ed.request_level("V7")
    assert v7["id"] == "V7"
    assert v7["availability"] == "AVAILABLE"

    dag = ed.EvidenceDAG()
    dag.add_node("P8", "V8", {"schema": ed.payload_hash("v8")})
    before = dag.work_count
    with pytest.raises(ed.UnavailableLevelError):
        dag.prove("P8")
    assert dag.work_count == before
    assert dag.nodes["P8"].status != "VALID"
    assert dag.nodes["P8"].proof_identity is None

    sealed = ed.unavailable_level_proof()
    assert sealed["request_V8_raises"] is True
    assert sealed["request_V9_raises"] is True
    assert sealed["request_V8_returned"] is None
    assert sealed["silent_downgrade_to_V7"] is False
    assert sealed["no_synthetic_V8_proof"] is True


def test_adaptive_depth_all_five_factors_move():
    proof = ed.adaptive_depth_proof()
    assert proof["cheap_not_v8"] is True
    assert proof["uncertainty_raises"] is True
    assert proof["risk_raises"] is True
    assert proof["upside_raises"] is True
    assert proof["scope_raises"] is True
    assert proof["promotion_adjacent_required"] == "V8"
    assert proof["tournament_adjacent_required"] == "V9"
    assert proof["bar_not_lowered_because_unavailable"] is True
    assert proof["promotion_adjacent_v7_refused"] is True
    assert proof["promotion_adjacent_v8_claim_refused"] is True


def test_unknown_scope_and_out_of_range_factors_fail_closed():
    with pytest.raises(rs.FailClosed) as ei:
        ed.required_level("not-a-scope", 0.1, 0.1, 0.1, 0.0)
    assert ei.value.fault == "unknown_mutation_scope"
    with pytest.raises(rs.FailClosed) as e2:
        ed.required_level("organ", 1.5, 0.1, 0.1, 0.0)
    assert e2.value.fault == "invalid_factor"
    with pytest.raises(rs.FailClosed) as e3:
        ed.required_level("organ", 0.1, float("nan"), 0.1, 0.0)
    assert e3.value.fault == "invalid_factor"
    with pytest.raises(rs.FailClosed) as e4:
        ed.required_level("organ", True, 0.1, 0.1, 0.0)
    assert e4.value.fault == "invalid_factor"


def test_claim_downgrade_uses_repro_science_ledger():
    proof = ed.claim_downgrade_on_dag_proof()
    assert proof["sibling_claim_survived"] is True
    assert proof["foundation_transitivity_holds"] is True
    assert proof["diamond"]["CLAIM_C"] == "VALID"
    assert proof["diamond"]["CLAIM_B"] == "DOWNGRADED"
    assert proof["diamond"]["CLAIM_JOIN"] == "DOWNGRADED"


def test_cycle_and_missing_node_fail_closed():
    dag = ed.EvidenceDAG()
    dag.add_node("A", "V0", {"schema": ed.payload_hash("a")})
    dag.add_node("B", "V1", {"schema": ed.payload_hash("b")})
    dag.add_edge("A", "B")
    with pytest.raises(rs.FailClosed) as ei:
        dag.add_edge("B", "A")
    assert ei.value.fault == "cycle"
    with pytest.raises(rs.FailClosed) as e2:
        dag.prove("NOPE")
    assert e2.value.fault == "missing_node"
    with pytest.raises(rs.FailClosed) as e3:
        dag.invalidate("NOPE")
    assert e3.value.fault == "missing_node"


def test_v8_workunit_is_sleeping_not_synthetic():
    units = ed.emit_work_units()
    by_id = {u["id"]: u for u in units}
    assert "future.evidence-dag.selftest" in by_id
    assert "future.evidence-dag.adapt-next-mutation" in by_id
    sleeping = by_id["future.evidence-dag.v8-protected-capability"]
    assert sleeping["status"] == "blocked"
    assert sleeping["classification"] == "BLOCKED"
    assert sleeping["sleeping"] is True
    assert sleeping["resource_class"] == "GPU_EXCLUSIVE"
    assert "UNAVAILABLE" in sleeping["blocked_reason"]
    assert sleeping["requested_level"] == "V8"
    selftest = by_id["future.evidence-dag.selftest"]
    assert selftest["status"] == "pending"
    assert selftest["resource_class"] == "STATIC_ANALYSIS"
    assert selftest["verifier"] == "future.evidence_dag.selftest"


def test_write_receipt_still_rejects_hardware_numbers():
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "MUST_NOT_EXIST_EVIDENCE_DAG.json",
            {"schema": "nope", "tps": 12.0},
            "tools/future/test_evidence_dag.py",
        )
    # Refusal is the proof; a sparse checkout must not be encoded as absence.


def test_unknown_level_request_fails_closed():
    with pytest.raises(rs.FailClosed) as ei:
        ed.request_level("V10")
    assert ei.value.fault == "unknown_level"
    with pytest.raises(rs.FailClosed) as e2:
        ed.level_ordinal("VX")
    assert e2.value.fault == "unknown_level"


def test_resident_callable_names_fail_closed_paths():
    units = ed.emit_work_units()
    doc = ed.resident_callable_doc(units)
    assert doc["entry_point"].endswith("evidence_dag.py --selftest")
    assert doc["receipt"] == "receipts/future/EVIDENCE_DAG.json"
    assert doc["frontier_fed"]["writes_frontier_file"] is False
    assert any("UnavailableLevelError" in s for s in doc["fail_closed"])
    assert any("BelowRequiredLevelError" in s for s in doc["fail_closed"])
    assert "future.evidence-dag.selftest" in doc["work_units_emitted"]
    assert "future.evidence-dag.v8-protected-capability" in doc["work_units_emitted"]
