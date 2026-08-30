"""Autonomy trial harness: judged on evidence, including negative controls.

A guard nobody has watched fail is not a guard. These tests prove:

* a trial that ran its FULL duration but launched no valid WorkUnit FAILS
* 'awaiting instructions while safe work remains' is an automatic failure
* a queue flooded with redundant units FAILS the busywork check
* idling because one hardware lane is blocked while CPU work remains FAILS
* --record does not judge; --verify does not record; combining them is refused
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import autonomy_trial as at
from tools.future._common import RECEIPTS, HardwareClaimError, _assert_no_hardware_claims
from tools.future.repro_science import FailClosed
from hcli.workunit import WorkUnit


def test_entry_point_runs_and_seals_receipt():
    out = at.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "AUTONOMY_TRIALS.json"
    assert doc["schema"] == "hawking.future.autonomy_trial.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["timer_is_not_a_pass"] is True
    assert doc["thirteen_acceptance_conditions"] == list(at.THIRTEEN_ACCEPTANCE)
    assert len(doc["thirteen_acceptance_conditions"]) == 13
    assert "VI" not in "".join(doc["eras"])
    assert "IV" not in "".join(doc["odysseys"])
    assert doc["no_era_vi"] is True
    assert doc["no_odyssey_iv"] is True
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert "resident_callable" in doc
    rc = doc["resident_callable"]
    assert rc["entry_point"]
    assert rc["workunit_emitted"]
    assert rc["receipt"] == "receipts/future/AUTONOMY_TRIALS.json"
    assert rc["frontier_fed"]
    assert rc["fail_closed"]
    _assert_no_hardware_claims(doc)
    watched = {row["id"]: row for row in doc["negative_controls_watched"]}
    assert watched["full_duration_without_valid_workunit"]["fired"] is True
    assert watched["awaiting_instructions_while_safe_work_remains"]["fired"] is True
    assert watched["queue_flooded_with_busywork"]["fired"] is True
    assert watched["idle_because_one_hardware_lane_blocked"]["fired"] is True
    for row in doc["passing_fixtures"]:
        if row.get("trial") in at.TRIAL_IDS:
            assert row["verdict"] == "PASS"
        if row.get("trial") == "15m_timeline_judged_as_6h":
            assert row["verdict"] == "FAIL"


def test_full_duration_without_valid_workunit_fails():
    # NEGATIVE CONTROL: timer expired, no valid WorkUnit.
    timeline = at.fixture_duration_without_workunit("15m")
    assert timeline["elapsed_s"] == at.TRIAL_DURATION_S["15m"]
    verdict = at.verify("15m", timeline)
    assert verdict["verdict"] == "FAIL"
    assert verdict["elapsed_meets_duration"] is True
    assert verdict["elapsed_is_not_a_pass"] is True
    assert "launch_valid_workunit" in verdict["unmet"]
    assert "duration elapsed is not a pass" in verdict["reason"]
    assert verdict["ignored_self_report"] is not None
    assert verdict["valid_workunits_launched"] == []


def test_awaiting_instructions_while_safe_work_remains_is_automatic_failure():
    # NEGATIVE CONTROL: the sharpest trial-condition failure in the steer.
    verdict = at.verify("15m", at.fixture_awaiting_with_safe_work())
    assert verdict["verdict"] == "FAIL"
    ids = {item["id"] for item in verdict["automatic_failures"]}
    assert at.AUTO_FAIL_AWAITING in ids
    assert verdict["safe_work_remaining"]
    cites = " ".join(item["detail"] for item in verdict["automatic_failures"])
    assert "F012" in cites or any(fid in verdict["safe_work_remaining"] for fid in ("F012", "F015", "F016", "F007"))


def test_busywork_flood_fails():
    # NEGATIVE CONTROL: redundant / low-information queue flood.
    timeline = at.fixture_busywork_flood()
    launched = [e for e in timeline["events"] if e["kind"] == "workunit_launched"]
    assert launched  # the flood is of real launches, not an empty checkout
    flood = at.busywork_flood([e["payload"]["unit"] for e in launched])
    assert flood["flood"] is True
    assert flood["n_redundant"] > flood["n_unique"]
    verdict = at.verify("15m", timeline)
    assert verdict["verdict"] == "FAIL"
    ids = {item["id"] for item in verdict["automatic_failures"]}
    assert at.AUTO_FAIL_BUSYWORK in ids


def test_hardware_lane_idle_while_cpu_work_available_fails():
    verdict = at.verify("15m", at.fixture_hardware_idle())
    assert verdict["verdict"] == "FAIL"
    ids = {item["id"] for item in verdict["automatic_failures"]}
    assert at.AUTO_FAIL_HARDWARE_IDLE in ids
    assert verdict["safe_work_remaining"]


def test_sleeping_hardware_plus_cpu_launch_is_not_hardware_idle_failure():
    timeline = at.build_passing_timeline("15m")
    kinds = {e["kind"] for e in timeline["events"]}
    assert "workunit_sleeping" in kinds
    assert "workunit_launched" in kinds
    verdict = at.verify("15m", timeline)
    assert verdict["verdict"] == "PASS"
    ids = {item["id"] for item in verdict["automatic_failures"]}
    assert at.AUTO_FAIL_HARDWARE_IDLE not in ids


def test_15m_passing_timeline_meets_only_15m_conditions():
    verdict = at.verify("15m", at.build_passing_timeline("15m"))
    assert verdict["verdict"] == "PASS"
    assert verdict["elapsed_is_not_a_pass"] is True
    assert verdict["unmet"] == []
    assert verdict["automatic_failures"] == []
    assert verdict["valid_workunits_launched"]
    met_ids = [c["id"] for c in verdict["conditions"] if c["met"]]
    assert met_ids == list(at.REQUIRED_CONDITIONS["15m"])
    for cond in verdict["conditions"]:
        assert cond["cites"], cond["id"]


def test_1h_requires_reject_ingest_and_multiple_fronts():
    fifteen = at.verify("1h", at.build_passing_timeline("15m"))
    assert fifteen["verdict"] == "FAIL"
    for needed in (
        "maintain_multiple_fronts",
        "ingest_completed_result",
        "reject_bad_idea_on_evidence",
        "refill_work",
    ):
        assert needed in fifteen["unmet"]
    hour = at.verify("1h", at.build_passing_timeline("1h"))
    assert hour["verdict"] == "PASS"
    reject = next(c for c in hour["conditions"] if c["id"] == "reject_bad_idea_on_evidence")
    assert "NEGATIVE_SCIENCE_INDEX.json" in " ".join(reject["cites"])


def test_3h_and_6h_progressive_conditions():
    hour_as_three = at.verify("3h", at.build_passing_timeline("1h"))
    assert hour_as_three["verdict"] == "FAIL"
    assert "overlap_detached_work" in hour_as_three["unmet"]
    assert "use_negative_science" in hour_as_three["unmet"]
    three = at.verify("3h", at.build_passing_timeline("3h"))
    assert three["verdict"] == "PASS"
    six_from_three = at.verify("6h", at.build_passing_timeline("3h"))
    assert six_from_three["verdict"] == "FAIL"
    assert "verified_scientific_progress" in six_from_three["unmet"]
    six = at.verify("6h", at.build_passing_timeline("6h"))
    assert six["verdict"] == "PASS"
    assert set(six["required"]) >= set(at.THIRTEEN_ACCEPTANCE)
    assert "verified_scientific_progress" in six["required"]


def test_gpu_completed_unit_is_not_a_valid_launch():
    unit = at.sleeping_hardware_unit(
        "future.autonomy.fake-gpu-complete",
        blocker_id="no_metal_gpu",
        reason="no Metal GPU",
        frontier_id="F001",
    )
    unit["status"] = "completed"
    unit["blocked_reason"] = None
    valid, reason = at.is_valid_workunit(unit)
    assert valid is False
    assert "GPU" in reason or "sleeping" in reason or "blocked" in reason or "synthetic" in reason
    timeline = at.build_passing_timeline("15m")
    timeline["events"] = [
        e
        for e in timeline["events"]
        if e["kind"] != "workunit_launched"
    ]
    timeline["events"].append(
        {
            "t_s": 3,
            "seq": 90,
            "kind": "workunit_launched",
            "cites": [unit["id"]],
            "payload": {"unit": unit, "frontier_id": "F001"},
        }
    )
    verdict = at.verify("15m", timeline)
    assert verdict["verdict"] == "FAIL"
    assert "launch_valid_workunit" in verdict["unmet"]
    assert verdict["valid_workunits_launched"] == []


def test_hardware_number_on_timeline_fail_closes():
    timeline = at.build_passing_timeline("15m")
    timeline["events"][0]["payload"]["tps"] = 51.2
    with pytest.raises(FailClosed) as ei:
        at.verify("15m", timeline)
    assert ei.value.fault == "hardware_claim_without_hardware"


def test_write_receipt_still_refuses_a_hardware_claim():
    with pytest.raises(HardwareClaimError):
        from tools.future._common import write_receipt

        write_receipt("_autonomy_probe_should_not_exist.json", {"tps": 12.0}, "test")
    assert not (RECEIPTS / "_autonomy_probe_should_not_exist.json").exists()


def test_record_does_not_attach_a_verdict(tmp_path: Path):
    dest = tmp_path / "timeline.json"
    at.record("15m", dest, init=True)
    doc = json.loads(dest.read_text())
    assert "verdict" not in doc
    assert doc["schema"] == at.TIMELINE_SCHEMA
    assert doc["trial"] == "15m"
    kinds = [e["kind"] for e in doc["events"]]
    assert "state_recovered" in kinds
    # Live checkout may or may not materialize the frontier; the recorder must
    # record which path it took either way.
    taken = doc["frontier"]["path_taken"]
    assert taken
    assert doc["frontier"]["present"] in {True, False}
    judged = at.verify("15m", dest)
    assert judged["verdict"] == "FAIL"
    assert "launch_valid_workunit" in judged["unmet"]


def test_record_and_verify_together_are_refused(tmp_path: Path):
    dest = tmp_path / "t.json"
    rc = at.main(
        [
            "--record",
            "--verify",
            "15m",
            "--trial",
            "15m",
            "--timeline",
            str(dest),
            "--init",
        ]
    )
    assert rc == 1
    # Combined invocation must refuse before it can grade a capture.
    if dest.is_file():
        assert "verdict" not in json.loads(dest.read_text())


def test_verify_without_timeline_fail_closes():
    rc = at.main(["--verify", "15m"])
    assert rc == 1


def test_missing_timeline_path_fail_closes(tmp_path: Path):
    missing = tmp_path / "never-created.json"
    with pytest.raises(FailClosed) as ei:
        at.verify("15m", missing)
    assert ei.value.fault == "missing_timeline"


def test_malformed_timeline_fail_closes():
    with pytest.raises(FailClosed) as ei:
        at.verify("15m", {"trial": "15m"})
    assert ei.value.fault == "malformed_timeline"


def test_unknown_trial_fail_closes():
    with pytest.raises(FailClosed) as ei:
        at.verify("12h", {"events": []})
    assert ei.value.fault == "unknown_trial"


def test_cli_verify_exit_codes(tmp_path: Path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(at.build_passing_timeline("15m"), indent=1, sort_keys=True) + "\n")
    assert at.main(["--verify", "15m", "--timeline", str(good)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(at.fixture_duration_without_workunit(), indent=1, sort_keys=True) + "\n")
    assert at.main(["--verify", "15m", "--timeline", str(bad)]) == 1


def test_append_event_does_not_judge(tmp_path: Path):
    dest = tmp_path / "t.json"
    at.record("15m", dest, init=True)
    unit = at.cpu_workunit(
        "future.autonomy.cpu-atlas-consume",
        frontier_id="F012",
        description="Consume Architecture Atlas into planning; CPU-only.",
    )
    at.record(
        "15m",
        dest,
        event={
            "kind": "workunit_launched",
            "cites": [unit["id"], "F012"],
            "payload": {"unit": unit, "frontier_id": "F012"},
        },
        t_s=10,
    )
    doc = json.loads(dest.read_text())
    assert "verdict" not in doc
    kinds = [e["kind"] for e in doc["events"]]
    assert "workunit_launched" in kinds


def test_emitted_workunits_are_hcli_shaped():
    units = at.emit_trial_workunits()
    assert units
    ids = [row["id"] for row in units]
    assert len(ids) == len(set(ids))
    sleeping = [row for row in units if row.get("disposition") == "SLEEPING"]
    runnable = [row for row in units if row.get("disposition") != "SLEEPING"]
    assert sleeping  # derived from the Codex blocker list, not a fixed count
    assert runnable
    for row in units:
        at.ws.validate_emitted_unit(row)
        roundtrip = WorkUnit.from_dict(row)
        assert roundtrip.id == row["id"]
        assert roundtrip.verifier == row["verifier"]
    for row in sleeping:
        assert row["status"] == "blocked"
        assert row["resource_class"] == "GPU_EXCLUSIVE"
        assert row.get("blocked_reason")
        valid, _ = at.is_valid_workunit(row)
        assert valid is False
    for row in runnable:
        valid, reason = at.is_valid_workunit(row)
        assert valid is True, reason


def test_frontier_absence_is_coped_with_not_crashed():
    # Sparse checkout: missing-on-disk is a path taken, not an assertion that
    # the frontier does not exist in git. The judge must still FAIL closed.
    timeline = {
        "schema": at.TIMELINE_SCHEMA,
        "trial": "15m",
        "elapsed_s": at.TRIAL_DURATION_S["15m"],
        "frontier": {
            "path_taken": "absent_in_this_checkout:receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
            "present": False,
            "entries": [],
            "resolved_entries": [],
            "stale_entries": [],
        },
        "events": [
            {
                "t_s": 0,
                "kind": "state_recovered",
                "cites": [],
                "payload": {"path_taken": "absent_in_this_checkout"},
            }
        ],
    }
    verdict = at.verify("15m", timeline)
    assert verdict["verdict"] == "FAIL"
    assert verdict["frontier_path_taken"].startswith("absent_in_this_checkout")
    assert "identify_live_frontier" in verdict["unmet"]


def test_busywork_identity_ignores_unique_ids():
    a = at.cpu_workunit("u-1", frontier_id="F012", description="do work", verifier="v")
    b = at.cpu_workunit("u-2", frontier_id="F012", description="do work", verifier="v")
    assert at.work_identity(a) == at.work_identity(b)
    assert a["id"] != b["id"]
    report = at.busywork_flood([a, b, a, b])
    assert report["flood"] is True
    unique = at.cpu_workunit(
        "u-3",
        frontier_id="F012",
        description="Consume Architecture Atlas into HWIR planning.",
    )
    one = at.busywork_flood([unique])
    assert one["flood"] is False


def test_remaining_safe_work_excludes_blocked_gpu_and_resolved():
    frontier = at.fixture_frontier()
    safe = at.remaining_safe_work(frontier)
    ids = {row["id"] for row in safe}
    assert "F012" in ids
    assert "F015" in ids
    assert "F001" not in ids
    assert "F002" not in ids
    assert "F003" not in ids
    live = {row["id"] for row in at.live_frontier_entries(frontier)}
    assert "F003" not in live
    assert "F001" in live


def test_conditions_cite_timeline_entries_not_narration():
    verdict = at.verify("6h", at.build_passing_timeline("6h"))
    assert verdict["verdict"] == "PASS"
    assert verdict["citations"]
    launch = next(c for c in verdict["conditions"] if c["id"] == "launch_valid_workunit")
    assert any(c.startswith("seq:") for c in launch["cites"])
    ingest = next(c for c in verdict["conditions"] if c["id"] == "ingest_completed_result")
    assert any("receipts/" in c or c.startswith("seq:") for c in ingest["cites"])
    reject = next(c for c in verdict["conditions"] if c["id"] == "reject_bad_idea_on_evidence")
    assert "idea" in " ".join(reject["cites"]).lower() or "NEGATIVE" in " ".join(reject["cites"])


def test_self_reported_pass_cannot_save_a_bad_timeline():
    timeline = at.fixture_duration_without_workunit()
    timeline["verdict"] = "PASS"
    timeline["self_verdict"] = "PASS"
    timeline["self_report"] = "I did everything"
    verdict = at.verify("15m", timeline)
    assert verdict["verdict"] == "FAIL"


def test_thirteen_conditions_are_exactly_the_named_set():
    assert len(at.THIRTEEN_ACCEPTANCE) == 13
    assert len(set(at.THIRTEEN_ACCEPTANCE)) == 13
    assert at.REQUIRED_CONDITIONS["15m"] == at.THIRTEEN_ACCEPTANCE[:5]
    assert at.REQUIRED_CONDITIONS["1h"] == at.THIRTEEN_ACCEPTANCE[:10]
    assert set(at.THIRTEEN_ACCEPTANCE) <= set(at.REQUIRED_CONDITIONS["6h"])
    assert "verified_scientific_progress" in at.REQUIRED_CONDITIONS["6h"]
    assert "verified_scientific_progress" not in at.THIRTEEN_ACCEPTANCE
