"""Odyssey I launch gate tests.

Negative control: can_launch() is False today and names every unmet
criterion; ODYSSEY_I_LAUNCH.json is not written while the gate refuses.
A guard nobody has watched fail is not a guard.

Sparse-checkout trap: these tests never encode the checkout by asserting
that an unrelated file is absent. They assert the module copes with either
state and records which path it took.
"""
from __future__ import annotations

import hashlib
import ast
import json
import pathlib
from pathlib import Path

import pytest

from tools.future import odyssey_launch as ol
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, write_receipt
from hcli.workunit import WorkUnit


def test_entry_point_runs_and_seals_gate_receipt():
    out = ol.build()
    path = Path(out["gate_path"])
    assert path.parent == RECEIPTS
    assert path.name == ol.RECEIPT
    doc = json.loads(path.read_text())
    assert doc["schema"] == ol.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["measurement_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["no_era_vi"] is True
    assert doc["no_odyssey_iv"] is True
    assert "recovered_implementation" in doc
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["workunit_emitted"]
    assert doc["resident_callable"]["receipt"]
    assert doc["resident_callable"]["frontier_fed"]
    assert doc["resident_callable"]["fail_closed"]


def test_sixteen_criteria_evaluated_in_contract_order():
    results = ol.evaluate_launch_criteria()
    assert [r["id"] for r in results] == list(ol.CRITERION_IDS)
    assert len(results) == len(ol.CRITERION_IDS)
    assert len(set(r["id"] for r in results)) == len(ol.CRITERION_IDS)
    for row in results:
        assert "met" in row
        assert row["reason"]
        assert "evidence" in row
        assert "operational" in row


def test_can_launch_false_today_names_every_unmet():
    """NEGATIVE CONTROL: refuse today, and name every unmet criterion."""
    results = ol.evaluate_launch_criteria()
    unmet = ol.unmet_criteria(results)
    assert ol.can_launch(results) is False
    assert unmet == [r["id"] for r in results if not r["met"]]
    assert unmet, "today several criteria are unmet; an empty unmet list would open the gate"
    # These two cannot be met without work that has not happened: no autonomy
    # trial has passed, and there is no callable NR/NX path on this host.
    assert "resident_autonomy_trial_pass" in unmet
    assert "nr_nx_path_callable" in unmet
    # Deliberately NOT asserting a frozen list. Criteria become met as their
    # capability actually lands -- `workgraphs` did, once the runtime was
    # exercised rather than merely looked for. Pinning the unmet set would make
    # this test fight real progress.
    verdict = ol.launch_verdict(results)
    assert verdict["verdict"] == "REFUSED"
    assert verdict["allowed"] is False
    assert verdict["unmet"] == unmet
    assert verdict["n_unmet"] == len(unmet)
    assert verdict["n_criteria"] == len(ol.CRITERION_IDS)


def test_unmet_aggregator_does_not_stop_at_first():
    """A first failure must not hide later failures."""
    fake = []
    for i, cid in enumerate(ol.CRITERION_IDS):
        fake.append({"id": cid, "met": i not in (0, 5, len(ol.CRITERION_IDS) - 1)})
    unmet = ol.unmet_criteria(fake)
    assert unmet[0] == ol.CRITERION_IDS[0]
    assert ol.CRITERION_IDS[5] in unmet
    assert unmet[-1] == ol.CRITERION_IDS[-1]
    assert unmet == [ol.CRITERION_IDS[0], ol.CRITERION_IDS[5], ol.CRITERION_IDS[-1]]
    assert ol.can_launch(fake) is False
    only_last = [{"id": cid, "met": cid != ol.CRITERION_IDS[-1]} for cid in ol.CRITERION_IDS]
    assert ol.unmet_criteria(only_last) == [ol.CRITERION_IDS[-1]]
    all_met = [{"id": cid, "met": True} for cid in ol.CRITERION_IDS]
    assert ol.can_launch(all_met) is True
    assert ol.unmet_criteria(all_met) == []


def test_launch_receipt_not_written_while_gate_refuses(tmp_path, monkeypatch):
    """NEGATIVE CONTROL: refuse must not write ODYSSEY_I_LAUNCH.json."""
    written: list[str] = []

    def spy(name, doc, recorded_by):
        written.append(name)
        path = tmp_path / name
        path.write_text(json.dumps({"name": name, "recorded_by": recorded_by}))
        return path

    out = ol.build(writer=spy)
    assert ol.RECEIPT in written
    assert ol.LAUNCH_RECEIPT not in written
    assert out["launch"]["written"] is False
    assert out["doc"]["odyssey_i_launch_written"] is False
    assert out["doc"]["phase_transition"] == "NOT_STARTED"
    assert out["doc"]["verdict"]["verdict"] == "REFUSED"
    assert (tmp_path / ol.LAUNCH_RECEIPT).exists() is False
    assert (tmp_path / ol.RECEIPT).is_file()


def test_forced_pass_writes_launch_receipt(tmp_path):
    payload = {"schema": ol.LAUNCH_SCHEMA, "phase_transition": "STARTED"}
    refused = ol.write_launch_if_passed(payload, allowed=False, writer=lambda n, d, r: tmp_path / n)
    assert refused["written"] is False
    launched = ol.write_launch_if_passed(
        payload,
        allowed=True,
        writer=lambda n, d, r: (tmp_path / n).write_text("ok") or (tmp_path / n),
    )
    assert launched["written"] is True
    assert launched["name"] == ol.LAUNCH_RECEIPT
    assert (tmp_path / ol.LAUNCH_RECEIPT).is_file()


def test_hcli_autonomy_gate_is_not_odyssey_trial():
    row = next(r for r in ol.evaluate_launch_criteria() if r["id"] == "resident_autonomy_trial_pass")
    assert row["met"] is False
    kinds = {e.get("kind") for e in row["evidence"]}
    assert "hcli_agentos_autonomy" in kinds
    hcli = next(e for e in row["evidence"] if e.get("kind") == "hcli_agentos_autonomy")
    assert hcli.get("not_this_criterion") is True
    assert "path_taken" in hcli
    # Cope with either state: found or not, the path taken is recorded.
    assert hcli["path_taken"] in {
        "worktree",
        "evidence_snapshot",
        "primary_checkout",
        "git_head",
        "not_found",
    }


def test_probe_json_records_path_taken_either_state():
    hit = ol.probe_json("receipts/future/ODYSSEY2_LAW_STORE.json")
    assert hit["found"] is True
    assert hit["path_taken"] in {"worktree", "evidence_snapshot", "primary_checkout", "git_head"}
    assert isinstance(hit["doc"], dict)
    miss = ol.probe_json("receipts/future/THIS_RECEIPT_IS_NOT_PART_OF_THE_CONTRACT_zzzz.json")
    assert miss["found"] is False
    assert miss["path_taken"] == "not_found"
    assert "searched" in miss
    assert miss["doc"] is None


def test_specimen_curriculum_five_roles_not_exhaustive():
    cur = ol.propose_specimen_curriculum()
    roles = [r["role"] for r in cur["roles"]]
    assert roles == [c[0] for c in ol.CURRICULUM_ROLES]
    assert cur["n_roles"] == len(ol.CURRICULUM_ROLES)
    assert cur["n_ready"] == sum(1 for r in cur["roles"] if r.get("ready"))
    assert cur["ready"] is (cur["n_ready"] == cur["n_roles"])
    # Unready today is expected; do not freeze the ready count.
    assert cur["ready"] is False
    purposes = {r["role"]: r["purpose"] for r in cur["roles"]}
    assert "procedural speed" in purposes["very_small_dense_procedural_speed"]
    assert "alternate architecture" in purposes["small_dense_alternate_architecture_transfer"]
    assert "compiler" in purposes["mid_size_dense_compiler"]
    assert "Qwen27" in purposes["qwen27_mature_physical"]
    assert "Flash" in purposes["flash_heterogeneous_frontier"]
    extras = {row["repo"] for row in cur.get("not_proposed") or []}
    first_wave = {r["repo"] for r in cur["roles"]}
    assert extras.isdisjoint(first_wave)
    assert "Do not exhaustively optimize" in cur["not_proposed_rule"]


def test_first_workgraphs_are_real_workunits_with_dependencies():
    cur = ol.propose_specimen_curriculum()
    first = cur["roles"][0]
    graphs = ol.emit_first_workgraphs(first)
    units = graphs["units"]
    assert graphs["n_units"] == len(units)
    assert graphs["stages"] == list(ol.GRAPH_STAGES)
    by_id = {u["id"]: u for u in units}
    assert len(by_id) == len(units)
    for u in units:
        ol.wus.validate_emitted_unit(u)
        WorkUnit.from_dict(u)
        assert u["provider"] == "future.odyssey_launch"
        assert u["verifier"]
        assert u["claim_boundary"]
    slug = graphs["specimen"]["slug"]
    prefix = f"odyssey-i.wg.{slug}"
    for prev, stage in zip([None, *ol.GRAPH_STAGES[:-1]], ol.GRAPH_STAGES):
        uid = f"{prefix}.{stage}"
        deps = by_id[uid]["dependencies"]
        if prev is None:
            assert deps == []
        else:
            assert deps == [f"{prefix}.{prev}"]
    laws_id = f"{prefix}.laws"
    ii = by_id[f"{prefix}.phase_ii_transfer"]
    iii = by_id[f"{prefix}.phase_iii_attack"]
    assert ii["dependencies"] == [laws_id]
    assert iii["dependencies"] == [laws_id]
    assert f"{prefix}.phase_iii_attack" not in ii["dependencies"]
    assert f"{prefix}.phase_ii_transfer" not in iii["dependencies"]
    for stage in ol.GPU_STAGES:
        u = by_id[f"{prefix}.{stage}"]
        assert u["status"] == "sleeping"
        assert u["classification"] == "SLEEPING"
        assert u["resource_class"] == "GPU_EXCLUSIVE"
        assert "synthetic" in (u.get("blocked_reason") or "").lower() or "SLEEPING" in (u.get("blocked_reason") or "")


def test_phase_ii_iii_listen_concurrently_no_barrier():
    policy = ol.phase_listen_policy("phase_i.laws")
    assert policy["barrier"] is None
    assert policy["global_barrier"] is False
    assert policy["phase_ii_depends_on_phase_iii"] is False
    assert policy["phase_iii_depends_on_phase_ii"] is False
    assert policy["phase_ii_depends_on"] == ["phase_i.laws"]
    assert policy["phase_iii_depends_on"] == ["phase_i.laws"]
    assert policy["no_odyssey_iv"] is True
    assert policy["no_era_vi"] is True


def test_sleeping_physical_blockers_are_workunits_not_synthetic_results():
    blockers = ol.physical_blockers()
    holding = [b for b in blockers if b.get("holds")]
    assert holding, "live physical blockers hold today; an empty list would invent a clear machine"
    for b in holding:
        assert b["sleeping"] is True
        wu = b.get("workunit")
        assert wu is not None
        ol.wus.validate_emitted_unit(wu)
        assert wu["status"] == "sleeping"
        assert wu["classification"] == "SLEEPING"
        assert wu.get("synthetic_result_forbidden") is True
        assert wu.get("blocked_reason")


def test_gate_workunit_is_hcli_shaped():
    results = ol.evaluate_launch_criteria()
    wu = ol._gate_workunit(ol.launch_verdict(results))
    ol.wus.validate_emitted_unit(wu)
    assert wu["id"] == "odyssey-i.launch-gate"
    assert wu["classification"] == "REFUSED"
    assert wu["status"] == "completed"
    assert wu["unmet"] == ol.unmet_criteria(results)


def test_receipt_has_no_numeric_hardware_fields():
    out = ol.build()
    doc = json.loads(Path(out["gate_path"]).read_text())

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{path}.{k}" if path else k
                if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                    raise AssertionError(f"hardware field {here}={v!r}")
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)
    assert doc["bench"]["state"] == "UNKNOWN"


def test_hardware_claim_still_raises_through_write_receipt(tmp_path, monkeypatch):
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "ODYSSEY_LAUNCH_GATE_SHOULD_NOT_LAND.json",
            {"schema": "test", "tps": 12.0},
            "tools/future/test_odyssey_launch.py",
        )
    leaked = RECEIPTS / "ODYSSEY_LAUNCH_GATE_SHOULD_NOT_LAND.json"
    if leaked.is_file():
        # write_receipt raises before write; if a prior run left this, do not
        # encode checkout absence — just refuse to treat it as this test's output.
        doc = json.loads(leaked.read_text())
        assert doc.get("tps") != 12.0


def test_siblings_are_exercised_not_merely_named():
    """The no-import rule was a WAVE-TIME constraint, and that wave has landed.

    While the siblings were being written concurrently, importing them would have
    coupled unfinished work. They are now committed, so the honest evaluation is
    to RUN them: a criterion that reports "not landed" about a module sitting on
    disk is stale, not careful. What must still hold is that every import is
    exercised rather than decorative, and that a failed exercise leaves the
    criterion unmet instead of flipping it true.
    """
    src = pathlib.Path(ol.__file__).read_text()
    exercised = {
        line.split('"')[1]
        for line in src.splitlines()
        if "_exercise(" in line and '"' in line
    }
    assert exercised, "no sibling is exercised; the criteria would be measuring absence"
    for dotted in exercised:
        assert dotted.startswith("tools.future."), dotted

    # A failed exercise must NOT be able to open a criterion.
    bad = ol._exercise("tools.future.does_not_exist", "build")
    assert bad["ok"] is False
    assert "import failed" in bad["why"]
    bad2 = ol._exercise("tools.future.frontiers", "no_such_function")
    assert bad2["ok"] is False
    assert "not callable" in bad2["why"]

def test_verify_cli_refuses_and_keeps_phase_not_started():
    rc = ol.verify()
    assert rc == 0
    doc = json.loads((RECEIPTS / ol.RECEIPT).read_text())
    assert doc["verdict"]["verdict"] == "REFUSED"
    assert doc["odyssey_i_launch_written"] is False
    assert doc["phase_transition"] == "NOT_STARTED"
    assert doc["launch_receipt"]["written"] is False


# ---------------------------------------------------------------------------
# Specimen readiness may be EARNED, but it must never be rounded up to.
# ---------------------------------------------------------------------------


def test_specimen_dirs_on_disk_copes_when_the_lake_is_not_mounted(monkeypatch):
    """ModelLake is an external volume. An unmounted lake is not an error."""
    assert isinstance(ol._specimen_dirs_on_disk(), set)


def test_disk_is_authority_over_the_census_cache():
    """A specimen the census never recorded still exists.

    Reading only the census reported Mistral-Small-24B as absent from the
    ModelLake specimens listing while its directory sat in that listing.
    """
    dirs = ol._specimen_dirs_on_disk()
    if not dirs:
        return  # lake not mounted here; the fixtures below carry the guarantees
    idx = ol._lake_index({})
    for name in dirs:
        slug, _, _rev = name.partition("@")
        repo = slug.replace("--", "/", 1)
        assert repo in idx, f"{repo} is on disk but absent from the index"
        assert idx[repo]["in_specimens_listing"] is True


def test_earned_verification_requires_a_real_recomputation(tmp_path, monkeypatch):
    """NEGATIVE CONTROLS: each way of NOT verifying must fail to count."""
    rows = {
        "hashed_nothing": {"specimen": "a", "status": "WHOLE_TREE_VERIFIED",
                           "bytes_hashed": 0, "mismatched": 0,
                           "no_remote_digest": 0, "verified": 3, "n_files": 3},
        "a_file_mismatched": {"specimen": "b", "status": "WHOLE_TREE_VERIFIED",
                              "bytes_hashed": 99, "mismatched": 1,
                              "no_remote_digest": 0, "verified": 3, "n_files": 3},
        "a_file_undigested": {"specimen": "c", "status": "WHOLE_TREE_VERIFIED",
                              "bytes_hashed": 99, "mismatched": 0,
                              "no_remote_digest": 1, "verified": 3, "n_files": 3},
        "counted_more_than_it_checked": {"specimen": "d", "status": "WHOLE_TREE_VERIFIED",
                                         "bytes_hashed": 99, "mismatched": 0,
                                         "no_remote_digest": 0, "verified": 2, "n_files": 3},
        "merely_partial": {"specimen": "e", "status": "PARTIAL_NO_REMOTE_DIGEST",
                           "bytes_hashed": 99, "mismatched": 0,
                           "no_remote_digest": 1, "verified": 2, "n_files": 3},
    }
    good = {"specimen": "ok", "status": "WHOLE_TREE_VERIFIED", "bytes_hashed": 99,
            "mismatched": 0, "no_remote_digest": 0, "verified": 3, "n_files": 3}

    def _probe(doc):
        monkeypatch.setattr(ol, "probe_json", lambda *a, **k: {"found": True, "doc": doc})
        return ol._independently_verified()

    for why, row in rows.items():
        assert _probe({"results": [row]}) == {}, f"a specimen that {why} was counted"
    assert set(_probe({"results": [good]})) == {"ok"}


def test_the_curriculum_never_asserts_an_absence_it_did_not_check():
    """Every unready role must give a reason derived from evidence, not a constant."""
    cur = ol.propose_specimen_curriculum()
    for role in cur["roles"]:
        if role.get("ready"):
            continue
        assert role.get("ready_reason"), f"{role['role']} refused without a reason"
        if "not in the ModelLake specimens listing" in role["ready_reason"]:
            slug = str(role["repo"]).replace("/", "--", 1)
            on_disk = {n.partition("@")[0] for n in ol._specimen_dirs_on_disk()}
            assert slug not in on_disk, (
                f"{role['repo']} was called absent from the listing but is on disk"
            )


# ---------------------------------------------------------------------------
# A tool whose parent is not on this host is not callable by anyone.
# ---------------------------------------------------------------------------


def test_declared_inputs_are_parsed_not_imported(tmp_path):
    """Importing an odyssey tool would run its module-level work inside the gate."""
    rel = "tools/future/_fixture_declared_inputs.py"
    src = ol.REPO / rel
    src.write_text(
        'from pathlib import Path\n'
        'PARENT = Path("/definitely/not/here")\n'
        'RELATIVE = Path("receipts/headless")\n'
        'SIDE_EFFECT = print("this must never run")\n'
    )
    try:
        found = ol._declared_inputs(rel)
    finally:
        src.unlink()
    names = {i["name"] for i in found}
    assert names == {"PARENT"}, "only absolute external inputs count"
    assert found[0]["present"] is False


def test_doctor_and_gravity_blame_the_real_blocker_not_the_runtime():
    """The reason must name what is actually blocking, or it sends work astray."""
    for evaluate in (ol._eval_doctor, ol._eval_gravity):
        row = evaluate()
        missing = row["evidence"][0]["missing_inputs"]
        if not missing:
            continue
        assert row["met"] is False
        assert not row["operational"]["flags"]["invoke"], (
            "a tool whose declared parent is unreachable cannot be invoked"
        )
        for item in missing:
            assert item["path"] in row["reason"], "the blocker is not named in the reason"


def test_a_moved_parent_is_reported_as_stale_not_as_missing():
    """52GB that is already on the disk must never be reported as absent.

    Doctor and Gravity declare /Users/scammermike/models/... which is gone. The
    parent itself is on the external volume. Calling that a missing model sends
    someone to re-download it.
    """
    row = ol._eval_doctor()
    stale = row["evidence"][0]["stale_declared_paths"]
    if not stale:
        return  # nothing moved on this host
    for item in stale:
        assert pathlib.Path(item["resolved_elsewhere"]).is_dir()
        assert pathlib.Path(item["path"]).name == pathlib.Path(item["resolved_elsewhere"]).name
    assert "STALE DECLARED PATH" in row["reason"]
    assert "not a missing model" in row["reason"]


def test_resolution_is_bounded_to_known_model_roots(tmp_path):
    """A find over a 4.5TB volume is not a probe; only named roots are searched."""
    assert ol._resolve_stale_input("/nowhere/at/all/a-name-that-does-not-exist") is None
    for root in ol.MODEL_ROOTS:
        assert root.startswith("/")


def test_negative_control_the_gate_cannot_certify_itself_as_the_driver():
    """odyssey_launch names these tools in an `owned = [...]` literal.

    An earlier version accepted any Assign as evidence of driving and so
    credited this gate module as Doctor's resident driver -- self-certification,
    which the constitution forbids outright.
    """
    sched = ol._resident_schedulable(["tools/odyssey/doctor_tournament.py"])
    assert sched.get("driver_module") != "odyssey_launch.py"
    # A real driver now exists (odyssey_tool_driver.py), so schedule is true --
    # but it must be true because a module DRIVES the tool, never because this
    # gate names it in an `owned = [...]` literal. That is the invariant; the
    # boolean was only ever a proxy for it while no driver existed.
    if sched["schedule"]:
        driver = sched["driver_module"]
        src = (ol.REPO / "tools" / "future" / driver).read_text(errors="replace")
        tree = ast.parse(src)
        in_call = any(
            isinstance(n, ast.Call) and "doctor_tournament.py" in ast.dump(n)
            for n in ast.walk(tree)
        )
        assert in_call, f"{driver} names the tool but never calls it"


def test_a_retired_patient_needs_BOTH_the_prior_seal_and_fresh_verification():
    """Recurrence is earned twice over, or it is a pass on a stale seal.

    The odysseys are recurrent phases and the first completion is historical, so
    a patient retired from that wave is a specimen with a proven role. But
    retirement alone would let a years-old seal stand in for verification, and
    verification alone would throw away the prior work.
    """
    verified_and_sealed = {
        "patient_state": "RETIRED", "patient_seal": "sha256:abc",
        "whole_tree_verified": True, "revision": "r1",
    }
    ready, why = ol._ready(verified_and_sealed, require_lake_verified=True)
    assert ready is True
    assert "RECURRENT_PATIENT" in why

    for missing in ("patient_seal", "whole_tree_verified"):
        row = dict(verified_and_sealed)
        row[missing] = None if missing == "patient_seal" else False
        ready, why = ol._ready(row, require_lake_verified=True)
        assert ready is False, f"a retired patient passed without {missing}"


def test_recurrence_is_labelled_so_it_is_not_read_as_a_first_wave_result():
    cur = ol.propose_specimen_curriculum()
    for role in cur["roles"]:
        if role.get("ready") and "RETIRED" in str(role.get("modellake", {}).get("patient_state", "")):
            assert "RECURRENT_PATIENT" in role["ready_reason"]


def test_a_specimen_under_partial_is_located_not_missing():
    """"Not in the specimens listing" was read as "the model is not here".

    Qwen3-0.6B was complete inside ModelLake the whole time -- ten files, ten
    published digests, zero incomplete markers -- sitting under partial/ rather
    than specimens/. Location was the only thing partial about it. Reporting
    that as a missing model would have sent someone to re-download it, which is
    the same failure as the three Odyssey tools pointing at a moved directory.
    """
    cur = ol.propose_specimen_curriculum()
    role = next(r for r in cur["roles"]
                if r["role"] == "very_small_dense_procedural_speed")
    if not role.get("located_under_partial"):
        return  # not verified on this host; the rule below still holds
    assert role["ready"] is True
    # Readiness still had to be EARNED by recomputation, not granted by location.
    verified = ol._independently_verified()
    row = verified["Qwen--Qwen3-0.6B@c1899de289a0#partial"]
    assert row["verified"] == row["n_files"]
    assert row["mismatched"] == 0 and row["no_remote_digest"] == 0
    assert row["bytes_hashed"] > 0, "readiness claimed without hashing anything"


def test_finding_a_specimen_does_not_lower_the_bar_for_the_others():
    """NEGATIVE CONTROL: every ready role must still name real verification."""
    for role in ol.propose_specimen_curriculum()["roles"]:
        if not role.get("ready"):
            continue
        why = role["ready_reason"]
        assert any(k in why for k in
                   ("whole-tree", "RECURRENT_PATIENT", "tree digest is sealed")), (
            f"{role['role']} was made ready by something other than verification: {why}"
        )
