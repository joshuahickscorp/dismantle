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
