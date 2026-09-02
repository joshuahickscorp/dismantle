"""Tests for tools/audit/reachability_triage.py.

Hermetic: decide() never leaves a row undispositioned, receipts/ are not
callers, own-test-only is not CONNECTED.

Live: the triage tool emits an inventory in which every module carries a
disposition, and count(undispositioned)==0. Spot-checks 5 UNREACHABLE and
5 CONNECTED rows against git grep.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.audit import reachability_triage as rt
from tools.future._common import RECEIPTS, load_json
from tools.future import capability_reachability as cr


# --------------------------------------------------------------------------
# Hermetic classification
# --------------------------------------------------------------------------


def _row(**overrides):
    base = {
        "module": "tools/future/widget.py",
        "callable_outside_tests": False,
        "hcli_reachable": False,
        "is_stub": False,
        "is_package_marker": False,
        "retired": None,
        "has_test": False,
        "test_exercises": False,
        "test_reads_receipt_only": False,
        "orchestration_bound": False,
    }
    base.update(overrides)
    rt.decide(base)
    return base


def test_decide_always_sets_a_known_disposition():
    cases = [
        {},
        {"hcli_reachable": True, "callable_outside_tests": True},
        {"callable_outside_tests": True},
        {"is_stub": True},
        {"has_test": True, "test_exercises": True},
        {"has_test": True, "test_reads_receipt_only": True},
        {"retired": "superseded by", "callable_outside_tests": False},
        {"orchestration_bound": True},
        {"is_stub": True, "is_package_marker": True, "hcli_reachable": True},
    ]
    for kwargs in cases:
        row = _row(**kwargs)
        assert row["disposition"] in rt.DISPOSITIONS, kwargs
        assert row["classification"] in rt.CLASSIFICATIONS, kwargs
        assert row["disposition_full"]


def test_hcli_reachable_is_connected():
    row = _row(hcli_reachable=True, callable_outside_tests=True)
    assert row["disposition"] == "CONNECTED"
    assert row["classification"] == "BUILT"


def test_sidecar_only_caller_is_parked_dormant():
    row = _row(callable_outside_tests=True, hcli_reachable=False)
    assert row["disposition"] == "PARKED"
    assert row["classification"] == "DORMANT"
    assert "HCLI" in row["wake_condition"]


def test_nothing_calls_it_is_unreachable_parked():
    row = _row(has_test=True, test_exercises=True)
    assert row["classification"] == "UNREACHABLE"
    assert row["disposition"] == "PARKED"
    assert "first production importer" in row["wake_condition"]


def test_stub_with_no_test_is_archive_candidate():
    row = _row(is_stub=True, has_test=False)
    assert row["disposition"] == "ARCHIVE_CANDIDATE"
    assert row["classification"] == "SCAFFOLDED"
    assert row["archive_reason"]


def test_retired_uncalled_is_archive_candidate():
    row = _row(retired="this module is retired")
    assert row["disposition"] == "ARCHIVE_CANDIDATE"
    assert row["classification"] == "ARCHIVE_CANDIDATE"


def test_mentioning_someone_elses_retirement_is_not_self_retired():
    """A docstring that says a campaign/driver was retired is not self-archive."""
    text = (
        '"""Retire the Ramanujan campaign by indexing it.\n\n'
        "The campaign is retired; its evidence is not. This script makes that durable.\n"
        '"""\n'
    )
    shape = rt.module_shape(text, "ramanujan_disband.py")
    assert shape["retired"] is None

    text2 = (
        '"""git commit takes the index.\n\n'
        "The legacy Odyssey driver was retired for ending each window with a bare commit.\n"
        '"""\n'
    )
    assert rt.module_shape(text2, "index_provenance.py")["retired"] is None

    text3 = (
        '"""This module is retired; use tools/future/wakeup.py instead.\n"""\n'
    )
    assert rt.module_shape(text3, "old.py")["retired"]


def test_data_path_helper():
    assert rt.is_data_path(Path("receipts/future/nowait/_bound_run.py"))
    assert rt.is_data_path(Path("hcli/fixtures/x.py"))
    assert not rt.is_data_path(Path("hcli/agentos/native_gate.py"))
    assert not rt.is_production_path(Path("tools/future/test_widget.py"))
    assert rt.is_production_path(Path("hcli/agentos/native_gate.py"))


def test_partition_drops_receipts_as_callers(tmp_path, monkeypatch):
    """A generated runner under receipts/ must not make a module callable."""
    rt.install_engine_extensions()
    monkeypatch.setattr(cr, "REPO", tmp_path)
    cr._TEXT_CACHE.clear()
    rt._INDEX_CACHE.clear()

    widget = tmp_path / "tools" / "future" / "widget.py"
    widget.parent.mkdir(parents=True)
    widget.write_text("def gadget():\n    return 1\n")
    own_test = tmp_path / "tools" / "future" / "test_widget.py"
    own_test.write_text(
        "from tools.future.widget import gadget\n\n"
        "def test_gadget():\n    assert gadget() == 1\n"
    )
    receipt_runner = tmp_path / "receipts" / "future" / "_bound_run.py"
    receipt_runner.parent.mkdir(parents=True)
    receipt_runner.write_text(
        "from tools.future.widget import gadget\n\n"
        "def run():\n    return gadget()\n"
    )
    files = [widget, own_test, receipt_runner]
    idx = cr.build_repo_index(files=files)
    sites = cr.find_module_import_sites(idx, "tools.future.widget", exclude_files=(widget,))
    cap = cr.build_capability(
        "tools.future.widget",
        "module",
        defined=True,
        registered=False,
        resident_visible=False,
        sites=sites,
    )
    assert cap["tested"] is True
    assert cap["callable"] is False, (
        "receipts/ import must not count as a production call site"
    )
    assert cap["call_sites"] == []


def test_real_outside_caller_still_counts(tmp_path, monkeypatch):
    rt.install_engine_extensions()
    monkeypatch.setattr(cr, "REPO", tmp_path)
    cr._TEXT_CACHE.clear()
    rt._INDEX_CACHE.clear()

    widget = tmp_path / "tools" / "future" / "widget.py"
    widget.parent.mkdir(parents=True)
    widget.write_text("def gadget():\n    return 1\n")
    caller = tmp_path / "hcli" / "gate.py"
    caller.parent.mkdir(parents=True)
    caller.write_text(
        "from tools.future.widget import gadget\n\n"
        "def use():\n    return gadget()\n"
    )
    idx = cr.build_repo_index(files=[widget, caller])
    sites = cr.find_module_import_sites(idx, "tools.future.widget", exclude_files=(widget,))
    cap = cr.build_capability(
        "tools.future.widget",
        "module",
        defined=True,
        registered=False,
        resident_visible=None,
        sites=sites,
    )
    assert cap["callable"] is True
    assert cap["call_sites"][0]["file"] == "hcli/gate.py"


# --------------------------------------------------------------------------
# Live inventory
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_doc():
    out = rt.selftest()
    return load_json(out)


def test_live_inventory_schema_and_zero_undisposed(live_doc):
    assert live_doc["schema"] == rt.SCHEMA
    assert live_doc["seal_sha256"]
    assert live_doc["evidence_tier"] == "STATIC"
    assert live_doc["engine"]["assemble_used"] is True
    counts = live_doc["counts"]
    assert counts["modules"] > 0
    assert counts["undispositioned"] == 0
    assert live_doc["undispositioned"] == []
    modules = live_doc["modules"]
    assert len(modules) == counts["modules"]
    for name, row in modules.items():
        assert row["disposition"] in rt.DISPOSITIONS, name
        assert row["classification"] in rt.CLASSIFICATIONS, name
        assert row["evidence_tier"] == "STATIC", name


def test_live_status_causality_is_connected_from_hcli_gates(live_doc):
    """The brief's worked example: tools.future is not uniformly dead.

    status_causality is imported by the HCLI gates. A blanket 'tools/future
    is dead' or 'tools/future is wired' fails this row.
    """
    row = live_doc["modules"]["tools/future/status_causality.py"]
    assert row["disposition"] == "CONNECTED"
    assert row["hcli_reachable"] is True
    files = {s["file"] for s in row["call_sites"]}
    assert any(f.startswith("hcli/agentos/") and f.endswith("_gate.py") for f in files), (
        f"expected an hcli/agentos/*_gate.py call site, got {sorted(files)}"
    )


def test_live_state_transitions_have_file_line_evidence(live_doc):
    findings = live_doc["state_transitions"]
    assert findings
    missing = [t for t in findings if t["status"] == "EXIT_MISSING"]
    assert missing, "the SEALED_SOURCE_READY / resident-child findings must surface"
    for t in findings:
        assert t["file"]
        assert t["enter"]
        assert t["status"] in {"EXIT_EXISTS", "EXIT_MISSING"}
        assert t["evidence_tier"] == "STATIC"


def test_live_spotcheck_five_unreachable_and_five_connected(live_doc):
    """Mandatory: 5 UNREACHABLE + 5 CONNECTED, each verified with git grep -nE."""
    spot = rt.run_spotchecks(live_doc)
    assert len(spot["unreachable"]) == 5
    assert len(spot["connected"]) == 5
    assert len(spot["checks"]) == 10
    by_verdict = {k: 0 for k in ("UNREACHABLE", "CONNECTED")}
    for check in spot["checks"]:
        by_verdict[check["verdict"]] += 1
        assert check["command"].startswith("git --no-optional-locks grep -nE")
        if check["verdict"] == "UNREACHABLE":
            assert check["production_import_hits"] == []
        else:
            assert check["hcli_path"] or check["hcli_hits"]
    assert by_verdict == {"UNREACHABLE": 5, "CONNECTED": 5}
    written = load_json(RECEIPTS / "REACHABILITY_TRIAGE_SPOTCHECKS.json")
    assert written["schema"] == "hawking.audit.reachability_triage_spotchecks.v1"
    assert len(written["checks"]) == 10
