"""Graph shape + adversarial auditor tests.

A validator nobody has watched refuse is decoration. The mutation check is the
load-bearing one: a BUILT gate must drop when its production call site is removed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.roadmap import ALLOWED_STATUSES, GRAPH_REL
from tools.roadmap import catalog
from tools.roadmap.auditor import audit
from tools.roadmap.gitfs import REPO, SourceView
from tools.roadmap.parse import parse_roadmap
from tools.roadmap.__main__ import mutation_check

ROADMAP = Path("/Users/scammermike/Downloads/H-ROADMAP.md")


@pytest.fixture(scope="module")
def parsed():
    return parse_roadmap(ROADMAP)


@pytest.fixture(scope="module")
def graph():
    return audit(include_assemble=False)


def test_graph_contains_at_least_71_gates_and_25_genes_with_source_spans(parsed, graph):
    assert len(parsed["gates"]) >= 71
    assert len(parsed["genes"]) >= 25
    assert len(graph["gates"]) >= 71
    assert len(graph["genes"]) >= 25
    for entry in list(graph["gates"].values()) + list(graph["genes"].values()):
        span = entry.get("source_span") or {}
        assert span.get("file"), f"{entry.get('id')} missing source_span.file"
        assert isinstance(span.get("start_line"), int) and span["start_line"] > 0
        assert isinstance(span.get("end_line"), int) and span["end_line"] >= span["start_line"]


def test_auditor_emits_status_for_every_gate_and_gene(graph):
    for name, row in graph["gates"].items():
        assert row["status"] in ALLOWED_STATUSES, f"{name} status {row['status']!r}"
    for name, row in graph["genes"].items():
        assert row["status"] in ALLOWED_STATUSES, f"{name} status {row['status']!r}"
    assert len(graph["gates"]) == 71
    assert len(graph["genes"]) == 25


def test_every_non_absent_verdict_cites_evidence(graph):
    for name, row in {**graph["gates"], **graph["genes"]}.items():
        if row["status"] == "ABSENT":
            continue
        refs = row.get("evidence_refs") or []
        assert refs, f"{name} status={row['status']} has empty evidence_refs"
        for ref in refs:
            assert ref.get("kind"), f"{name} evidence missing kind"
            assert (
                ref.get("file")
                or ref.get("command")
                or ref.get("note")
            ), f"{name} evidence has no file/command/note"


def test_all_13_blocked_hardware_gates_have_wake_condition(graph):
    blocked = [g for g in graph["gates"].values() if g["status"] == "BLOCKED_HARDWARE"]
    assert len(blocked) == 13, (
        f"expected 13 BLOCKED_HARDWARE gates, got {len(blocked)}: "
        + ",".join(sorted(g["id"] for g in blocked))
    )
    for g in blocked:
        wake = g.get("wake_condition")
        assert isinstance(wake, str) and wake.strip(), f"{g['id']} missing wake_condition"
        assert wake == wake.upper() and "_" in wake, f"{g['id']} wake {wake!r} is not a machine id"


def test_built_gates_have_a_non_test_call_site(graph):
    built = [g for g in graph["gates"].values() if g["status"] == "BUILT"]
    assert built, "auditor rated nothing BUILT; catalog/look-up is too timid or the tree is empty"
    for g in built:
        callers = g.get("runtime_caller") or []
        assert callers, f"{g['id']} is BUILT with no runtime_caller"
        kinds = {site.get("kind") for site in callers}
        assert kinds & {"call", "subprocess"}, (
            f"{g['id']} is BUILT but runtime_caller has no call/subprocess: {kinds}"
        )
        assert not kinds <= {"import"}, (
            f"{g['id']} is BUILT on exclusively import-kind runtime_caller"
        )
        for site in callers:
            from tools.roadmap.reach import is_test_path

            assert not is_test_path(site["file"]), f"{g['id']} caller {site['file']} is a test"
            assert site.get("kind") != "import", (
                f"{g['id']} runtime_caller includes kind=import at {site['file']}:{site['line']}"
            )
            assert site.get("kind") != "weak_signal", (
                f"{g['id']} runtime_caller includes a weak_signal at {site['file']}:{site['line']}"
            )


def test_no_built_gate_is_import_only(graph):
    for g in graph["gates"].values():
        if g["status"] != "BUILT":
            continue
        callers = g.get("runtime_caller") or []
        kinds = [c.get("kind") for c in callers]
        assert kinds, f"{g['id']} BUILT with empty runtime_caller"
        assert any(k in {"call", "subprocess"} for k in kinds), (
            f"{g['id']} BUILT without a call/subprocess citation: {kinds}"
        )
        assert not all(k == "import" for k in kinds), (
            f"{g['id']} BUILT with exclusively import-kind runtime_caller"
        )


def test_runtime_caller_contains_only_invocations(graph):
    allowed = {"call", "subprocess"}
    for g in graph["gates"].values():
        for site in g.get("runtime_caller") or []:
            assert site.get("kind") in allowed, (
                f"{g['id']} runtime_caller has non-invocation kind {site}"
            )


def test_no_two_built_gates_share_identical_runtime_caller(graph):
    built = [g for g in graph["gates"].values() if g["status"] == "BUILT"]
    groups: dict[str, list[str]] = {}
    for g in built:
        key = json.dumps(g.get("runtime_caller") or [], sort_keys=True)
        groups.setdefault(key, []).append(g["id"])
    collisions = {tuple(v) for v in groups.values() if len(v) > 1}
    assert not collisions, (
        "BUILT gates share a byte-identical runtime_caller list (import-as-call "
        f"regression): {sorted(collisions)}"
    )


def test_graph_keeps_71_gates_25_genes_with_source_spans(graph):
    assert len(graph["gates"]) == 71
    assert len(graph["genes"]) == 25
    for entry in list(graph["gates"].values()) + list(graph["genes"].values()):
        span = entry.get("source_span") or {}
        assert span.get("file"), f"{entry.get('id')} missing source_span.file"
        assert isinstance(span.get("start_line"), int) and span["start_line"] > 0
        assert isinstance(span.get("end_line"), int) and span["end_line"] >= span["start_line"]


def test_mutation_downgrades_a_built_gate():
    result = mutation_check()
    assert result["before_status"] == "BUILT"
    assert result["after_status"] != "BUILT", (
        f"{result['gate']} stayed BUILT after overlaying {result['mutated_files']}; "
        "the auditor is not adversarial"
    )
    assert result["downgraded"] is True
    assert result["after_status"] in ALLOWED_STATUSES
    # Leave the result on stdout so the required evidence paste is a real run.
    print("MUTATION_BEFORE", json.dumps({
        "gate": result["gate"],
        "status": result["before_status"],
        "callers": result["before_callers"],
        "counts": result["before_counts"],
    }, indent=2))
    print("MUTATION_AFTER", json.dumps({
        "gate": result["gate"],
        "status": result["after_status"],
        "callers": result["after_callers"],
        "counts": result["after_counts"],
        "mutated_files": result["mutated_files"],
    }, indent=2))


def test_disk_truth_modules_are_present_in_git(graph):
    rows = {r["path"]: r for r in graph["disk_truth_modules"]}
    for path in catalog.DISK_TRUTH_MODULES:
        assert rows[path]["present_in_git"] is True, f"disk-truth module missing from git: {path}"


def test_theia_is_absent_from_hawking_tree(graph):
    claims = {c["claim"]: c for c in graph["verified_absent"]}
    assert claims["theia"]["verdict"] == "ABSENT"


def test_catalog_covers_every_appendix_o_gate(parsed):
    missing = [n for n in parsed["gates"] if n not in catalog.GATES]
    extra = [n for n in catalog.GATES if n not in parsed["gates"]]
    assert missing == [], f"catalog missing gates {missing}"
    assert extra == [], f"catalog has unknown gates {extra}"


def test_capability_reachability_assemble_is_importable():
    from tools.future.capability_reachability import assemble, build_repo_index, find_module_import_sites

    assert callable(assemble)
    assert callable(build_repo_index)
    assert callable(find_module_import_sites)


def test_no_status_is_hand_written_in_the_catalog():
    blob = Path(__file__).with_name("catalog.py").read_text()
    for status in ALLOWED_STATUSES:
        assert f'"{status}"' not in blob, f"catalog.py contains a hand-written status {status}"
        assert f"'{status}'" not in blob


def test_import_alone_cannot_justify_built():
    from tools.roadmap.auditor import _local_status

    look = {
        "defined": True,
        "defined_refs": [{"file": "hcli/scheduler.py", "line": 1, "kind": "definition"}],
        "missing_paths": [],
        "runtime_caller": [],
        "import_sites": [
            {"file": "hcli/agentos/__init__.py", "line": 17, "kind": "import"},
            {"file": "hcli/mission.py", "line": 29, "kind": "import"},
        ],
        "weak_signals": [],
        "tests": [],
        "receipts": [],
    }
    status, evidence, _hw, _sw = _local_status(
        era="I", look=look, hw_id=None, hw_probe=None, ext=None
    )
    assert status == "SCAFFOLDED", status
    assert all(e.get("kind") != "call" for e in evidence if e.get("kind") == "import")
    assert any(e.get("kind") == "import" for e in evidence)


def test_call_of_implementing_symbol_justifies_built():
    from tools.roadmap.auditor import _local_status

    look = {
        "defined": True,
        "defined_refs": [
            {"file": "hcli/scheduler.py", "line": 72, "kind": "symbol", "note": "Scheduler"}
        ],
        "missing_paths": [],
        "runtime_caller": [
            {"file": "hcli/mission.py", "line": 245, "kind": "call", "symbol": "Scheduler"}
        ],
        "import_sites": [
            {"file": "hcli/mission.py", "line": 29, "kind": "import"},
        ],
        "weak_signals": [],
        "tests": [],
        "receipts": [],
    }
    status, evidence, _hw, _sw = _local_status(
        era="I", look=look, hw_id=None, hw_probe=None, ext=None
    )
    assert status == "BUILT", status
    assert any(e.get("kind") == "call" for e in evidence)


def test_weak_signal_never_moves_status():
    from tools.roadmap.auditor import _local_status

    look = {
        "defined": True,
        "defined_refs": [{"file": "hcli/scheduler.py", "line": 1, "kind": "definition"}],
        "missing_paths": [],
        "runtime_caller": [],
        "import_sites": [],
        "weak_signals": [
            {
                "file": "hcli/scheduler.py",
                "line": 55,
                "kind": "weak_signal",
                "symbol": "NO_PROGRESS",
                "note": "name-only assignment",
            }
        ],
        "tests": [],
        "receipts": [],
    }
    status, _evidence, _hw, _sw = _local_status(
        era="I", look=look, hw_id=None, hw_probe=None, ext=None
    )
    assert status == "SCAFFOLDED", status


def test_classify_symbol_rejects_assignments():
    from tools.roadmap.gitfs import classify_symbol, definition_line

    text = "NO_PROGRESS = 3\n\ndef verification_passed(outcome):\n    return True\n"
    kind, line = classify_symbol(text, "NO_PROGRESS")
    assert kind == "assignment"
    assert line == 1
    assert definition_line(text, "NO_PROGRESS") is None
    kind, line = classify_symbol(text, "verification_passed")
    assert kind == "function"
    assert definition_line(text, "verification_passed") == line


def test_classify_symbol_accepts_exception_class():
    from tools.roadmap.gitfs import classify_symbol, definition_line

    text = "class NO_PROGRESS(Exception):\n    pass\n"
    kind, line = classify_symbol(text, "NO_PROGRESS")
    assert kind == "class"
    assert definition_line(text, "NO_PROGRESS") == line


def test_exact_cli_path_rejects_suffix_of_another_tree():
    from tools.roadmap.reach import is_exact_cli_path

    assert is_exact_cli_path("hcli/scheduler.py", "hcli/scheduler.py")
    assert is_exact_cli_path("hcli/scheduler.py", "./hcli/scheduler.py")
    assert not is_exact_cli_path(
        "hcli/scheduler.py", "tools/haider/hcli/scheduler.py"
    )
    assert not is_exact_cli_path("hcli/scheduler.py", "MAX_REPAIR_DEPTH")
