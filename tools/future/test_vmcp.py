"""Tests for the VMCP compact surface and organ disposition.

CONNECTED acts (see/hold/know/check/prove) must run end-to-end on a real
file. PARKED acts must carry a non-empty wake and must not look like an
empty success. An import of tools.future.vmcp is not a call site — the
capability-manifest adapter must AST-Call compact_surface.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from tools.audit import reachability_triage as rt
from tools.future import vmcp as vm
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


ADAPTER = Path(rt.__file__)
VMCP_CALL = "result = compact_surface(act, arguments)"
VMCP_MUTATION = 'result = {"status": "UNREACHABLE_MUTATION", "empty_success": True}'


def _call_sites(path: Path, module: str, symbol: str) -> list[int]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    binds: dict[str, tuple[str, str | None]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                binds[alias.asname or alias.name] = (node.module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                binds[alias.asname or alias.name.split(".")[0]] = (alias.name, None)
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        target = None
        if isinstance(func, ast.Name) and func.id in binds:
            mod, name = binds[func.id]
            target = f"{mod}.{name or func.id}"
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id in binds:
                mod, name = binds[func.value.id]
                target = f"{mod}.{func.attr}" if name is None else f"{mod}.{name}.{func.attr}"
        if target == f"{module}.{symbol}":
            lines.append(node.lineno)
    return lines


def test_vmcp_nine_acts_are_disposed_connected_or_parked():
    doc = vm.disposition()
    assert doc["schema"] == vm.DISPOSITION_SCHEMA
    assert doc["subsystem"] == "vmcp"
    acts = {row["act"]: row for row in doc["acts"]}
    assert set(acts) == set(vm.NINE_ACTS)
    for act in vm.CONNECTED_ACTS:
        assert acts[act]["disposition"] == "CONNECTED"
        assert acts[act]["wake"] is None
        assert acts[act]["empty_success"] is False
    for act in vm.PARKED_ACTS:
        assert acts[act]["disposition"] == "PARKED"
        wake = acts[act]["wake"]
        assert isinstance(wake, dict)
        assert wake["required_kind"] == "call"
        assert wake["predicate"]
        assert wake["missing_dependency"]
        assert wake["schema"] == vm.WAKE_SCHEMA
        assert acts[act]["looked"] is False
        assert acts[act]["empty_success"] is False
    organs = {row["id"]: row for row in doc["organs"]}
    assert organs["vmcp.file_eye"]["disposition"] == "CONNECTED"
    for parked_id in (
        "vmcp.capture_bus",
        "vmcp.web_eye",
        "vmcp.spatial_eye",
        "vmcp.visual_proof",
        "vmcp.pty_eye",
        "vmcp.behavior_lab",
        "vmcp.tool_doctor",
        "vmcp.laboratory_profile",
        "vmcp.app_generation",
    ):
        assert organs[parked_id]["disposition"] == "PARKED"
        assert organs[parked_id]["wake"]["predicate"]
        assert organs[parked_id]["wake"]["missing_dependency"]
    for row in doc["lattice"]:
        assert row["disposition"] == "PARKED"
        assert row["wake"]["predicate"]
    assert "hawking-perception" in doc["not_this"]
    assert doc["gpu_authority"] is False
    _assert_no_hardware_claims(doc)


def test_vmcp_see_hold_check_prove_on_a_real_file(tmp_path: Path):
    subject = tmp_path / "subject.txt"
    payload = b"the claim under observation\n"
    subject.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    seen = vm.see(subject)
    assert seen["status"] == "CONNECTED"
    assert seen["present"] is True
    assert seen["sha256"] == digest
    assert seen["empty_success"] is False
    assert seen["looked"] is True
    assert seen["evidence_tier"] == "FUNCTIONAL_SIM"
    assert "present" in seen["matches_file_eye_fields"]

    held = vm.hold(subject)
    assert held["bound"] is True
    assert held["asset_id"] == f"sha256:{digest}"

    known = vm.know(subject)
    assert known["identity_kind"] == "content_sha256"
    assert known["content_identity"] == "sha256"

    ok = vm.check(subject, arguments={"expected_sha256": digest})
    assert ok["ok"] is True
    stale = vm.check(subject, arguments={"expected_sha256": "0" * 64})
    assert stale["ok"] is False
    assert stale["reason"] == "EVIDENCE_STALE"

    proof = vm.prove(subject)
    assert proof["red"] is True
    assert proof["green"] is True
    assert proof["ok"] is True
    assert proof["baseline_sha256"] != proof["mutated_sha256"]
    assert proof["restored_sha256"] == proof["baseline_sha256"]
    # The caller's file is not the mutation surface.
    assert subject.read_bytes() == payload


def test_vmcp_absent_file_is_target_absent_not_empty_success(tmp_path: Path):
    missing = vm.see(tmp_path / "nope.bin")
    assert missing["present"] is False
    assert "TARGET_ABSENT" in missing["limitations"]
    assert missing["empty_success"] is False
    assert missing["sha256"] is None
    assert missing["looked"] is True


def test_vmcp_know_distinguishes_content_identity_from_subject(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("same-bytes\n", encoding="utf-8")
    b.write_text("same-bytes\n", encoding="utf-8")
    known = vm.know(a, arguments={"other_path": str(b)})
    assert known["same_bytes"] is True
    assert known["same_subject"] is False
    assert known["subject_identity"] == "path"


def test_vmcp_parked_acts_never_look_like_empty_success():
    for act in vm.PARKED_ACTS:
        out = vm.compact_surface(act, {})
        assert out["status"] == "PARKED", act
        assert out["looked"] is False, act
        assert out["empty_success"] is False, act
        assert out.get("results") is None, act
        assert out.get("items") is None, act
        wake = out["wake"]
        assert wake["required_kind"] == "call", act
        assert wake["predicate"], act
        assert wake["missing_dependency"], act
        assert wake["required_symbol"], act


def test_vmcp_selftest_and_build_receipt():
    proof = vm.selftest()
    assert proof["ok"] is True
    assert proof["n_connected_acts"] == len(vm.CONNECTED_ACTS)
    assert proof["n_parked_acts"] == len(vm.PARKED_ACTS)
    out = vm.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "VMCP_DISPOSITION.json"
    assert doc["schema"] == vm.SCHEMA
    assert doc["status"] == "BUILT_NOT_PROMOTED"
    assert doc["promoted"] is False
    assert doc["gpu_authority"] is False
    assert doc["disposition"]["subsystem"] == "vmcp"
    _assert_no_hardware_claims(doc)


def test_vmcp_invoke_via_capability_manifest(tmp_path: Path):
    called = rt.adapter_called_symbols()
    assert ("tools.future.vmcp", "compact_surface") in called
    assert rt.wired_status("future.vmcp") == "CALLABLE"

    subject = tmp_path / "wired.txt"
    subject.write_text("wired-vmcp\n", encoding="utf-8")
    digest = hashlib.sha256(b"wired-vmcp\n").hexdigest()
    out = rt.handle(
        "capability.invoke",
        {"id": "future.vmcp", "arguments": {"act": "see", "path": str(subject)}},
    )
    assert out["ok"] is True, out
    result = out["value"]["result"]
    assert result["sha256"] == digest
    assert result["present"] is True
    assert out["value"]["symbol"] == "compact_surface"

    parked = rt.handle(
        "capability.invoke",
        {"id": "future.vmcp", "arguments": {"act": "make"}},
    )
    assert parked["ok"] is True, parked
    body = parked["value"]["result"]
    assert body["status"] == "PARKED"
    assert body["wake"]["predicate"]
    assert body["empty_success"] is False

    disp = rt.handle(
        "capability.invoke",
        {"id": "future.vmcp", "arguments": {"act": "disposition"}},
    )
    assert disp["ok"] is True, disp
    assert disp["value"]["result"]["result"]["subsystem"] == "vmcp"


def test_vmcp_wired_call_mutation_reports_unreachable():
    original = ADAPTER.read_text(encoding="utf-8")
    assert VMCP_CALL in original
    assert VMCP_MUTATION not in original
    before = rt.wired_status("future.vmcp", source=original)
    assert before == "CALLABLE"
    mutated = original.replace(VMCP_CALL, VMCP_MUTATION, 1)
    assert VMCP_CALL not in mutated
    try:
        ADAPTER.write_text(mutated, encoding="utf-8")
        after = rt.wired_status("future.vmcp", source=mutated)
        live = rt.wired_status("future.vmcp")
        assert after == "UNREACHABLE"
        assert live == "UNREACHABLE"
        assert "from tools.future.vmcp import compact_surface" in mutated
    finally:
        ADAPTER.write_text(original, encoding="utf-8")
    restored = ADAPTER.read_text(encoding="utf-8")
    assert VMCP_CALL in restored
    assert VMCP_MUTATION not in restored
    assert rt.wired_status("future.vmcp") == "CALLABLE"


def test_vmcp_adapter_is_a_kind_call_of_compact_surface():
    lines = _call_sites(ADAPTER, "tools.future.vmcp", "compact_surface")
    assert lines, "reachability_triage.py must Call compact_surface (import is not a call)"


def test_vmcp_locate_records_path_taken_without_claiming_absence():
    located = vm.locate_visionmcp()
    assert "found" in located
    assert located["evidence_tier"] == "STATIC"
    assert located["in_this_repo_HEAD"] is False or isinstance(located["in_this_repo_HEAD"], bool)
    assert located["candidates"]
    # Sparse / missing is recorded, never used as a proof the package is gone.
    assert "foreign package" in located["note"]
