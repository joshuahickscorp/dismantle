"""The completion attack, mechanized.

A campaign fails most often not by producing nothing but by producing something
that LOOKS finished. This module tries to invalidate the sidecar's completion.
It is adversarial on purpose: a green run here is the only thing that earns the
right to say the suite is done.

What it hunts:

* a receipt asserting a hardware number while the sidecar holds no GPU
* a module whose "implementation" is a placeholder (pass / TODO / NotImplemented
  in the load-bearing path)
* a test file with no negative control -- a guard nobody has watched fail
* a test that passes by skipping
* a receipt that is not sealed, or whose seal does not match its content
* a module with no test at all
* a sidecar file that landed outside the sidecar write partition
* a receipt claiming PROMOTED / VERIFIED status it has no authority to claim

    python3 tools/future/integration_attack.py --adversarial
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from tools.future._common import HARDWARE_FIELDS, RECEIPTS, REPO, write_receipt

FUTURE = REPO / "tools" / "future"

# Modules that are infrastructure, not campaign deliverables.
INFRA = {"_common.py", "__init__.py", "integration_attack.py", "handoff.py"}

# A promotion vocabulary the sidecar has no authority to use about its own work.
FORBIDDEN_STATUS = re.compile(
    r"\b(PROMOTED|PROTECTED_PASS|PROTECTED_ABSOLUTE|MEASURED_ON_HARDWARE)\b"
)

PLACEHOLDER = re.compile(
    r"^\s*(pass\s*(#.*)?$|\.\.\.\s*$|raise NotImplementedError|# ?TODO|# ?FIXME|# ?STUB)",
    re.MULTILINE,
)


class Finding(dict):
    pass


def _finding(kind: str, where: str, detail: str, severity: str = "P1") -> Finding:
    return Finding(kind=kind, where=where, detail=detail, severity=severity)


# --------------------------------------------------------------------------
# attacks
# --------------------------------------------------------------------------


def attack_hardware_claims() -> list[Finding]:
    """No receipt may assert a hardware number. The sidecar has no GPU."""
    out = []
    for p in sorted(RECEIPTS.glob("*.json")):
        try:
            doc = json.loads(p.read_text())
        except Exception as e:
            out.append(_finding("unreadable_receipt", str(p), repr(e), "P0"))
            continue

        def walk(node: Any, path: str = "") -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    here = f"{path}.{k}" if path else k
                    if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                        out.append(
                            _finding(
                                "hardware_claim_without_hardware",
                                f"{p.name}:{here}",
                                f"{here} = {v!r}",
                                "P0",
                            )
                        )
                    walk(v, here)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(doc)

        bench = doc.get("bench") or {}
        if bench.get("state") != "UNKNOWN":
            out.append(
                _finding(
                    "bench_state_overridden",
                    p.name,
                    f"bench.state = {bench.get('state')!r}, expected UNKNOWN",
                    "P0",
                )
            )
        if bench.get("gpu_authority") is not False:
            out.append(
                _finding("gpu_authority_claimed", p.name, str(bench.get("gpu_authority")), "P0")
            )
    return out


def attack_seals() -> list[Finding]:
    """A seal that does not match its content is worse than no seal."""
    out = []
    for p in sorted(RECEIPTS.glob("*.json")):
        try:
            doc = json.loads(p.read_text())
        except Exception:
            continue  # already reported by attack_hardware_claims
        claimed = doc.get("seal_sha256")
        if not claimed:
            out.append(_finding("unsealed_receipt", p.name, "no seal_sha256", "P1"))
            continue
        body = {k: v for k, v in doc.items() if k != "seal_sha256"}
        blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        actual = hashlib.sha256(blob).hexdigest()
        if actual != claimed:
            out.append(
                _finding("seal_mismatch", p.name, f"claimed {claimed[:12]} actual {actual[:12]}", "P0")
            )
    return out


def attack_forbidden_status() -> list[Finding]:
    """The sidecar cannot promote anything. It may only describe."""
    out = []
    for p in sorted(RECEIPTS.glob("*.json")):
        text = p.read_text()
        for m in FORBIDDEN_STATUS.finditer(text):
            # Naming the vocabulary in a policy/refusal string is legitimate; asserting
            # it as this artifact's own status is not. Only flag a status-ish key.
            line_start = text.rfind("\n", 0, m.start()) + 1
            line = text[line_start : text.find("\n", m.start())]
            if re.search(r'"(status|verdict|result|outcome)"\s*:', line):
                out.append(
                    _finding("forbidden_promotion_status", f"{p.name}", line.strip()[:160], "P0")
                )
    return out


def attack_placeholders() -> list[Finding]:
    """A module whose load-bearing path is a placeholder is not an implementation."""
    out = []
    for p in sorted(FUTURE.glob("*.py")):
        if p.name in INFRA or p.name.startswith("test_"):
            continue
        src = p.read_text()
        for m in PLACEHOLDER.finditer(src):
            line_no = src.count("\n", 0, m.start()) + 1
            frag = src[m.start() : src.find("\n", m.start())].strip()
            # `pass` inside an exception class body is idiomatic, not a stub.
            ctx_start = max(0, src.rfind("\n", 0, m.start() - 200))
            ctx = src[ctx_start : m.start()]
            if re.search(r"class \w+\((\w|\.)*(Error|Exception)\w*\):\s*$", ctx.strip().split("\n")[-1] if ctx.strip() else ""):
                continue
            out.append(_finding("placeholder_in_module", f"{p.name}:{line_no}", frag[:120], "P1"))
    return out


def attack_missing_tests() -> list[Finding]:
    """Every deliverable module ships a test module."""
    out = []
    for p in sorted(FUTURE.glob("*.py")):
        if p.name in INFRA or p.name.startswith("test_"):
            continue
        if not (FUTURE / f"test_{p.name}").exists():
            out.append(_finding("module_without_test", p.name, f"no test_{p.name}", "P1"))
    return out


def attack_missing_negative_controls() -> list[Finding]:
    """A guard nobody has watched fail is not a guard.

    Every test module must contain at least one test that proves a refusal
    actually fires: pytest.raises, an assertion that a checker returned
    non-zero/False, or an explicitly named negative control.
    """
    out = []
    signals = (
        "pytest.raises",
        "negative_control",
        "negative control",
        "== 1",
        "is False",
        "assert not ",
        "REJECT",
        "refus",
    )
    for p in sorted(FUTURE.glob("test_*.py")):
        src = p.read_text()
        if not any(s in src for s in signals):
            out.append(
                _finding(
                    "test_without_negative_control",
                    p.name,
                    "no refusal/raises assertion found; guard never watched to fail",
                    "P1",
                )
            )
    return out


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def attack_skipped_tests() -> list[Finding]:
    """A suite that passes by skipping is a suite that measured nothing.

    This project has already shipped grades that rested on SKIPPED tests, so an
    unconditional skip in a deliverable test module is a P0.

    Parsed with `ast`, not regex: a test that PROVES skip-detection works has to
    write the string `@pytest.mark.skip` into a fixture, and a text scan flags
    that as a real skip. Carving out the file would leave a blind spot in exactly
    the module whose job is to have none, so match syntax instead of text.
    """
    out = []
    for p in sorted(FUTURE.glob("test_*.py")):
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError as e:
            out.append(_finding("test_unparseable", p.name, repr(e), "P0"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    target = dec.func if isinstance(dec, ast.Call) else dec
                    name = _dotted(target)
                    if name in ("pytest.mark.skip", "pytest.mark.skipif"):
                        out.append(
                            _finding("test_skip", f"{p.name}:{dec.lineno}", f"@{name}", "P0")
                        )
                    elif name == "pytest.mark.xfail":
                        out.append(
                            _finding("test_skip", f"{p.name}:{dec.lineno}", f"@{name}", "P1")
                        )
            elif isinstance(node, ast.Call) and _dotted(node.func) == "pytest.skip":
                out.append(
                    _finding("test_skip", f"{p.name}:{node.lineno}", "pytest.skip()", "P1")
                )
    return out


def attack_write_partition() -> list[Finding]:
    """Nothing a lane produced may live outside the sidecar partition."""
    from tools.future import mutation_surface as ms

    out = []
    if ms.check_disjoint(["tools/future", "receipts/future"]) != 0:
        out.append(
            _finding("write_partition_violated", "tools/future", "intersects Codex surface", "P0")
        )
    return out


ATTACKS = {
    "hardware_claims": attack_hardware_claims,
    "seals": attack_seals,
    "forbidden_status": attack_forbidden_status,
    "placeholders": attack_placeholders,
    "missing_tests": attack_missing_tests,
    "missing_negative_controls": attack_missing_negative_controls,
    "skipped_tests": attack_skipped_tests,
    "write_partition": attack_write_partition,
}


def run() -> dict[str, Any]:
    findings: list[Finding] = []
    per_attack = {}
    for name, fn in ATTACKS.items():
        got = fn()
        per_attack[name] = len(got)
        findings.extend(got)
    p0 = [f for f in findings if f["severity"] == "P0"]
    p1 = [f for f in findings if f["severity"] == "P1"]
    modules = sorted(
        p.name for p in FUTURE.glob("*.py") if p.name not in INFRA and not p.name.startswith("test_")
    )
    return {
        "schema": "hawking.future.integration_attack.v1",
        "version": 1,
        "purpose": "adversarial completion attack over the sidecar; its job is to find a reason NOT to finish",
        "modules_examined": modules,
        "module_count": len(modules),
        "receipts_examined": sorted(p.name for p in RECEIPTS.glob("*.json")),
        "attacks_run": per_attack,
        "counts": {"P0": len(p0), "P1": len(p1), "total": len(findings)},
        "findings": findings,
        "verdict": "CLEAN" if not findings else ("P0_PRESENT" if p0 else "P1_PRESENT"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adversarial", action="store_true", help="run and fail on any P0")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    doc = run()
    out = write_receipt("INTEGRATION_ATTACK.json", doc, "tools/future/integration_attack.py")
    print(out)
    print(f"verdict={doc['verdict']} P0={doc['counts']['P0']} P1={doc['counts']['P1']}")
    for f in doc["findings"][:60]:
        print(f"  [{f['severity']}] {f['kind']}: {f['where']} — {f['detail']}")
    if a.report_only:
        return 0
    return 1 if doc["counts"]["P0"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
