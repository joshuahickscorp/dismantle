"""Program-level AgentOS checkpoint and maturity census.

This is deliberately a census, not a success stamp.  It records which
control-plane surfaces and receipts exist, separates fixture proof from
production qualification, and leaves explicit blockers for work that has not
been measured.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hcli.persist import atomic_write_json
from hcli.providers import (
    CAPABILITY_SCHEMA,
    FAILURE_SCHEMA,
    GENERATION_REQUEST_SCHEMA,
    GENERATION_RESPONSE_SCHEMA,
    HEALTH_SCHEMA,
    PROFILE_SCHEMA,
    RECEIPT_SCHEMA,
    ROLE_SCHEMA,
)
from hcli.tool_registry import default_tool_registry


SCHEMA = "hcli.agentos.program_checkpoint.v1"
DEFAULT_NAME = "HCLI_AGENTOS_CHECKPOINT.json"


def _read_object(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _git_revision(repo_root: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = (proc.stdout or "").strip()
    return value if proc.returncode == 0 and value else None


def _receipt_inventory(roots: Iterable[Path], *, limit: int = 200) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if len(rows) >= limit:
                break
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            try:
                rows.append({
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "suffix": path.suffix.lower(),
                })
            except OSError:
                continue
    return rows


def _gate_summary(workspace: Path, repo_root: Path) -> Dict[str, Any]:
    recovery_candidates = [
        workspace / ".hcli" / "receipts" / "recovery-gate.json",
        repo_root / "receipts" / "headless" / "recovery-gate.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_RECOVERY_GATE.json",
    ]
    research_candidates = [
        workspace / ".hcli" / "receipts" / "research-gate.json",
        repo_root / "receipts" / "headless" / "research-gate.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_RESEARCH_GATE.json",
    ]
    vmcp_candidates = [
        workspace / ".hcli" / "receipts" / "vmcp-gate.json",
        repo_root / "receipts" / "headless" / "vmcp-gate.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_VMCP_GATE.json",
    ]
    native_candidates = [
        workspace / ".hcli" / "receipts" / "native-gate.json",
        repo_root / "receipts" / "headless" / "native-gate.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_NATIVE_GATE.json",
    ]
    resident_candidates = [
        workspace / ".hcli" / "receipts" / "resident-gate.json",
        repo_root / "receipts" / "headless" / "resident-gate.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_RESIDENT_GATE.json",
    ]
    mission_candidates = [
        workspace / ".hcli" / "receipts" / "native-mission-gate.json",
        repo_root / "receipts" / "headless" / "native-mission-gate.json",
        repo_root / "receipts" / "headless" / "HCLI_NATIVE_MISSION_GATE.json",
    ]
    accelerator_candidates = [
        workspace / ".hcli" / "receipts" / "accelerator-native-smoke.json",
        repo_root / "receipts" / "headless" / "accelerator-native-smoke.json",
        repo_root / "receipts" / "headless" / "HCLI_ACCELERATOR_NATIVE_SMOKE.json",
    ]
    recovery = None
    for path in recovery_candidates:
        value = _read_object(path)
        if value is not None:
            recovery = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
            }
            break
    research = None
    for path in research_candidates:
        value = _read_object(path)
        if value is not None:
            research = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
            }
            break
    vmcp = None
    for path in vmcp_candidates:
        value = _read_object(path)
        if value is not None:
            vmcp = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
            }
            break
    native = None
    for path in native_candidates:
        value = _read_object(path)
        if value is not None:
            native = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "profile_path": value.get("profile_path"),
            }
            break
    resident = None
    for path in resident_candidates:
        value = _read_object(path)
        if value is not None:
            resident = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "profile_path": value.get("profile_path"),
            }
            break
    mission = None
    for path in mission_candidates:
        value = _read_object(path)
        if value is not None:
            mission = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "profile_path": value.get("profile_path"),
            }
            break
    accelerator = None
    for path in accelerator_candidates:
        value = _read_object(path)
        if value is not None:
            accelerator = {
                "status": "PASSED" if value.get("pass") is True else "FAILED",
                "qualification": "LIVE_ACCELERATOR_EXECUTION_NO_PERF_CLAIM",
                "receipt_path": str(path),
                "pass": value.get("pass"),
                "bench_state": (value.get("bench") or {}).get("state")
                if isinstance(value.get("bench"), dict) else None,
            }
            break
    return {
        "recovery_gate": {
            **(recovery or {
                "status": "NOT_RUN",
                "qualification": "NONE",
                "receipt_path": None,
                "checks": {},
            }),
        },
        "research_gate": research or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
        },
        "vmcp_gate": vmcp or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
        },
        "native_gate": native or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "profile_path": None,
        },
        "resident_gate": resident or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "profile_path": None,
        },
        "native_mission_gate": mission or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "profile_path": None,
        },
        "accelerator_smoke": accelerator or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "pass": None,
            "bench_state": None,
        },
        "unattended": "NOT_PROVEN",
        "production_provider_gate": "NOT_RUN",
    }


def build_program_checkpoint(
    repo_root: Optional[str | os.PathLike[str]] = None,
    *,
    workspace: Optional[str | os.PathLike[str]] = None,
    network: bool = False,
) -> Dict[str, Any]:
    repo = Path(repo_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    ws = Path(workspace or repo).expanduser().resolve()
    registry = default_tool_registry(ws, repo_root=repo)
    from hcli.connectivity import probe_connectivity

    connectivity = probe_connectivity(repo, workspace=ws, network=network)
    gates = _gate_summary(ws, repo)
    blockers = [
        "unattended production-provider continuation has not been proven",
        "accelerator performance qualification and unattended sovereign-resident operation remain unproven",
    ]
    if gates["recovery_gate"]["status"] == "PASSED":
        blockers = [item for item in blockers if "continuation" not in item]
    if gates["research_gate"]["status"] != "PASSED":
        blockers.append("public research operational gate has not passed")
    if gates["vmcp_gate"]["status"] != "PASSED":
        blockers.append("VMCP operational evidence-boundary gate has not passed")
    if gates["native_gate"]["status"] != "PASSED":
        blockers.append("live native HCLI A1-A6 ladder has not passed")
    if gates["resident_gate"]["status"] != "PASSED":
        blockers.append("20-request native resident proof has not passed")
    if gates["native_mission_gate"]["status"] != "PASSED":
        blockers.append("native tool/verifier mission gate has not passed")
    if gates["accelerator_smoke"]["status"] != "PASSED":
        blockers.append("live native accelerator smoke receipt has not passed")
    if connectivity.get("surfaces", {}).get("modellake", {}).get("status") != "AVAILABLE":
        blockers.append("ModelLake is not mounted in this environment")
    vmcp = connectivity.get("surfaces", {}).get("vmcp", {})
    if vmcp.get("status") not in {"AVAILABLE", "AUTHENTICATED"}:
        blockers.append("VMCP public surface is not fully importable/selected")
    tool_specs = registry.discover()
    return {
        "schema": SCHEMA,
        "generated_at": time.time(),
        "repo_root": str(repo),
        "workspace": str(ws),
        "git_revision": _git_revision(repo),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "provider_neutral_contracts": {
            "schemas": [
                PROFILE_SCHEMA,
                CAPABILITY_SCHEMA,
                GENERATION_REQUEST_SCHEMA,
                GENERATION_RESPONSE_SCHEMA,
                HEALTH_SCHEMA,
                FAILURE_SCHEMA,
                RECEIPT_SCHEMA,
                ROLE_SCHEMA,
            ],
            "role_policy_source": "hcli.providers.RoleRouter",
            "model_name_is_not_a_control_plane_type": True,
        },
        "tools": {
            "count": len(tool_specs),
            "names": [str(item.get("name")) for item in tool_specs],
            "specs": tool_specs,
        },
        "connectivity": connectivity,
        "gates": gates,
        "receipts": _receipt_inventory(
            (repo / "receipts", ws / ".hcli" / "receipts"),
        ),
        "maturity": {
            "control_plane": "FOUNDATION",
            "provider_generalization": "IMPLEMENTED_CONTRACTS",
            "durable_mission": "IMPLEMENTED",
            "fixture_recovery": gates["recovery_gate"]["status"],
            "research": gates["research_gate"]["qualification"],
            "vmcp": gates["vmcp_gate"]["qualification"],
            "native_hcli": gates["native_gate"]["qualification"],
            "native_resident": gates["resident_gate"]["qualification"],
            "native_mission": gates["native_mission_gate"]["qualification"],
            "accelerator_smoke": gates["accelerator_smoke"]["qualification"],
            "unattended_sovereignty": "NOT_CLAIMED",
        },
        "blockers": blockers,
        "next_actions": [
            "run recovery-gate against every configured production provider",
            "persist research provenance and protected benchmark receipts",
            "qualify additional providers only after their own deterministic verification closes",
        ],
        "claim_boundary": "This checkpoint is an evidence census; it does not certify a model, runtime, hardware accelerator, or unattended sovereignty.",
    }


def write_program_checkpoint(
    repo_root: Optional[str | os.PathLike[str]] = None,
    *,
    workspace: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    network: bool = False,
) -> Dict[str, Any]:
    report = build_program_checkpoint(repo_root, workspace=workspace, network=network)
    repo = Path(repo_root or report["repo_root"]).expanduser().resolve()
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / DEFAULT_NAME
    atomic_write_json(destination, report)
    report["checkpoint_path"] = str(destination)
    return report


__all__ = ["DEFAULT_NAME", "SCHEMA", "build_program_checkpoint", "write_program_checkpoint"]
