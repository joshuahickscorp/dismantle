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
    autonomy_candidates = [
        workspace / ".hcli" / "receipts" / "autonomy-gate.json",
        repo_root / "receipts" / "headless" / "autonomy-gate.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_AUTONOMY_GATE.json",
    ]
    unattended_candidates = [
        workspace / ".hcli" / "receipts" / "unattended-window.json",
        repo_root / "receipts" / "headless" / "unattended-window.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_UNATTENDED_WINDOW.json",
    ]
    accelerator_regression_candidates = [
        workspace / ".hcli" / "receipts" / "accelerator-regression.json",
        repo_root / "receipts" / "headless" / "accelerator-regression.json",
        repo_root / "receipts" / "headless" / "HCLI_ACCELERATOR_REGRESSION.json",
    ]
    modellake_candidates = [
        workspace / ".hcli" / "receipts" / "modellake-census.json",
        repo_root / "receipts" / "headless" / "modellake-census.json",
        repo_root / "receipts" / "headless" / "HCLI_MODELLAKE_FLASH_CENSUS.json",
    ]
    flash_science_candidates = [
        workspace / ".hcli" / "receipts" / "flash-science.json",
        repo_root / "receipts" / "headless" / "flash-science.json",
        repo_root / "receipts" / "headless" / "HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json",
    ]
    preboard_candidates = [
        workspace / ".hcli" / "receipts" / "preboard.json",
        repo_root / "receipts" / "headless" / "preboard.json",
        repo_root / "receipts" / "headless" / "HCLI_AGENTOS_PREBOARD.json",
    ]
    charge_candidates = [
        workspace / ".hcli" / "receipts" / "initial-charge.json",
        repo_root / "receipts" / "headless" / "initial-charge.json",
        repo_root / "receipts" / "headless" / "HAWKING_INITIAL_CHARGE.json",
    ]
    transfer_map_candidates = [repo_root / "receipts" / "headless" / "QWEN38_ACCELERATOR_TRANSFER_MAP.json"]
    precedent_map_candidates = [repo_root / "receipts" / "headless" / "FLASH_NEXT_PRECEDENT_MAP.json"]
    ab_candidates = [repo_root / "receipts" / "headless" / "HCLI_DENSE_VS_NF_AB_SCAFFOLD.json"]
    fpga_candidates = [repo_root / "receipts" / "headless" / "HCLI_FPGA_PREBOARD.json"]
    lake_supervision_candidates = [repo_root / "receipts" / "headless" / "HCLI_MODELLAKE_FLASH_ACQUISITION_SUPERVISION.json"]
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
    autonomy = None
    for path in autonomy_candidates:
        value = _read_object(path)
        if value is not None:
            autonomy = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "stage_status": value.get("stage_status"),
            }
            break
    unattended = None
    for path in unattended_candidates:
        value = _read_object(path)
        if value is not None:
            unattended = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "metrics": value.get("metrics"),
                "duration_requested_s": value.get("duration_requested_s"),
                "elapsed_s": value.get("elapsed_s"),
            }
            break
    accelerator_regression = None
    for path in accelerator_regression_candidates:
        value = _read_object(path)
        if value is not None:
            accelerator_regression = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "bench_state": ((value.get("experiment") or {}).get("bench") or {}).get("state")
                if isinstance(value.get("experiment"), dict)
                else None,
                "perf_qualified": ((value.get("experiment") or {}).get("perf_qualified"))
                if isinstance(value.get("experiment"), dict)
                else None,
            }
            break
    modellake = None
    for path in modellake_candidates:
        value = _read_object(path)
        if value is not None:
            modellake = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "download_performed": (value.get("acquisition_policy") or {}).get("download_performed")
                if isinstance(value.get("acquisition_policy"), dict)
                else None,
                "pinned_revision": (value.get("source") or {}).get("requested_revision")
                if isinstance(value.get("source"), dict)
                else None,
            }
            break
    flash_science = None
    for path in flash_science_candidates:
        value = _read_object(path)
        if value is not None:
            flash_science = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "promotion_gate": value.get("promotion_gate"),
                "architecture_fingerprint": value.get("architecture_fingerprint"),
            }
            break
    preboard = None
    for path in preboard_candidates:
        value = _read_object(path)
        if value is not None:
            preboard = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "claim_boundary": value.get("claim_boundary"),
            }
            break
    charge = None
    for path in charge_candidates:
        value = _read_object(path)
        if value is not None:
            charge = {
                "status": value.get("status"),
                "charge_id": value.get("charge_id"),
                "receipt_path": str(path),
                "mission_id": value.get("mission_id"),
                "workspace": value.get("workspace"),
                "unit_count": len(value.get("units") or []) if isinstance(value.get("units"), list) else None,
                "provider_neutral": value.get("provider_neutral"),
            }
            break
    transfer_map = None
    for path in transfer_map_candidates:
        value = _read_object(path)
        if value is not None:
            transfer_map = {
                "status": "PRESENT",
                "receipt_path": str(path),
                "schema": value.get("schema"),
                "fingerprint": value.get("fingerprint"),
                "entries": len(value.get("transfer_matrix") or []) if isinstance(value.get("transfer_matrix"), list) else None,
            }
            break
    precedent_map = None
    for path in precedent_map_candidates:
        value = _read_object(path)
        if value is not None:
            precedent_map = {
                "status": "PRESENT",
                "receipt_path": str(path),
                "schema": value.get("schema"),
                "fingerprint": value.get("fingerprint"),
                "entries": len(value.get("entries") or []) if isinstance(value.get("entries"), list) else None,
            }
            break
    ab_scaffold = None
    for path in ab_candidates:
        value = _read_object(path)
        if value is not None:
            ab_scaffold = {
                "status": value.get("status"),
                "receipt_path": str(path),
                "schema": value.get("schema"),
                "evaluation": value.get("evaluation"),
            }
            break
    fpga = None
    for path in fpga_candidates:
        value = _read_object(path)
        if value is not None:
            fpga = {
                "status": value.get("status"),
                "receipt_path": str(path),
                "schema": value.get("schema"),
                "fingerprint": value.get("fingerprint"),
                "checks": value.get("checks"),
                "physical_board": value.get("physical_board"),
            }
            break
    lake_supervision = None
    for path in lake_supervision_candidates:
        value = _read_object(path)
        if value is not None:
            lake_supervision = {
                "status": value.get("status"),
                "qualification": value.get("qualification"),
                "receipt_path": str(path),
                "checks": value.get("checks"),
                "job": value.get("job"),
                "target": value.get("target"),
                "capacity": value.get("capacity"),
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
        "autonomy_gate": autonomy or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "stage_status": {},
        },
        "unattended": unattended or "NOT_PROVEN",
        "accelerator_regression": accelerator_regression or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "bench_state": None,
            "perf_qualified": None,
        },
        "modellake": modellake or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "download_performed": None,
            "pinned_revision": None,
        },
        "flash_science": flash_science or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "promotion_gate": None,
            "architecture_fingerprint": None,
        },
        "preboard": preboard or {
            "status": "NOT_RUN",
            "qualification": "NONE",
            "receipt_path": None,
            "checks": {},
            "claim_boundary": None,
        },
        "initial_charge": charge or {
            "status": "NOT_RUN",
            "charge_id": None,
            "receipt_path": None,
            "mission_id": None,
            "workspace": None,
            "unit_count": None,
            "provider_neutral": None,
        },
        "transfer_map": transfer_map or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "fingerprint": None, "entries": None},
        "precedent_map": precedent_map or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "fingerprint": None, "entries": None},
        "ab_scaffold": ab_scaffold or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "evaluation": None},
        "fpga_preboard": fpga or {"status": "NOT_RUN", "receipt_path": None, "schema": None, "fingerprint": None, "checks": {}, "physical_board": None},
        "modellake_supervision": lake_supervision or {"status": "NOT_RUN", "qualification": "NONE", "receipt_path": None, "checks": {}, "job": None, "target": None, "capacity": None},
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
    if gates["autonomy_gate"]["status"] != "PASSED":
        blockers.append("A1-A5 AgentOS autonomy and crash-recovery qualification has not passed")
    if gates["accelerator_smoke"]["status"] != "PASSED":
        blockers.append("live native accelerator smoke receipt has not passed")
    if gates["accelerator_regression"]["status"] != "PASSED":
        blockers.append("current-vs-historical accelerator regression audit has not passed")
    if gates["modellake"]["status"] != "PASSED":
        blockers.append("pinned Flash-Next ModelLake census has not passed")
    if gates["flash_science"]["status"] != "PASSED":
        blockers.append("Flash-Next pre-runtime architecture/organ science has not passed")
    if gates["preboard"]["status"] != "PASSED":
        blockers.append("negative-science/FPGA compiler preboard has not passed")
    if gates["initial_charge"]["status"] not in {"CREATED", "IDEMPOTENT_EXISTING_CHARGE"}:
        blockers.append("provider-neutral Hawking initial charge has not been persisted")
    if gates["transfer_map"]["status"] != "PRESENT":
        blockers.append("two-Qwen accelerator transfer map has not been persisted")
    if gates["precedent_map"]["status"] != "PRESENT":
        blockers.append("Flash-Next precedent map has not been persisted")
    if gates["ab_scaffold"]["status"] != "READY_SCAFFOLD":
        blockers.append("dense-vs-NF A/B scaffold has not been persisted")
    if gates["fpga_preboard"]["status"] != "PASSED":
        blockers.append("two-model FPGA preboard maps have not passed")
    if gates["modellake_supervision"]["status"] not in {"RUNNING_SAFE", "PASSED", "WAITING_OR_NOT_OBSERVED"}:
        blockers.append("pinned Flash-Next ModelLake acquisition is not in a safe observed state")
    flash_promotion = gates["flash_science"].get("promotion_gate")
    if isinstance(flash_promotion, dict) and flash_promotion.get("status") != "PROMOTABLE":
        blockers.append("Flash-Next final promotion gate is not PROMOTABLE (complete EBPW/TPS or required evidence is missing)")
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
            "autonomy": gates["autonomy_gate"]["qualification"],
            "accelerator_smoke": gates["accelerator_smoke"]["qualification"],
            "unattended_sovereignty": (
                gates["unattended"].get("qualification")
                if isinstance(gates.get("unattended"), dict)
                else "NOT_CLAIMED"
            ),
            "flash_pre_runtime": gates["flash_science"]["qualification"],
            "negative_science_preboard": gates["preboard"]["qualification"],
            "initial_charge": gates["initial_charge"]["status"],
            "qwen38_transfer_map": gates["transfer_map"]["status"],
            "flash_precedent_map": gates["precedent_map"]["status"],
            "dense_vs_nf_ab": gates["ab_scaffold"]["status"],
            "fpga_preboard": gates["fpga_preboard"]["status"],
            "modellake_supervision": gates["modellake_supervision"]["status"],
            "flash_promotion": flash_promotion.get("status") if isinstance(flash_promotion, dict) else "NOT_PROVEN",
        },
        "blockers": blockers,
        "next_actions": [
            "run recovery-gate against every configured production provider",
            "complete the one-hour unattended production-provider observation before making any sovereignty claim",
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
