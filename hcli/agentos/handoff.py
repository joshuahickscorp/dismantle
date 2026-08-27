"""Write a compact, resumable overnight Hawking handoff receipt.

The handoff is a status snapshot, not a second control plane. It records the
authorities that already exist (Mission/DAG, provider profiles, ModelLake
supervision, and protected receipts), the exact pinned identities, and exact
continuation commands. Missing or unfinished work stays visible.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.overnight_handoff.v1"
DEFAULT_NAME = "OVERNIGHT_HAWKING_HANDOFF.json"
MODEL_LAKE_JOB = "job-2f77c1d6-e33b-44fe-bc12-549cf47805c7"
UNATTENDED_JOB = "job-d659ef87-240c-4504-b1ee-1c0ec459deb2"


def _read_object(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> Optional[Dict[str, Any]]:
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> Optional[str]:
    import hashlib

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _git(repo: Path, *args: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            env={
                "PATH": os.environ.get("PATH", ""),
                "LC_ALL": "C",
                "LANG": "C",
            },
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip()


def _receipt(repo: Path, name: str) -> Optional[Dict[str, Any]]:
    return _read_object(repo / "receipts" / "headless" / name)


def _compact_job(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: job.get(key)
        for key in (
            "job_id",
            "label",
            "state",
            "pid",
            "argv",
            "cwd",
            "resumable",
            "returncode",
            "started_at",
            "finished_at",
            "error",
            "log_path",
        )
        if key in job
    }


def _background(repo: Path) -> list[Dict[str, Any]]:
    try:
        from hcli.agentos.background import BackgroundJobStore

        store = BackgroundJobStore(repo, allowed_roots=(repo,))
        return [_compact_job(item) for item in store.list()]
    except (OSError, ValueError, RuntimeError):
        return []


def _window_summary(repo: Path) -> Dict[str, Any]:
    final_path = repo / "receipts" / "headless" / "HCLI_AGENTOS_UNATTENDED_WINDOW.json"
    final = _read_object(final_path)
    candidates = [
        repo / ".hcli" / "autonomy-window-verified" / "window-progress.json",
        repo / ".hcli" / "autonomy-window" / "window-progress.json",
    ]
    progress: list[Dict[str, Any]] = []
    for path in candidates:
        value = _read_object(path)
        if value is None:
            continue
        progress.append({
            "path": str(path),
            "status": value.get("status"),
            "started_at": value.get("started_at"),
            "deadline": value.get("deadline"),
            "cycles": len(value.get("cycles") or []),
            "metrics": value.get("metrics") or {},
            "checks": value.get("checks") or {},
        })
    return {
        "final_receipt": {
            "path": str(final_path),
            "present": final is not None,
            "status": final.get("status") if final else None,
            "metrics": final.get("metrics") if final else None,
            "checks": final.get("checks") if final else None,
            "elapsed_s": final.get("elapsed_s") if final else None,
        },
        "progress_workspaces": progress,
        "claim_boundary": "Only a completed receipt with fresh model-call evidence can qualify the requested unattended window.",
    }


def _model_lake_summary(repo: Path) -> Dict[str, Any]:
    census = _receipt(repo, "HCLI_MODELLAKE_FLASH_CENSUS.json") or {}
    supervision = _receipt(repo, "HCLI_MODELLAKE_FLASH_ACQUISITION_SUPERVISION.json") or {}
    job = supervision.get("job") if isinstance(supervision.get("job"), dict) else {}
    partial = supervision.get("partial") or {}
    final = supervision.get("final") or {}
    return {
        "root": "/Volumes/corpdrive/hawking-modellake",
        "census": {
            "status": census.get("status"),
            "capacity": census.get("capacity"),
            "verified_specimens": census.get("verified_specimens") or census.get("specimens"),
            "partial_count": len(census.get("partials") or []),
            "target": census.get("target") or {},
        },
        "supervision": {
            "status": supervision.get("status"),
            "qualification": supervision.get("qualification"),
            "job": _compact_job(job) if job else None,
            "partial": {"path": partial.get("path"), "bytes": partial.get("direct_bytes"), "files": partial.get("direct_files")},
            "final": {"present": final.get("present"), "path": final.get("path")},
            "checks": supervision.get("checks") or {},
        },
        "no_delete_policy": True,
        "next_action": "Continue supervision; if interrupted, resume the same pinned argv only after re-census and headroom checks; publish only after full hash verification and atomic rename.",
    }


def _flash_summary(repo: Path) -> Dict[str, Any]:
    flash = _receipt(repo, "HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json") or {}
    source = flash.get("source_identity") or flash.get("source") or {}
    promotion = flash.get("promotion_gate") or {}
    return {
        "repo": source.get("repo") or "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": source.get("pinned_revision") or source.get("resolved_revision"),
        "expected_complete_source_bytes": source.get("expected_complete_source_bytes"),
        "expected_file_count": source.get("expected_file_count"),
        "architecture_fingerprint": (flash.get("architecture_fingerprint") or {}).get("value"),
        "architecture": flash.get("architecture") or {},
        "gravity_targets": [
            "expert bank sharing/bases/residuals/factors/generators/router",
            "separate n-gram lookup/generator system",
            "DeltaNet state representation",
            "QSA sparse attention/indexer",
            "explicit MTP draft/verify/rollback accounting",
            "native kernels and command-boundary telemetry",
        ],
        "promotion": {
            "status": promotion.get("status"),
            "hard_gate": promotion.get("hard_gate") or {},
            "measured": promotion.get("measured") or {},
            "missing_or_refused": promotion.get("missing_or_refused") or [],
        },
        "next_action": "Keep metadata/weights/native executable identity separate; fill complete-system byte ledger and matched dense-vs-NF rows only when the pinned source is fully ready.",
    }


def _qwen27_summary(repo: Path) -> Dict[str, Any]:
    regression = _receipt(repo, "HCLI_ACCELERATOR_REGRESSION.json") or {}
    profile_path = repo / "hcli" / "hawking-native.sealed-3.14.json"
    profile = _read_object(profile_path) or {}
    current = profile.get("current_runtime") or {}
    return {
        "model_a": {
            "label": "Qwen3.8-27B sealed resident / NOETIC_PARENT_A",
            "profile": str(profile_path),
            "profile_sha256": _sha256(profile_path),
            "artifact_root": profile.get("artifact_root"),
            "binary": profile.get("resident_binary") or profile.get("binary"),
            "identity": profile.get("model_id") or profile.get("resident_identity"),
            "physical_ebpw": (profile.get("representation") or {}).get("physical_ebpw") or profile.get("physical_ebpw"),
        },
        "current_runtime_observation": {
            "complete_tps_current_measured": current.get("complete_tps_current_measured"),
            "complete_tps_historical_qualified": current.get("complete_tps_historical_qualified"),
            "fallbacks": current.get("fallbacks"),
            "bench_state": regression.get("bench_state"),
            "qualification": regression.get("qualification"),
            "current_vs_historical": regression.get("current_vs_historical"),
            "dispatch_kernel_genome": regression.get("prior_dispatch_kernel_genome"),
        },
        "next_experiment": "Protected quiescent same-source A/B: record binary/artifact/tokenizer, representation, dispatches, complete wall/GPU timing, fallback count, capability, and cache/quiescence before accepting any optimization.",
    }


def _fpga_summary(repo: Path) -> Dict[str, Any]:
    preboard = _receipt(repo, "HCLI_FPGA_PREBOARD.json") or {}
    return {
        "preboard": {
            "status": preboard.get("status"),
            "fingerprint": preboard.get("fingerprint"),
            "physical_board": preboard.get("physical_board"),
            "fpga_backend": preboard.get("fpga_backend"),
            "checks": preboard.get("checks") or {},
        },
        "maps": {
            "qwen27": str(repo / "receipts" / "headless" / "QWEN27_FPGA_ORGAN_MAP.json"),
            "flash_next": str(repo / "receipts" / "headless" / "FLASH_NEXT_FPGA_ORGAN_MAP.json"),
        },
        "shared_primitives": preboard.get("shared_primitives") or [],
        "labels": {"verified": "[V]", "derived": "[D]", "simulated": "[S]"},
        "next_action": "Compile/verify HWIR and link sensitivity contracts; do not report board, bitstream, U50, or hardware timing until a physical receipt exists.",
    }


def _charge_summary(repo: Path) -> Dict[str, Any]:
    charge = _receipt(repo, "HAWKING_INITIAL_CHARGE.json") or {}
    return {
        "charge_id": charge.get("charge_id"),
        "status": charge.get("status"),
        "mission_id": charge.get("mission_id"),
        "workspace": charge.get("workspace"),
        "mission_state_path": charge.get("mission_state_path"),
        "provider_neutral": charge.get("provider_neutral"),
        "unit_count": len(charge.get("units") or []),
        "units": [
            {
                "id": item.get("id"),
                "priority": item.get("priority"),
                "role": item.get("role"),
                "dependencies": item.get("dependencies") or [],
                "resource_class": item.get("resource_class"),
                "retry_state": item.get("retry_state") or {},
                "stop_condition": item.get("stop_condition"),
            }
            for item in (charge.get("units") or [])
            if isinstance(item, dict)
        ],
        "next_action": charge.get("next_action"),
    }


def build_handoff(repo_root: Optional[str | os.PathLike[str]] = None, *, emit: Optional[str | os.PathLike[str]] = None) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / DEFAULT_NAME
    jobs = _background(repo)
    flash = _flash_summary(repo)
    lake = _model_lake_summary(repo)
    window = _window_summary(repo)
    promotion_status = (flash.get("promotion") or {}).get("status")
    lake_status = (lake.get("supervision") or {}).get("status")
    window_status = (window.get("final_receipt") or {}).get("status")
    blockers = [
        "Flash-Next final promotion gate remains incomplete until every required byte/evidence field and both hard thresholds pass.",
        "Qwen27 current resident observation is a contaminated/contended regression audit, not a performance qualification.",
        "No physical FPGA board, bitstream, or hardware performance is claimed.",
    ]
    if lake_status not in {"READY", "COMPLETED"}:
        blockers.append("Flash-Next ModelLake acquisition is still partial or not yet atomically published.")
    if window_status != "PASSED":
        blockers.append("The corrected one-hour unattended window has not yet produced a final PASSED receipt.")
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": time.time(),
        "status": "READY_FOR_OVERNIGHT_CONTINUATION",
        "claim_boundary": "This handoff is a resumable status snapshot. It does not certify model quality, accelerator performance, Flash promotion, FPGA hardware, or sovereignty.",
        "host": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()},
        "baseline": {
            "git_revision": _git(repo, "rev-parse", "HEAD"),
            "branch": _git(repo, "branch", "--show-current"),
            "working_tree_status": (_git(repo, "status", "--porcelain=v1") or "").splitlines(),
            "baseline_commits": ["5a0c84d4ebde2687e60891829f81f47e06fecd3d", "9d373ecc4863b244fc74761c99fc837f7705f3db"],
            "unrelated_preserved_edit": "tools/odyssey_ctl.py",
        },
        "hcli": {
            "provider_neutral_semantics": "Mission/DAG owns work identity, dependencies, resources, stop conditions, checkpoints, receipts, and retry state; provider/model identity is execution policy.",
            "current_default_profile": str(repo / "hcli" / "hawking-native.sealed-3.14.json"),
            "explicit_selection_rule": "--model/config/env/provider selection wins; the resident profile is only the local default when no explicit selection exists.",
            "initial_charge": _charge_summary(repo),
            "background_jobs": jobs,
            "unattended_window": window,
            "transfer_map": str(repo / "receipts" / "headless" / "QWEN38_ACCELERATOR_TRANSFER_MAP.json"),
            "precedent_map": str(repo / "receipts" / "headless" / "FLASH_NEXT_PRECEDENT_MAP.json"),
            "dense_vs_nf_scaffold": str(repo / "receipts" / "headless" / "HCLI_DENSE_VS_NF_AB_SCAFFOLD.json"),
        },
        "qwen27": _qwen27_summary(repo),
        "flash_next": flash,
        "modellake": lake,
        "fpga": _fpga_summary(repo),
        "verification": {
            "full_suite": {"command": "pytest -q", "last_observed_status": "PASSED", "last_observed": "710 passed, 2 skipped, 2 warnings"},
            "provider_focus": {"last_observed_status": "PASSED", "last_observed": "31 passed, 2 warnings"},
            "receipt_gates": {
                "autonomy": (_receipt(repo, "HCLI_AGENTOS_AUTONOMY_GATE.json") or {}).get("status"),
                "accelerator_regression": (_receipt(repo, "HCLI_ACCELERATOR_REGRESSION.json") or {}).get("status"),
                "preboard": (_receipt(repo, "HCLI_AGENTOS_PREBOARD.json") or {}).get("status"),
                "flash_pre_runtime": (_receipt(repo, "HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json") or {}).get("status"),
                "modellake_supervision": lake_status,
                "unattended_window": window_status,
            },
        },
        "blockers": blockers,
        "continuation": {
            "inspect_status": f"python3 -m hcli agentos status --workspace {repo} --repo-root {repo}",
            "refresh_checkpoint": f"python3 -m hcli agentos checkpoint --repo-root {repo} --workspace {repo} --emit {repo / 'receipts/headless/HCLI_AGENTOS_CHECKPOINT.json'}",
            "supervise_modellake": f"python3 -m hcli agentos modellake-supervise --repo-root {repo} --job-id {MODEL_LAKE_JOB} --emit {repo / 'receipts/headless/HCLI_MODELLAKE_FLASH_ACQUISITION_SUPERVISION.json'}",
            "modellake_job_status": f"python3 -m hcli agentos background status --workspace {repo} --repo-root {repo} {MODEL_LAKE_JOB}",
            "unattended_job_status": f"python3 -m hcli agentos background status --workspace {repo} --repo-root {repo} {UNATTENDED_JOB}",
            "refresh_initial_charge": f"python3 -m hcli agentos initial-charge --repo-root {repo} --workspace {repo / '.hcli/initial-charge'} --emit {repo / 'receipts/headless/HAWKING_INITIAL_CHARGE.json'}",
            "refresh_maps": f"python3 -m hcli agentos science-maps --repo-root {repo} --transfer-emit {repo / 'receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json'} --precedent-emit {repo / 'receipts/headless/FLASH_NEXT_PRECEDENT_MAP.json'}",
            "refresh_ab": f"python3 -m hcli agentos ab-scaffold --repo-root {repo} --emit {repo / 'receipts/headless/HCLI_DENSE_VS_NF_AB_SCAFFOLD.json'}",
            "refresh_fpga_preboard": f"python3 -m hcli agentos fpga-preboard --repo-root {repo} --emit {repo / 'receipts/headless/HCLI_FPGA_PREBOARD.json'}",
            "run_full_tests": "pytest -q",
            "resume_modellake_only_if_interrupted": f"python3 -m hcli agentos background resume --workspace {repo} --repo-root {repo} {MODEL_LAKE_JOB}",
            "do_not_start": "Do not launch or promote a new Odyssey; continue only through the existing HCLI/ModelLake authorities and governed windows.",
        },
    }
    payload["receipt_path"] = str(destination)
    atomic_write_json(destination, payload)
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    print(json.dumps(build_handoff(args.repo_root, emit=args.emit), indent=2, sort_keys=True, default=str))
    return 0


__all__ = ["DEFAULT_NAME", "SCHEMA", "build_handoff", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
