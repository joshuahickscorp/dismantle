"""Build the non-fabricated Flash-Next executable work contract.

This module is intentionally a scaffold.  It turns the pinned header science
into the next executable interfaces—representation, loader, native kernels,
graph, capability, and complete-token timing—without pretending that a
weight body, a native Flash runtime, or a performance result exists.  It is
safe to run while ModelLake is acquiring: local inspection is bounded and
never reads or mutates the acquisition tree.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.flash_next import (
    ACCEPTED_CAPABILITY_PRESERVING_TPS_MIN,
    COMPLETE_SYSTEM_BYTE_FIELDS,
    COMPLETE_SYSTEM_EBPW_MAX,
    EXPECTED_BYTES,
    PINNED_REVISION,
    REPO_ID,
    evaluate_flash_promotion,
)
from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.flash_next_noetic_executable.v1"
EBPW_SCHEMA = "hcli.agentos.flash_ebpw_budget.v1"
TOKEN_NS_SCHEMA = "hcli.agentos.flash_token_ns_budget.v1"
DERIVED = "[D]"
NOT_MEASURED = "NOT_MEASURED"
DEFAULT_SCIENCE = "HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
DEFAULT_TENSOR_PROBE = "FLASH_FIRST_TENSOR_PROBE.json"
DEFAULT_EXECUTABLE = "FLASH_NEXT_NOETIC_EXECUTABLE.json"
DEFAULT_EBPW = "FLASH_EBPW_BUDGET.json"
DEFAULT_TOKEN_NS = "FLASH_TOKEN_NS_BUDGET.json"
LAKE_ROOT = Path("/Volumes/corpdrive/hawking-modellake")
LAKE_SLUG = REPO_ID.replace("/", "--") + "@" + PINNED_REVISION[:12]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _value(row: Any, key: str = "value") -> Any:
    return row.get(key) if isinstance(row, Mapping) else None


def _direct_inventory(path: Path) -> Dict[str, Any]:
    """Inspect direct children only; never walk a potentially huge lake."""
    if not path.is_dir():
        return {"path": str(path), "present": False, "direct_files": 0, "direct_bytes": 0}
    files = 0
    total = 0
    names: list[str] = []
    try:
        with os.scandir(path) as entries:
            for index, entry in enumerate(entries):
                if index >= 512:
                    break
                try:
                    if entry.is_file(follow_symlinks=False):
                        files += 1
                        total += entry.stat(follow_symlinks=False).st_size
                        names.append(entry.name)
                except OSError:
                    continue
    except OSError as exc:
        return {"path": str(path), "present": True, "direct_files": files, "direct_bytes": total, "error": str(exc)[:400]}
    return {
        "path": str(path),
        "present": True,
        "direct_files": files,
        "direct_bytes": total,
        "entries": sorted(names),
    }


def _modellake_identity(repo: Path) -> Dict[str, Any]:
    census_path = repo / "receipts" / "headless" / "HCLI_MODELLAKE_FLASH_CENSUS.json"
    supervision_path = repo / "receipts" / "headless" / "HCLI_MODELLAKE_FLASH_ACQUISITION_SUPERVISION.json"
    census = _read_json(census_path) or {}
    supervision = _read_json(supervision_path) or {}
    final = LAKE_ROOT / "specimens" / LAKE_SLUG
    partial = LAKE_ROOT / "partial" / LAKE_SLUG
    manifest_path = LAKE_ROOT / "manifests" / f"{LAKE_SLUG}.json"
    manifest = _read_json(manifest_path)
    final_verified = bool(
        final.is_dir()
        and isinstance(manifest, Mapping)
        and manifest.get("resolved_sha") == PINNED_REVISION
    )
    return {
        "repo": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "expected_bytes": EXPECTED_BYTES,
        "census_receipt": {"path": str(census_path), "sha256": _sha256(census_path), "present": census_path.is_file()},
        "supervision_receipt": {"path": str(supervision_path), "sha256": _sha256(supervision_path), "present": supervision_path.is_file()},
        "final": {"path": str(final), "present": final.is_dir(), "verified_manifest": final_verified},
        "partial": _direct_inventory(partial),
        "manifest": {
            "path": str(manifest_path),
            "present": manifest is not None,
            "resolved_sha": manifest.get("resolved_sha") if isinstance(manifest, Mapping) else None,
        },
        "observed_job_status": supervision.get("status"),
        "census_final_present": (census.get("flash_target_manifest") or {}).get("final_present") if isinstance(census.get("flash_target_manifest"), Mapping) else None,
        "body_read_by_this_scaffold": False,
        "mutation_by_this_scaffold": False,
        "status": "VERIFIED_FINAL_IDENTITY" if final_verified else ("PARTIAL_ACQUISITION" if partial.is_dir() else "NOT_STAGED"),
    }


def _tensor_probe_summary(
    repo: Path,
    receipt: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Read bounded probe evidence without treating it as a full model build."""
    path = Path(receipt).expanduser().resolve() if receipt else repo / "receipts" / "headless" / DEFAULT_TENSOR_PROBE
    probe = _read_json(path)
    if probe is None:
        return {
            "status": "NOT_RUN",
            "receipt_path": str(path),
            "source_tensor": None,
            "dense_vs_packed_low_bit": None,
            "whole_model_capability": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        }
    return {
        "status": probe.get("status"),
        "receipt_path": str(path),
        "receipt_sha256": _sha256(path),
        "source_label": probe.get("source_label"),
        "candidate_label": probe.get("candidate_label"),
        "root": probe.get("root"),
        "tensor_name": probe.get("tensor_name"),
        "source_tensor": probe.get("source_tensor"),
        "organ": probe.get("organ"),
        "dense_vs_packed_low_bit": probe.get("dense_vs_packed_low_bit"),
        "next_experiment": probe.get("next_experiment"),
        "body_mutated": probe.get("body_mutated"),
        "model_loaded": probe.get("model_loaded"),
        "whole_model_capability": "NOT_TESTED",
        "whole_model_runtime": "NOT_TESTED",
    }


def _primary_organs(science: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = science.get("organ_graph")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping) and row.get("accounting_role") == "PRIMARY"]


def _source_summary(science: Mapping[str, Any]) -> Dict[str, Any]:
    audit = science.get("safetensors_header_audit") if isinstance(science.get("safetensors_header_audit"), Mapping) else {}
    architecture = science.get("architecture") if isinstance(science.get("architecture"), Mapping) else {}
    payload = _int(audit.get("payload_bytes"))
    dtype_bytes = 2
    # Header science currently reports BF16. Keep this derived from the
    # observed tensor layouts when possible, but never infer missing bodies.
    for organ in science.get("organ_graph") or []:
        if not isinstance(organ, Mapping):
            continue
        for layout in organ.get("tensor_layout") or []:
            if isinstance(layout, Mapping) and str(layout.get("dtype") or "").upper() in {"F16", "BF16"}:
                dtype_bytes = 2
                break
    parameters = payload // dtype_bytes if payload is not None else None
    return {
        "architecture_fingerprint": _safe(science.get("architecture_fingerprint")),
        "source_identity": _safe(science.get("source_identity")),
        "header_audit": {
            "complete": audit.get("complete"),
            "payload_bytes": payload,
            "header_tensor_count": audit.get("header_tensor_count"),
            "body_bytes_requested": audit.get("body_bytes_requested", 0),
            "body_bytes_loaded": audit.get("body_bytes_loaded", 0),
        },
        "declared_dtype_bytes": dtype_bytes,
        "source_parameter_count": parameters,
        "source_payload_bytes": payload,
        "index_total_size": _int(architecture.get("index_total_size")),
    }


def _organ_records(science: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    primary = _primary_organs(science)
    total_source = sum(
        max(0, _int(_value(row.get("stored_bytes"))) or 0)
        for row in primary
    )
    for row in primary:
        organ_id = str(row.get("id") or "unknown")
        source_bytes = _int(_value(row.get("stored_bytes")))
        source_active = _int(_value(row.get("active_bytes_per_token")))
        source_flops = _int(_value(row.get("flops_per_token")))
        fraction = (source_bytes / total_source) if source_bytes is not None and total_source else None
        rows.append({
            "organ": organ_id,
            "label": DERIVED,
            "source_bytes": source_bytes,
            "source_bytes_label": "[V] pinned safetensors header payload" if source_bytes is not None else NOT_MEASURED,
            "source_active_bytes_per_token": source_active,
            "source_flops_per_token": source_flops,
            "source_tensor_count": row.get("tensors"),
            "source_allocation_fraction": fraction,
            "representation_status": "CANDIDATE_NOT_BUILT",
            "chosen_representation": "shared-basis-plus-NF-residual for expert/routed paths; organ-native representation for state, sparse, n-gram, MTP, and vision paths",
            "actual_representation_bytes": None,
            "actual_bytes_label": NOT_MEASURED,
            "native_loader_status": "NOT_IMPLEMENTED",
            "native_kernel_status": "PLAN_ONLY",
            "capability_status": "NOT_RUN",
        })
    # These are explicit views/state requirements and must not disappear just
    # because they are not additive source tensors.
    rows.extend([
        {
            "organ": "recurrent_state",
            "label": DERIVED,
            "source_bytes": 0,
            "source_bytes_label": "[D] virtual runtime state; no source tensor payload",
            "source_active_bytes_per_token": _int(_value(next((r for r in science.get("organ_graph") or [] if isinstance(r, Mapping) and r.get("id") == "recurrent_state"), {}).get("active_bytes_per_token"))),
            "source_flops_per_token": _int(_value(next((r for r in science.get("organ_graph") or [] if isinstance(r, Mapping) and r.get("id") == "recurrent_state"), {}).get("flops_per_token"))),
            "source_allocation_fraction": 0.0,
            "representation_status": "REQUIRED_RESIDENT_STATE_NOT_BUILT",
            "chosen_representation": "resident sequence-isolated state with explicit read/modify/write accounting",
            "actual_representation_bytes": None,
            "actual_bytes_label": NOT_MEASURED,
            "native_loader_status": "NOT_IMPLEMENTED",
            "native_kernel_status": "PLAN_ONLY",
            "capability_status": "NOT_RUN",
        },
    ])
    return rows


def _ebpw_budget(
    science: Mapping[str, Any],
    source: Mapping[str, Any],
    tensor_probe: Mapping[str, Any],
) -> Dict[str, Any]:
    parameters = source.get("source_parameter_count")
    system_ceiling = int(parameters * COMPLETE_SYSTEM_EBPW_MAX) if isinstance(parameters, int) else None
    organs = _organ_records(science)
    accounting = {
        field: {
            "budget_bytes": None,
            "actual_bytes": None,
            "label": DERIVED,
            "status": "WAITING_FOR_REPRESENTATION_AND_LOADER",
        }
        for field in COMPLETE_SYSTEM_BYTE_FIELDS
    }
    chosen_representation = {
        "id": "flash-expert-shared-basis-nf-residual-v0",
        "status": "HYPOTHESIS_NOT_BUILT",
        "label": DERIVED,
        "expert_bank": "shared basis plus per-expert NF residual",
        "router": "resident route metadata and selected-expert gather",
        "deltanet": "resident state-native representation",
        "ngram": "lookup/compositional representation, not generic dense quantization",
        "sparse_attention": "budget/index-native sparse representation",
        "mtp": "explicit draft/verify/rollback representation",
        "vision": "conditional multimodal path; text-resident omission requires an explicit separate contract",
    }
    if tensor_probe.get("status") == "PASSED":
        chosen_representation.update({
            "status": "BOUNDED_SLICE_OBSERVED_NOT_WHOLE_MODEL",
            "bounded_source_probe": {
                "receipt_path": tensor_probe.get("receipt_path"),
                "tensor_name": tensor_probe.get("tensor_name"),
                "organ": tensor_probe.get("organ"),
                "candidate_scheme": ((tensor_probe.get("dense_vs_packed_low_bit") or {}).get("candidate") or {}).get("scheme"),
                "candidate_effective_bits_per_value": ((tensor_probe.get("dense_vs_packed_low_bit") or {}).get("candidate") or {}).get("effective_bits_per_value"),
                "candidate_is_smaller_on_slice": ((tensor_probe.get("dense_vs_packed_low_bit") or {}).get("comparison") or {}).get("candidate_is_smaller"),
                "capability_parity": "NOT_TESTED",
                "whole_model_runtime": "NOT_TESTED",
                "label": DERIVED,
            },
        })
    return {
        "schema": EBPW_SCHEMA,
        "status": "PLANNED_UNTIL_VERIFIED_BODY",
        "label": DERIVED,
        "source_identity": source,
        "target_contract": {
            "complete_system_ebpw_max": COMPLETE_SYSTEM_EBPW_MAX,
            "denominator": "source_parameter_count derived from complete pinned header payload / declared dtype bytes",
            "complete_system_byte_fields": list(COMPLETE_SYSTEM_BYTE_FIELDS),
            "target_ceiling_bytes": system_ceiling,
            "target_ceiling_is_not_an_actual_measurement": True,
        },
        "chosen_representation": chosen_representation,
        "bounded_source_probe": tensor_probe,
        "organs": organs,
        "complete_system_accounting": accounting,
        "measured": {
            "complete_system_bytes": None,
            "complete_system_ebpw": None,
            "all_required_bytes_included": False,
            "fallback_count": None,
            "dense_parent_execution_fallback": None,
            "hidden_dense_rematerialization": None,
        },
        "budget_policy": {
            "allocation": "No guessed per-organ actual is accepted. A provisional ceiling may be allocated only after a representation body and loader manifest exist.",
            "source_payload_is_not_candidate_storage": True,
            "overhead_must_be_counted": True,
            "zero_bytes_are_not_inferred_from_missing_fields": True,
        },
        "promotion_allowed": False,
        "claim_boundary": "This is an EBPW budget and representation contract, not a compressed artifact measurement. Missing actual bytes remain missing.",
    }


def _token_ns_budget(science: Mapping[str, Any], source: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for organ in _organ_records(science):
        rows.append({
            "organ": organ["organ"],
            "label": DERIVED,
            "source_active_bytes_per_token": organ.get("source_active_bytes_per_token"),
            "source_flops_per_token": organ.get("source_flops_per_token"),
            "target_gpu_ns_per_token": None,
            "target_wall_ns_per_token": None,
            "target_dispatches_per_token": None,
            "target_state_read_write_bytes_per_token": None,
            "target_sync_ns": None,
            "target_copy_bytes": None,
            "actual_gpu_ns_per_token": None,
            "actual_complete_wall_ns_per_accepted_token": None,
            "actual_dispatches_per_token": None,
            "actual_state_read_write_bytes_per_token": None,
            "actual_sync_ns": None,
            "actual_copy_bytes": None,
            "status": "WAITING_FOR_NATIVE_EXECUTION",
        })
    return {
        "schema": TOKEN_NS_SCHEMA,
        "status": "PLANNED_UNTIL_NATIVE_EXECUTION",
        "label": DERIVED,
        "source_identity": source,
        "target_contract": {
            "accepted_capability_preserving_tps_min": ACCEPTED_CAPABILITY_PRESERVING_TPS_MIN,
            "complete_wall_ns_per_accepted_token_max": int(1_000_000_000 / ACCEPTED_CAPABILITY_PRESERVING_TPS_MIN),
            "timing_unit": "complete accepted generated token wall time, including all required graph/runtime/host ceremony",
            "kernel_only_or_raw_draft_timing_is_not_acceptable": True,
        },
        "organs": rows,
        "system_ledger": {
            "complete_generation_wall_ns": None,
            "accepted_tokens": None,
            "rejected_draft_tokens": None,
            "prefill_wall_ns": None,
            "decode_wall_ns": None,
            "host_wait_ns": None,
            "gpu_ns": None,
            "sync_ns": None,
            "copy_bytes": None,
            "dispatches": None,
            "fallback_count": None,
            "capability_parity": None,
            "protected_benchmark_class": None,
        },
        "measurement_protocol": {
            "same_source_input_and_output_contract": True,
            "dense_vs_nf_matched_controls": True,
            "complete_token_accounting": True,
            "protected_quiescent_before_and_after": True,
            "native_kernel_genome_and_dispatch_trace": True,
            "no_dense_parent_or_deep_rematerialization": True,
        },
        "promotion_allowed": False,
        "claim_boundary": "No Flash token rate, GPU time, dispatch count, or accepted-token result is claimed until a native executable produces a protected complete-token receipt.",
    }


def _executable_manifest(
    science: Mapping[str, Any],
    source: Mapping[str, Any],
    lake: Mapping[str, Any],
    ebpw: Mapping[str, Any],
    token_ns: Mapping[str, Any],
    tensor_probe: Mapping[str, Any],
) -> Dict[str, Any]:
    organs = [str(row.get("organ")) for row in ebpw.get("organs") or [] if isinstance(row, Mapping)]
    return {
        "schema": SCHEMA,
        "status": "SCAFFOLD_ONLY",
        "qualification": False,
        "NOT_FOR_PROMOTION": True,
        "complete_system_ebpw": None,
        "accepted_capability_preserving_tps": None,
        "fallback_count": None,
        "dense_parent_execution_fallback": False,
        "hidden_dense_rematerialization": False,
        "declarations_are_requirements_not_runtime_evidence": True,
        "label": DERIVED,
        "source_identity": {
            "repo": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "architecture_fingerprint": _safe(science.get("architecture_fingerprint")),
            "science_receipt": source.get("science_receipt"),
            "header_only_source": tensor_probe.get("status") != "PASSED",
            "bounded_source_slice_observed": tensor_probe.get("status") == "PASSED",
            "bounded_probe_receipt": tensor_probe.get("receipt_path"),
            "weight_body_loaded": False,
        },
        "model_lake": lake,
        "source_tensor_probe": tensor_probe,
        "chosen_representation": ebpw.get("chosen_representation"),
        "native_loader": {
            "status": "NOT_IMPLEMENTED",
            "required": ["verified body manifest", "zero-copy/streaming policy", "per-organ ownership", "resident lifetime", "loader hash"],
            "body_read_by_scaffold": False,
        },
        "native_kernels": {
            "status": "PLAN_ONLY",
            "coverage": [
                {"organ": "embeddings", "kernel": "partitioned_embedding_lookup", "status": "NOT_IMPLEMENTED"},
                {"organ": "routed_experts", "kernel": "native_nf_expert_gemv", "status": "NOT_IMPLEMENTED"},
                {"organ": "shared_expert", "kernel": "shared_expert_fused_gemv", "status": "NOT_IMPLEMENTED"},
                {"organ": "router", "kernel": "router_topk_gather", "status": "NOT_IMPLEMENTED"},
                {"organ": "deltanet", "kernel": "persistent_state_update", "status": "NOT_IMPLEMENTED"},
                {"organ": "recurrent_state", "kernel": "resident_state_read_modify_write", "status": "NOT_IMPLEMENTED"},
                {"organ": "sparse_attention", "kernel": "budgeted_sparse_gather_reduce", "status": "NOT_IMPLEMENTED"},
                {"organ": "ngram_engine", "kernel": "lookup_or_compositional_generator", "status": "NOT_IMPLEMENTED"},
                {"organ": "mtp", "kernel": "draft_verify_rollback", "status": "NOT_IMPLEMENTED"},
                {"organ": "norms", "kernel": "fused_norm_epilogue", "status": "NOT_IMPLEMENTED"},
                {"organ": "lm_head", "kernel": "vocabulary_projection_and_reduce", "status": "NOT_IMPLEMENTED"},
                {"organ": "vision_backbone", "kernel": "conditional_multimodal_vision_path", "status": "NOT_IMPLEMENTED"},
                {"organ": "residual_hyperconnections", "kernel": "low_rank_residual_mix", "status": "NOT_IMPLEMENTED"},
                {"organ": "support_misc", "kernel": "ownership_audit_required", "status": "UNRESOLVED"},
            ],
            "dense_rematerialization": "FORBIDDEN_BY_FINAL_RUNTIME_POLICY",
        },
        "graph_runtime": {
            "status": "PLAN_ONLY",
            "organ_order": organs,
            "graph_source": "pinned header organ graph; runtime edges still require body-backed implementation",
            "text_only_vision_bypass": "CONDITIONAL_AND_UNPROVEN",
            "mtp_accept_reject": "EXPLICIT_REQUIRED_EDGE",
            "fallbacks": "No fallback may be silently counted as native Flash execution.",
        },
        "capability_contract": {
            "status": "NOT_RUN",
            "required": ["same-model output parity", "multimodal/text contract", "sequence isolation", "zero hidden dense rematerialization", "fallback disclosure", "accepted-token accounting"],
            "fallback_count": None,
            "dense_parent_execution_fallback": False,
            "hidden_dense_rematerialization": False,
            "declarations_are_requirements_not_runtime_evidence": True,
        },
        "complete_token_timing": {
            "status": "NOT_MEASURED",
            "budget_receipt": token_ns.get("system_ledger"),
            "complete_token_definition": token_ns.get("target_contract", {}).get("timing_unit"),
            "accepted_tps": None,
            "complete_wall_ns_per_accepted_token": None,
        },
        "runtime_genome": {
            "status": "NOT_COMPILED",
            "executable_sha256": None,
            "loader_sha256": None,
            "kernel_source_hashes": [],
            "kernel_binary_hashes": [],
            "graph_fingerprint": None,
            "device_identity": None,
            "compiler_identity": None,
            "representation_manifest_sha256": None,
            "receipt_schema": SCHEMA,
        },
        "ebpw_budget_receipt": DEFAULT_EBPW,
        "token_ns_budget_receipt": DEFAULT_TOKEN_NS,
        "promotion_gate": evaluate_flash_promotion({
            "artifact_contract": "FLASH_NEXT_COMPLETE_MULTIMODAL",
            "complete_system_ebpw": None,
            "accepted_capability_preserving_tps": None,
            "dense_parent_execution_fallback": False,
            "hidden_dense_rematerialization": False,
        }),
        "promotion_allowed": False,
        "claim_boundary": "FLASH_NEXT_NOETIC_EXECUTABLE is not built. This manifest is the reproducible contract for the next implementation steps; it contains no native Flash capability or performance claim.",
    }


def run_flash_executable_scaffold(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    science_receipt: Optional[str | os.PathLike[str]] = None,
    tensor_probe_receipt: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    ebpw_emit: Optional[str | os.PathLike[str]] = None,
    token_ns_emit: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    science_path = Path(science_receipt).expanduser().resolve() if science_receipt else repo / "receipts" / "headless" / DEFAULT_SCIENCE
    science = _read_json(science_path)
    started = time.time()
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / DEFAULT_EXECUTABLE
    ebpw_path = Path(ebpw_emit).expanduser() if ebpw_emit else destination.parent / DEFAULT_EBPW
    token_path = Path(token_ns_emit).expanduser() if token_ns_emit else destination.parent / DEFAULT_TOKEN_NS
    if not ebpw_path.is_absolute():
        ebpw_path = ebpw_path.resolve()
    if not token_path.is_absolute():
        token_path = token_path.resolve()
    result: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "repo_root": str(repo),
        "science_receipt": str(science_path),
        "generated_at": started,
    }
    try:
        if science is None:
            raise FileNotFoundError(science_path)
        source = _source_summary(science)
        source["science_receipt"] = {"path": str(science_path), "sha256": _sha256(science_path), "status": science.get("status")}
        lake = _modellake_identity(repo)
        tensor_probe = _tensor_probe_summary(repo, tensor_probe_receipt)
        ebpw = _ebpw_budget(science, source, tensor_probe)
        token_ns = _token_ns_budget(science, source)
        manifest = _executable_manifest(science, source, lake, ebpw, token_ns, tensor_probe)
        atomic_write_json(ebpw_path, ebpw)
        atomic_write_json(token_path, token_ns)
        manifest["ebpw_budget_receipt"] = str(ebpw_path)
        manifest["token_ns_budget_receipt"] = str(token_path)
        atomic_write_json(destination, manifest)
        result.update({
            "status": "PASSED",
            "manifest": manifest,
            "ebpw_budget": ebpw,
            "token_ns_budget": token_ns,
            "checks": {
                "source_receipt_present": True,
                "source_revision_pinned": (science.get("source_identity") or {}).get("pinned_revision") == PINNED_REVISION if isinstance(science.get("source_identity"), Mapping) else False,
                "header_only_boundary_preserved": source.get("header_audit", {}).get("body_bytes_loaded") == 0 and source.get("header_audit", {}).get("body_bytes_requested") == 0,
                "model_lake_not_mutated": lake.get("mutation_by_this_scaffold") is False,
                "bounded_probe_is_explicit": tensor_probe.get("status") in {"NOT_RUN", "PASSED"},
                "bounded_probe_does_not_claim_whole_model": tensor_probe.get("whole_model_capability") == "NOT_TESTED" and tensor_probe.get("whole_model_runtime") == "NOT_TESTED",
                "native_loader_status_explicit": manifest.get("native_loader", {}).get("status") == "NOT_IMPLEMENTED",
                "native_kernels_status_explicit": manifest.get("native_kernels", {}).get("status") == "PLAN_ONLY",
                "complete_token_timing_not_fabricated": manifest.get("complete_token_timing", {}).get("accepted_tps") is None,
                "promotion_refused": manifest.get("promotion_allowed") is False,
                "budgets_written": destination.is_file() and ebpw_path.is_file() and token_path.is_file(),
            },
        })
        result["status"] = "PASSED" if all(result["checks"].values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001 - keep scaffold failures durable
        result["status"] = "FAILED"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    result["finished_at"] = time.time()
    result["elapsed_s"] = round(result["finished_at"] - started, 3)
    result["receipt_path"] = str(destination)
    if result.get("status") == "FAILED":
        atomic_write_json(destination, result)
    return result


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--science-receipt")
    parser.add_argument("--tensor-probe-receipt")
    parser.add_argument("--emit")
    parser.add_argument("--ebpw-emit")
    parser.add_argument("--token-ns-emit")
    args = parser.parse_args(argv)
    report = run_flash_executable_scaffold(
        repo_root=args.repo_root,
        science_receipt=args.science_receipt,
        tensor_probe_receipt=args.tensor_probe_receipt,
        emit=args.emit,
        ebpw_emit=args.ebpw_emit,
        token_ns_emit=args.token_ns_emit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = ["DEFAULT_TENSOR_PROBE", "EBPW_SCHEMA", "SCHEMA", "TOKEN_NS_SCHEMA", "main", "run_flash_executable_scaffold"]


if __name__ == "__main__":
    raise SystemExit(main())
