"""Execute bounded Flash router softmax/top-k semantics over a persisted body.

The matrix body and native Q4 matvec are already qualified as bounded
primitives.  This module adds the next executable edge without loading the
model: it decodes the persisted source-independent body, runs the same
deterministic reference vector used by the native parity probe, applies the
router's FP32 softmax/top-k contract, and records the selected expert ids and
weights.  The selection is a derived CPU execution; the native Metal receipt
remains a matvec parity reference, not evidence of a complete token runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.agentos import flash_router_graph
from hcli.agentos import flash_transform_parity as transform
from hcli.flash_next import PINNED_REVISION, REPO_ID
from hcli.nomenclature import NOMENCLATURE_VERSION
from hcli.persist import atomic_write_json
from hcli.physical_graph import compile_physical_graph


SCHEMA = "hcli.agentos.flash_noetic_router_selection.v1"
BODY_SCHEMA = "hcli.agentos.flash_noetic_component_body.v1"
KERNEL_SCHEMA = "hawking.flash_noetic_q4_kernel_parity.v1"
DEFAULT_BODY = "FLASH_NOETIC_ROUTER_COMPONENT_FULL_BODY.json"
DEFAULT_KERNEL = "FLASH_NOETIC_ROUTER_COMPONENT_FULL_KERNEL_PARITY.json"
DEFAULT_EMIT = "FLASH_NOETIC_ROUTER_SELECTION.json"
DEFAULT_TENSOR = "model.language_model.layers.0.mlp.gate.weight"
REFERENCE_MULTIPLIER = 71
REFERENCE_MODULUS = 509
REFERENCE_OFFSET = 254


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _resolve(repo: Path, value: Optional[str | Path], default: str) -> Path:
    path = Path(value).expanduser() if value else Path(default)
    if path.is_absolute():
        return path.resolve()
    return (repo / "receipts" / "headless" / path).resolve()


def _ref(path: Path, value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema": value.get("schema"),
        "status": value.get("status"),
        "label": "[V]",
    }


def _identity(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "repo": value.get("repo"),
        "revision": value.get("pinned_revision"),
        "root": value.get("root"),
    }


def _router_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    nested = config.get("text_config")
    values = nested if isinstance(nested, Mapping) else config
    try:
        num_experts = int(values["num_experts"])
        top_k = int(values["num_experts_per_tok"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("pinned config does not expose router expert counts") from exc
    if num_experts <= 0 or top_k <= 0 or top_k > num_experts:
        raise ValueError("router expert counts are outside a valid range")
    norm_present = "norm_topk_prob" in values
    norm_topk_prob = bool(values.get("norm_topk_prob", True))
    return {
        "config_scope": "text_config" if isinstance(nested, Mapping) else "root",
        "num_experts": num_experts,
        "num_experts_per_tok": top_k,
        "norm_topk_prob": norm_topk_prob,
        "norm_topk_prob_present_in_pinned_config": norm_present,
        "norm_topk_prob_default_when_absent": True,
        "router_logits": "F.linear(hidden_states, gate.weight)",
        "router_probability": "softmax(router_logits, dtype=float32, dim=-1)",
        "selection": "topk(router_probs, num_experts_per_tok)",
        "shared_expert_sigmoid_is_not_router_selection": True,
    }


def reference_input(np: Any, columns: int) -> Any:
    """Return the deterministic vector shared with the native parity probe."""
    indices = np.arange(int(columns), dtype=np.int64)
    return (((indices * REFERENCE_MULTIPLIER) % REFERENCE_MODULUS) - REFERENCE_OFFSET).astype(np.float32) / np.float32(REFERENCE_MODULUS)


def stable_softmax(np: Any, logits: Any) -> Any:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("router logits must be a non-empty rank-1 vector")
    shifted = values - np.max(values)
    exponent = np.exp(shifted).astype(np.float32, copy=False)
    denominator = np.sum(exponent, dtype=np.float32)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("router softmax denominator is not finite and positive")
    return (exponent / denominator).astype(np.float32, copy=False)


def select_router(np: Any, logits: Any, *, top_k: int, norm_topk_prob: bool) -> Dict[str, Any]:
    """Apply deterministic FP32 router softmax, stable top-k, and normalization."""
    probabilities = stable_softmax(np, logits)
    count = int(top_k)
    if count <= 0 or count > probabilities.size:
        raise ValueError("router top_k is outside the logits range")
    order = np.argsort(-probabilities, kind="stable")[:count]
    selected_probabilities = probabilities[order].astype(np.float32, copy=True)
    if norm_topk_prob:
        selected_sum = np.sum(selected_probabilities, dtype=np.float32)
        if not np.isfinite(selected_sum) or selected_sum <= 0.0:
            raise ValueError("selected router probabilities cannot be normalized")
        selected_weights = (selected_probabilities / selected_sum).astype(np.float32, copy=False)
    else:
        selected_weights = selected_probabilities
    return {
        "expert_ids": [int(value) for value in order.tolist()],
        "router_probabilities": [float(value) for value in selected_probabilities.tolist()],
        "selected_weights": [float(value) for value in selected_weights.tolist()],
        "selected_probability_sum": float(np.sum(selected_probabilities, dtype=np.float32)),
        "selected_weight_sum": float(np.sum(selected_weights, dtype=np.float32)),
        "probability_vector_sha256": _sha256_bytes(probabilities.astype("<f4", copy=False).tobytes()),
        "logits_sha256": _sha256_bytes(np.asarray(logits, dtype="<f4").tobytes()),
        "probabilities_finite": bool(np.isfinite(probabilities).all()),
    }


def _validate_pair(body_path: Path, body: Mapping[str, Any], kernel_path: Path, kernel: Mapping[str, Any]) -> list[str]:
    errors = list(flash_router_graph._validate_body(body_path, body))
    if kernel.get("schema") != KERNEL_SCHEMA or kernel.get("status") != "PASSED":
        errors.append("router kernel receipt is not PASSED")
    native_loader = kernel.get("native_loader") if isinstance(kernel.get("native_loader"), Mapping) else {}
    native_kernel = kernel.get("native_kernel") if isinstance(kernel.get("native_kernel"), Mapping) else {}
    parity = kernel.get("parity") if isinstance(kernel.get("parity"), Mapping) else {}
    if native_loader.get("status") != "BOUNDED_NOETIC_DESCRIPTOR_AND_BODY_LOAD":
        errors.append("router kernel does not prove a persisted body load")
    if native_loader.get("source_independent_execution") is not True or native_loader.get("candidate_body_persisted") is not True:
        errors.append("router kernel does not prove source-independent execution")
    if native_kernel.get("kernel_registered") is not True or parity.get("within_tolerance") is not True:
        errors.append("router native kernel registration/parity is not verified")
    if kernel.get("body_mutated") is not False or kernel.get("model_loaded") is not False:
        errors.append("router kernel mutation/model-load guards are not false")
    if _identity(body) != _identity(kernel):
        errors.append("router body and kernel identities disagree")
    if body.get("repo") != REPO_ID or body.get("pinned_revision") != PINNED_REVISION:
        errors.append("router body does not identify the pinned Flash source")
    source = body.get("source_block") if isinstance(body.get("source_block"), Mapping) else {}
    kernel_source = kernel.get("source_tensor") if isinstance(kernel.get("source_tensor"), Mapping) else {}
    for field in ("tensor_name", "shape", "dtype"):
        if source.get(field) != kernel_source.get(field):
            errors.append(f"router body and kernel {field} disagree")
    for field in ("bytes", "payload_sha256"):
        kernel_field = "selected_block_bytes" if field == "bytes" else "selected_block_sha256"
        if source.get(field) != kernel_source.get(kernel_field):
            errors.append(f"router body and kernel source-block {field} disagree")
    if source.get("row_start") != kernel_source.get("selected_row_start") or source.get("row_count") != kernel_source.get("selected_row_count"):
        errors.append("router body and kernel source-block windows disagree")
    body_record = body.get("body") if isinstance(body.get("body"), Mapping) else {}
    candidate_body = kernel.get("candidate_body") if isinstance(kernel.get("candidate_body"), Mapping) else {}
    for field in ("path", "sha256", "bytes"):
        if body_record.get(field) != candidate_body.get(field):
            errors.append(f"router kernel candidate body {field} does not match the persisted body")
    shape = source.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        errors.append("router source shape is not rank-2")
    else:
        rows, columns = (int(shape[0]), int(shape[1]))
        if source.get("row_start") != 0 or source.get("row_count") != rows:
            errors.append("router selection requires the complete expert-row window")
        if rows <= 0 or columns <= 0 or columns % transform.GROUP_SIZE:
            errors.append("router source shape is not compatible with G64 decoding")
    if source.get("tensor_name") != DEFAULT_TENSOR:
        errors.append("router selection is currently bound to the pinned layer-0 gate tensor")
    return errors


def _decode_body(np: Any, body: Mapping[str, Any], shape: list[int]) -> Any:
    body_record = body.get("body") if isinstance(body.get("body"), Mapping) else {}
    body_path = Path(str(body_record.get("path"))).expanduser().resolve()
    payload = body_path.read_bytes()
    rows, columns = int(shape[0]), int(shape[1])
    expected_code_bytes = rows * columns // 2
    expected_scale_bytes = rows * (columns // transform.GROUP_SIZE) * 2
    code_bytes = int(body_record.get("code_bytes", expected_code_bytes))
    scale_bytes = int(body_record.get("scale_bytes", expected_scale_bytes))
    if code_bytes != expected_code_bytes or scale_bytes != expected_scale_bytes or len(payload) != code_bytes + scale_bytes:
        raise ValueError("persisted router body has unexpected Q4/G64 byte layout")
    decoded = transform._decode_q4(np, payload[:code_bytes], payload[code_bytes:], rows * columns)
    if not bool(np.isfinite(decoded).all()):
        raise ValueError("decoded router body contains non-finite values")
    return decoded.reshape(rows, columns).astype(np.float32, copy=False)


def run_flash_router_selection(
    *,
    repo_root: Optional[str | Path] = None,
    root: Optional[str | Path] = None,
    body_receipt: Optional[str | Path] = None,
    kernel_receipt: Optional[str | Path] = None,
    emit: Optional[str | Path] = None,
) -> Dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    body_path = _resolve(repo, body_receipt, DEFAULT_BODY)
    kernel_path = _resolve(repo, kernel_receipt, DEFAULT_KERNEL)
    destination = Path(emit).expanduser().resolve() if emit else repo / "receipts" / "headless" / DEFAULT_EMIT
    started = time.time()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "nomenclature_version": NOMENCLATURE_VERSION,
        "semantic_type": "NoeticExecutableCandidate",
        "compiler_stage": "NoeticExecutableCandidate",
        "status": "RUNNING",
        "repo": REPO_ID,
        "pinned_revision": PINNED_REVISION,
        "whole_model_capability": "NOT_TESTED",
        "complete_token_runtime": "NOT_TESTED",
        "complete_system_ebpw": None,
        "flash_tps": None,
        "promotion_allowed": False,
        "model_loaded": False,
        "body_mutated": False,
        "native_selection_execution_observed": False,
    }
    errors: list[str] = []
    try:
        import numpy as np

        body = _read_json(body_path)
        kernel = _read_json(kernel_path)
        if not isinstance(body, Mapping):
            errors.append(f"router body receipt is missing or invalid: {body_path}")
        if not isinstance(kernel, Mapping):
            errors.append(f"router kernel receipt is missing or invalid: {kernel_path}")
        if isinstance(body, Mapping) and isinstance(kernel, Mapping):
            errors.extend(_validate_pair(body_path, body, kernel_path, kernel))
        if errors:
            report["validation"] = {"errors": errors}
            report["status"] = "FAILED"
        elif isinstance(body, Mapping) and isinstance(kernel, Mapping):
            source = body["source_block"]
            shape = [int(value) for value in source["shape"]]
            source_root = Path(str(root)).expanduser().resolve() if root else Path(str(body.get("root"))).expanduser().resolve()
            if source_root != Path(str(body.get("root"))).expanduser().resolve():
                errors.append("selected root does not match the body receipt root")
            if not source_root.is_dir():
                errors.append(f"pinned ModelLake specimen root is missing: {source_root}")
            if not errors:
                manifest = transform._pinned_manifest(source_root)
                config_path = source_root / "config.json"
                config = _read_json(config_path)
                if not isinstance(config, Mapping):
                    errors.append(f"pinned router config is missing or invalid: {config_path}")
                else:
                    router_config = _router_config(config)
                    if router_config["num_experts"] != shape[0]:
                        errors.append("router config expert count does not match the full body rows")
                    if router_config["num_experts_per_tok"] > shape[0]:
                        errors.append("router config top_k exceeds the full body rows")
            if errors:
                report["validation"] = {"errors": errors}
                report["status"] = "FAILED"
            else:
                body_record = body["body"]
                descriptor = body["representation_descriptor"]
                native_kernel = kernel["native_kernel"]
                values = _decode_body(np, body, shape)
                vector = reference_input(np, shape[1])
                logits = np.matmul(values, vector).astype(np.float32, copy=False)
                selection = select_router(
                    np,
                    logits,
                    top_k=int(router_config["num_experts_per_tok"]),
                    norm_topk_prob=bool(router_config["norm_topk_prob"]),
                )
                input_hash = _sha256_bytes(vector.astype("<f4", copy=False).tobytes())
                native_input = kernel.get("input") if isinstance(kernel.get("input"), Mapping) else {}
                if native_input.get("deterministic_sha256") and native_input.get("deterministic_sha256") != input_hash:
                    errors.append("selection reference vector does not match the native parity input")
                if errors:
                    report["validation"] = {"errors": errors}
                    report["status"] = "FAILED"
                else:
                    config_digest = _sha256_file(config_path)
                    body_ref = _ref(body_path, body)
                    kernel_ref = _ref(kernel_path, kernel)
                    physical = compile_physical_graph(
                        {
                            "model_id": REPO_ID,
                            "architecture": {
                                "component": "router_selection",
                                "tensor_name": source.get("tensor_name"),
                                "shape": shape,
                                "num_experts": router_config["num_experts"],
                                "num_experts_per_tok": router_config["num_experts_per_tok"],
                            },
                            "organs": [{"id": "router", "present": True, "tensor_count": 1, "confidence": "[V]/[D]"}],
                            "evidence": [body_ref, kernel_ref],
                        },
                        provider={"provider": "derived-cpu-selection", "kernel": native_kernel.get("kernel"), "native_parity_reference": "apple-metal"},
                        devices=("cpu", "apple_metal"),
                    )
                    physical["component_scope"] = "full pinned Flash router matrix body through derived FP32 softmax/top-k selection; no token graph"
                    physical["computation"] = [
                        {"id": "router_source_reference", "stage": "SourceSpecimen", "kind": "verified_source_reference", "execution_input": False, "tensor_name": source.get("tensor_name"), "shape": shape},
                        {"id": "router_body_load", "stage": "NoeticCompiler", "kind": "source_independent_component_body", "execution_input": True, "candidate_id": body.get("candidate_id"), "body_persisted": True},
                        {"id": "router_q4_matvec_reference", "stage": "HawkingAccelerator", "kind": "native_kernel_parity_reference", "execution_input": True, "kernel": native_kernel.get("kernel"), "native_execution_observed": False, "parity_within_tolerance": True},
                        {"id": "router_fp32_softmax", "stage": "NoeticExecutableCandidate", "kind": "router_probability_normalization", "execution_input": True, "dtype": "float32"},
                        {"id": "router_top_k", "stage": "NoeticExecutableCandidate", "kind": "router_top_k_selection", "execution_input": True, "top_k": router_config["num_experts_per_tok"], "stable_tie_break": "expert_id_ascending"},
                        {"id": "router_selected_weight_normalization", "stage": "NoeticExecutableCandidate", "kind": "selected_probability_normalization", "execution_input": True, "enabled": router_config["norm_topk_prob"]},
                    ]
                    physical["dependencies"] = [
                        {"from": "router_body_load", "to": "router_q4_matvec_reference", "kind": "body_to_matvec"},
                        {"from": "router_q4_matvec_reference", "to": "router_fp32_softmax", "kind": "logits_to_probabilities"},
                        {"from": "router_fp32_softmax", "to": "router_top_k", "kind": "probabilities_to_selection"},
                        {"from": "router_top_k", "to": "router_selected_weight_normalization", "kind": "selection_to_weights"},
                    ]
                    physical["device_placement"]["selected"] = "cpu"
                    physical["graph_execution_observed"] = True
                    physical["native_kernel_execution_observed"] = False
                    physical["qualification"] = "BOUNDED_ROUTER_SELECTION_DERIVED"
                    physical["representation"] = {
                        "candidate_id": body.get("candidate_id"),
                        "source_layout": descriptor.get("source_tensor", {}).get("layout"),
                        "component_body_bytes": body_record.get("bytes"),
                        "candidate_body_persisted": True,
                        "source_independent_execution": True,
                        "dense_rematerialization": "forbidden",
                    }
                    physical["residency"] = {
                        "component_body": "persisted body is the execution input",
                        "cold_source": "parity reference only",
                        "whole_model": "not loaded",
                    }
                    physical["fingerprint"] = hashlib.sha256(json.dumps({k: v for k, v in physical.items() if k not in {"generated_at", "fingerprint"}}, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
                    report.update({
                        "status": "PASSED",
                        "source_identity": {**_identity(body), "model_lake_manifest": manifest},
                        "config": {
                            "path": str(config_path),
                            "sha256": config_digest,
                            "router": router_config,
                            "label": "[V]",
                        },
                        "source_block": {
                            "tensor_name": source.get("tensor_name"),
                            "shape": shape,
                            "dtype": source.get("dtype"),
                            "row_start": source.get("row_start"),
                            "row_count": source.get("row_count"),
                            "bytes": source.get("bytes"),
                            "payload_sha256": source.get("payload_sha256"),
                            "label": "[V]",
                        },
                        "candidate_body": body_ref | {"body": body_record},
                        "native_kernel_parity": kernel_ref,
                        "input": {
                            "definition": f"((index * {REFERENCE_MULTIPLIER}) mod {REFERENCE_MODULUS} - {REFERENCE_OFFSET}) / {REFERENCE_MODULUS}",
                            "values": shape[1],
                            "deterministic_sha256": input_hash,
                            "label": "[V]",
                        },
                        "execution": {
                            "provider": "derived-cpu-selection",
                            "dtype": "float32",
                            "body_decoded": True,
                            "matvec": "candidate Q4/G64 body",
                            "native_matvec_parity_reference": kernel_ref["path"],
                            "source_independent": True,
                            "candidate_body_persisted": True,
                            "model_loaded": False,
                        },
                        "selection": selection,
                        "physical_graph": physical,
                        "noetic_ir": {
                            "schema": "hcli.noetic.ir.v1",
                            "semantic_type": "NoeticIR",
                            "candidate_id": body.get("candidate_id"),
                            "operations": ["load_persisted_source_independent_router_body", "derive_router_logits_with_q4_g64_matvec", "compute_fp32_router_softmax", "select_router_top_k", "normalize_selected_router_weights", "emit_selected_expert_ids_and_weights"],
                            "source_independent": True,
                            "complete_model": False,
                        },
                        "validation": {"errors": []},
                        "claim_boundary": "Derived source-independent Flash router selection executed from a persisted full Q4/G64 body using the pinned FP32 softmax/top-k contract; native Metal evidence is matvec parity only, and no source-model router-output parity, whole-model capability, complete-token runtime, EBPW, or Flash TPS claim is made.",
                        "next_action": "compare selected ids and weights against a source-model router reference, then connect the verified selection edge to expert dispatch and a protected complete-token graph",
                    })
    except Exception as exc:  # noqa: BLE001 - preserve a durable execution boundary
        errors.append(f"{type(exc).__name__}: {str(exc)[:2000]}")
        report.update({"status": "FAILED", "validation": {"errors": errors}, "claim_boundary": "Router selection execution failed; no Flash capability or performance claim is made."})
    report["generated_at"] = started
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    report["receipt_path"] = str(destination)
    atomic_write_json(destination, report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--root", help="pinned ModelLake specimen root; defaults to the body receipt root")
    parser.add_argument("--body-receipt")
    parser.add_argument("--kernel-receipt")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = run_flash_router_selection(
        repo_root=args.repo_root,
        root=args.root,
        body_receipt=args.body_receipt,
        kernel_receipt=args.kernel_receipt,
        emit=args.emit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


__all__ = [
    "DEFAULT_BODY",
    "DEFAULT_EMIT",
    "DEFAULT_KERNEL",
    "SCHEMA",
    "main",
    "reference_input",
    "run_flash_router_selection",
    "select_router",
    "stable_softmax",
]
