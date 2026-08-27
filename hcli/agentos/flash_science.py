"""Pre-runtime Flash-Next architecture and joint-promotion science.

This lane reads only pinned upstream metadata.  It creates an architecture
fingerprint, an organ graph, structural active-compute bounds, and a
representation-native Gravity/Accelerator worklist.  It does not load model
weights, compile a native executable, or promote a metadata observation into a
capability/performance claim.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hcli.flash_next import (
    ACCEPTED_CAPABILITY_PRESERVING_TPS_MIN,
    COMPLETE_SYSTEM_EBPW_MAX,
    EXPECTED_BYTES,
    EXPECTED_FILE_COUNT,
    PINNED_REVISION,
    REPO_ID,
    evaluate_flash_promotion,
)
from hcli.persist import atomic_write_json


SCHEMA = "hcli.flash-next.pre-runtime-science.v1"
UPSTREAM_SOURCE = "UPSTREAM_SOURCE"
DERIVED = "DERIVED"
LOCAL_PHYSICAL = "LOCAL_PHYSICAL"
MODEL_PAGE = f"https://huggingface.co/{REPO_ID}"
RAW_BASE = f"https://huggingface.co/{REPO_ID}/resolve/{PINNED_REVISION}/"
METADATA_FILES = (
    ("config.json", 8 * 1024 * 1024),
    ("tokenizer.json", 32 * 1024 * 1024),
    ("tokenizer_config.json", 8 * 1024 * 1024),
    ("model.safetensors.index.json", 8 * 1024 * 1024),
    ("README.md", 8 * 1024 * 1024),
    ("generation_config.json", 8 * 1024 * 1024),
    ("preprocessor_config.json", 8 * 1024 * 1024),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(name: str, limit: int, timeout_s: float) -> Dict[str, Any]:
    url = RAW_BASE + name
    request = urllib.request.Request(url, headers={"User-Agent": "hawking-hcli/1"})
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_s))) as response:
            data = response.read(limit + 1)
            content_type = response.headers.get("Content-Type")
    except Exception as exc:  # noqa: BLE001 - preserve each metadata boundary
        return {"name": name, "url": url, "label": UPSTREAM_SOURCE, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
    if len(data) > limit:
        return {"name": name, "url": url, "label": UPSTREAM_SOURCE, "status": "REFUSED_TOO_LARGE", "max_bytes": limit}
    row: Dict[str, Any] = {
        "name": name,
        "url": url,
        "revision": PINNED_REVISION,
        "label": UPSTREAM_SOURCE,
        "status": "FETCHED",
        "bytes": len(data),
        "sha256": _sha256_bytes(data),
        "content_type": content_type,
        "data": data,
    }
    if name.endswith(".json"):
        try:
            row["parsed"] = json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            row["parse_error"] = f"{type(exc).__name__}: {exc}"
    return row


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _organ_graph(config: Mapping[str, Any], weight_map: Mapping[str, Any], shard_sizes: Mapping[str, int]) -> list[Dict[str, Any]]:
    text = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
    names = [str(name) for name in weight_map]
    groups = {
        "ngram_embedding": ("ngram",),
        "moe_router": ("router", "gate"),
        "moe_experts": ("experts", "expert"),
        "moe_shared_expert": ("shared_expert", "shared.expert"),
        "deltanet": ("linear_attn", "linear_attention", "deltanet", "delta_net"),
        "qsa_sparse_attention": ("indexer", "full_attention", "q_proj", "k_proj", "v_proj"),
        "mtp": ("mtp", "multi_token"),
        "lm_head": ("lm_head", "output"),
    }
    common = {
        "hidden_size": text.get("hidden_size"),
        "layers": text.get("num_hidden_layers"),
        "experts": text.get("num_experts"),
        "experts_per_token": text.get("num_experts_per_tok"),
        "vocab_size": text.get("vocab_size"),
    }
    rows = []
    for organ, signals in groups.items():
        matched = [name for name in names if any(signal in name.lower() for signal in signals)]
        shards = sorted({str(weight_map[name]) for name in matched if name in weight_map})
        shard_bytes = sum(int(shard_sizes.get(shard) or 0) for shard in shards)
        if organ == "ngram_embedding":
            dimensions = {"ngram_vocab_size_base": text.get("ngram_vocab_size_base"), "ngram_size": text.get("ngram_size"), "split_ngram_parts": text.get("split_ngram_parts"), "hidden_size": text.get("hidden_size")}
        elif organ in {"moe_router", "moe_experts", "moe_shared_expert"}:
            dimensions = {**common, "moe_intermediate_size": text.get("moe_intermediate_size"), "shared_expert_intermediate_size": text.get("shared_expert_intermediate_size")}
        elif organ == "deltanet":
            layer_types = text.get("layer_types") if isinstance(text.get("layer_types"), list) else []
            dimensions = {"layer_types": layer_types, "linear_layers": layer_types.count("linear_attention"), "linear_key_head_dim": text.get("linear_key_head_dim"), "linear_value_head_dim": text.get("linear_value_head_dim"), "linear_num_key_heads": text.get("linear_num_key_heads"), "linear_num_value_heads": text.get("linear_num_value_heads"), "conv_kernel_dim": text.get("linear_conv_kernel_dim")}
        elif organ == "qsa_sparse_attention":
            dimensions = {"indexer_budget": text.get("indexer_budget"), "indexer_compress_ratio": text.get("indexer_compress_ratio"), "indexer_head_dim": text.get("indexer_head_dim"), "indexer_kv_heads": text.get("indexer_kv_heads"), "indexer_n_heads": text.get("indexer_n_heads"), "full_attention_interval": text.get("full_attention_interval")}
        elif organ == "mtp":
            dimensions = {"mtp": text.get("mtp"), "mtp_num_hidden_layers": text.get("mtp_num_hidden_layers"), "dedicated_embeddings": text.get("mtp_use_dedicated_embeddings")}
        else:
            dimensions = {"hidden_size": text.get("hidden_size"), "vocab_size": text.get("vocab_size")}
        rows.append({
            "id": organ,
            "label": DERIVED,
            "dimensions": dimensions,
            "tensors": len(matched),
            "tensor_name_examples": matched[:10],
            "shards": shards,
            "shard_bytes_observed": shard_bytes,
            "bytes": {"value": None, "label": LOCAL_PHYSICAL, "status": "NOT_MEASURED_TENSOR_SHAPES_NOT_IN_INDEX"},
            "active_bytes_per_token": {"value": None, "label": LOCAL_PHYSICAL, "status": "NOT_MEASURED"},
            "flops_per_token": {"value": None, "label": LOCAL_PHYSICAL, "status": "NOT_MEASURED"},
            "state_bytes": {"value": None, "label": LOCAL_PHYSICAL, "status": "NOT_MEASURED"},
            "regularity": {"value": "repeated layer/name patterns" if matched else "not observed", "label": DERIVED, "status": "STRUCTURAL_ONLY"},
            "bottleneck": {"value": "unresolved until native execution", "label": LOCAL_PHYSICAL, "status": "NOT_MEASURED"},
            "gravity": {"status": "PLAN_ONLY", "native_representation_required": True},
            "accelerator": {"status": "PLAN_ONLY", "native_kernel_required": True},
        })
    return rows


def _metadata_summary(rows: Mapping[str, Dict[str, Any]]) -> list[Dict[str, Any]]:
    result = []
    for name, _limit in METADATA_FILES:
        row = rows.get(name) or {"name": name, "label": UPSTREAM_SOURCE, "status": "MISSING"}
        result.append({key: value for key, value in row.items() if key != "data" and key != "parsed"})
    return result


def _write(report: Dict[str, Any], emit: Optional[str], repo: Path) -> None:
    destination = Path(emit).expanduser() if emit else repo / "receipts" / "headless" / "HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
    if not destination.is_absolute():
        destination = repo / destination
    report["receipt_path"] = str(destination.resolve())
    atomic_write_json(destination, report)


def run_flash_science_gate(
    *,
    repo_root: Optional[str | os.PathLike[str]] = None,
    emit: Optional[str | os.PathLike[str]] = None,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Fetch pinned metadata and emit an identity-only pre-runtime science receipt."""
    repo = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[2]
    started = time.time()
    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "qualification": "FLASH_NEXT_PRE_RUNTIME_IDENTITY_AND_PLAN_ONLY",
        "started_at": started,
        "source_identity": {
            "repo": REPO_ID,
            "page": MODEL_PAGE,
            "pinned_revision": PINNED_REVISION,
            "expected_file_count": EXPECTED_FILE_COUNT,
            "expected_complete_source_bytes": EXPECTED_BYTES,
            "labels": {"upstream": UPSTREAM_SOURCE, "derived": DERIVED, "local_physical": LOCAL_PHYSICAL},
        },
        "promotion_law": {
            "complete_system_ebpw_max": COMPLETE_SYSTEM_EBPW_MAX,
            "accepted_capability_preserving_tps_min": ACCEPTED_CAPABILITY_PRESERVING_TPS_MIN,
            "dense_parent_execution_fallback": False,
            "hidden_dense_rematerialization": False,
            "text_resident_exception": "FLASH_NEXT_TEXT_RESIDENT may omit vision tensors only when explicitly declared and qualified",
        },
    }
    fetched: Dict[str, Dict[str, Any]] = {}
    try:
        for name, limit in METADATA_FILES:
            fetched[name] = _fetch(name, limit, timeout_s)
        config_row = fetched.get("config.json") or {}
        index_row = fetched.get("model.safetensors.index.json") or {}
        config = config_row.get("parsed") if isinstance(config_row.get("parsed"), Mapping) else {}
        index = index_row.get("parsed") if isinstance(index_row.get("parsed"), Mapping) else {}
        weight_map = index.get("weight_map") if isinstance(index.get("weight_map"), Mapping) else {}
        text_config = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
        remote_sizes: Dict[str, int] = {}
        census_path = repo / "receipts" / "headless" / "HCLI_MODELLAKE_FLASH_CENSUS.json"
        census = None
        try:
            census = json.loads(census_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            census = None
        if isinstance(census, Mapping) and isinstance(census.get("remote_manifest"), Mapping):
            for item in census["remote_manifest"].get("files") or []:
                if isinstance(item, Mapping) and item.get("file"):
                    remote_sizes[str(item["file"])] = int(item.get("size") or 0)
        layer_types = text_config.get("layer_types") if isinstance(text_config.get("layer_types"), list) else []
        architecture = {
            "model_type": config.get("model_type"),
            "text_model_type": text_config.get("model_type"),
            "architectures": config.get("architectures"),
            "hidden_size": text_config.get("hidden_size"),
            "layers": text_config.get("num_hidden_layers"),
            "layer_types": layer_types,
            "full_attention_layers": layer_types.count("full_attention"),
            "linear_attention_layers": layer_types.count("linear_attention"),
            "experts": text_config.get("num_experts"),
            "experts_per_token": text_config.get("num_experts_per_tok"),
            "vocab_size": text_config.get("vocab_size"),
            "ngram_vocab_size_base": text_config.get("ngram_vocab_size_base"),
            "ngram_size": text_config.get("ngram_size"),
            "split_ngram_parts": text_config.get("split_ngram_parts"),
            "indexer_budget": text_config.get("indexer_budget"),
            "mtp": text_config.get("mtp"),
            "vision_config": config.get("vision_config"),
            "observed_index_metadata": index.get("metadata"),
        }
        architecture_fingerprint = _canonical_hash({"architecture": architecture, "tensor_names": sorted(str(name) for name in weight_map), "metadata_sha256": {name: (fetched.get(name) or {}).get("sha256") for name, _ in METADATA_FILES}})
        report.update({
            "metadata": _metadata_summary(fetched),
            "architecture_fingerprint": {"value": architecture_fingerprint, "algorithm": "sha256(canonical pinned metadata + sorted tensor names)", "label": DERIVED},
            "architecture": {**architecture, "label": DERIVED, "tensor_count": len(weight_map), "index_total_size": index.get("metadata", {}).get("total_size") if isinstance(index.get("metadata"), Mapping) else None},
            "organ_graph": _organ_graph(config, weight_map, remote_sizes),
            "active_compute_bounds": {
                "label": DERIVED,
                "layers": text_config.get("num_hidden_layers"),
                "linear_attention_layers": layer_types.count("linear_attention"),
                "full_attention_layers": layer_types.count("full_attention"),
                "routed_experts_per_token": text_config.get("num_experts_per_tok"),
                "total_experts": text_config.get("num_experts"),
                "routed_expert_fraction": (
                    (float(text_config.get("num_experts_per_tok")) / float(text_config.get("num_experts")))
                    if text_config.get("num_experts_per_tok") and text_config.get("num_experts")
                    else None
                ),
                "shared_expert_included": any("shared_expert" in str(name).lower() for name in weight_map),
                "active_bytes_per_token": {"value": None, "label": LOCAL_PHYSICAL, "status": "NOT_MEASURED"},
                "device_bytes_touched_per_token": {"value": None, "label": LOCAL_PHYSICAL, "status": "NOT_MEASURED"},
                "flops_per_token": {"value": None, "label": LOCAL_PHYSICAL, "status": "NOT_MEASURED"},
                "state_bytes_per_token": {"value": None, "label": LOCAL_PHYSICAL, "status": "NOT_MEASURED"},
            },
            "gravity_plan": [
                {"organ": "moe_experts", "hypothesis": "cross-expert shared basis plus residual", "native_kernel": "representation-native routed expert decode", "status": "PLAN_ONLY"},
                {"organ": "ngram_embedding", "hypothesis": "factorized/generative lookup rather than generic matrix quantization", "native_kernel": "ngram lookup/generator", "status": "PLAN_ONLY"},
                {"organ": "deltanet", "hypothesis": "persistent resident state-machine execution", "native_kernel": "linear-attention state update", "status": "PLAN_ONLY"},
                {"organ": "qsa_sparse_attention", "hypothesis": "indexer-budget sparse block traversal", "native_kernel": "sparse attention selection and gather", "status": "PLAN_ONLY"},
                {"organ": "mtp", "hypothesis": "accepted drafts reduce effective decode steps only when accepted work is counted", "native_kernel": "MTP accept/reject path", "status": "PLAN_ONLY"},
            ],
            "required_primitives": [
                "dense-vs-NF reference comparator",
                "shared-basis/residual decoder",
                "sparse expert router and index lookup",
                "ngram factorized lookup/generator",
                "linear recurrent-state update",
                "QSA sparse traversal",
                "MTP acceptance accounting",
                "complete-token wall meter",
            ],
            "local_physical": {
                "machine": platform.platform(),
                "architecture": platform.machine(),
                "model_lake_target_present": (repo / "receipts" / "headless" / "HCLI_MODELLAKE_FLASH_CENSUS.json").is_file() and bool((census or {}).get("flash_target_manifest", {}).get("final_present")),
                "weights_loaded": False,
                "native_executable": False,
                "native_kernel": "NOT_COMPILED",
                "label": LOCAL_PHYSICAL,
            },
            "promotion_gate": evaluate_flash_promotion(None),
            "checks": {
                "all_metadata_fetched": all((fetched.get(name) or {}).get("status") == "FETCHED" for name, _ in METADATA_FILES),
                "config_parsed": isinstance(config, Mapping) and bool(config),
                "index_parsed": isinstance(index, Mapping) and bool(weight_map),
                "pinned_metadata_urls": all((fetched.get(name) or {}).get("revision") == PINNED_REVISION for name, _ in METADATA_FILES if (fetched.get(name) or {}).get("status") == "FETCHED"),
                "no_weights_downloaded": True,
                "promotion_not_allowed": evaluate_flash_promotion(None).get("promotion_allowed") is False,
            },
        })
        report["status"] = "PASSED" if all(report["checks"].values()) else "FAILED"
    except Exception as exc:  # noqa: BLE001 - persist the metadata boundary
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        report["metadata"] = _metadata_summary(fetched)
    report["finished_at"] = time.time()
    report["elapsed_s"] = round(report["finished_at"] - started, 3)
    _write(report, str(emit) if emit is not None else None, repo)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root")
    parser.add_argument("--emit")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    report = run_flash_science_gate(repo_root=args.repo_root, emit=args.emit, timeout_s=args.timeout_s)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report.get("status") == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "run_flash_science_gate"]
