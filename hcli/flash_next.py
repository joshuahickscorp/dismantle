"""Pinned identity and staging plan for Qwen3.8-Flash-Next.

This module intentionally does not download weights.  It records the exact
upstream revision, the expected ModelLake acquisition, and the organ worklist
so a later acquisition can be resumed and verified without treating an
un-staged model as a qualified resident.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .persist import atomic_write_json
from .providers import CapabilityContract, ResidentProfile


REPO_ID = "Qwen/Qwen3.8-Flash-Next"
REQUESTED_REVISION = "main"
PINNED_REVISION = "34567a4712bc9766c4449e2e98e4468bfa24d915"
MODEL_PAGE = f"https://huggingface.co/{REPO_ID}"
MODEL_TREE = f"{MODEL_PAGE}/tree/{PINNED_REVISION}"
EXPECTED_BYTES = 360 * 10**9
EXPECTED_SAFETENSOR_SHARDS = 131
SCHEMA = "hcli.flash-next.identity.v1"

ORGANS = (
    {"id": "ngram_embedding", "status": "IDENTITY_ONLY", "target": "51B ngram path / 20M ngram vocabulary"},
    {"id": "moe_router", "status": "IDENTITY_ONLY", "target": "512 experts, 10 routed + shared"},
    {"id": "moe_shared_expert", "status": "IDENTITY_ONLY", "target": "shared expert path"},
    {"id": "deltanet", "status": "IDENTITY_ONLY", "target": "3 DeltaNet blocks per repeated group"},
    {"id": "qsa_sparse_attention", "status": "IDENTITY_ONLY", "target": "512-block / 2048-token QSA budget"},
    {"id": "mtp", "status": "IDENTITY_ONLY", "target": "4B multi-token prediction head"},
    {"id": "lm_head", "status": "IDENTITY_ONLY", "target": "output projection / vocabulary head"},
)


def _sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _local_observation(root: Optional[Path]) -> Dict[str, Any]:
    if root is None:
        return {"status": "NOT_STAGED", "reason": "set HCLI_FLASH_NEXT_ROOT to inspect a local acquisition"}
    root = root.expanduser().resolve()
    if not root.is_dir():
        return {"status": "NOT_STAGED", "path": str(root), "reason": "configured root does not exist"}
    total = 0
    files = 0
    for path in root.rglob("*"):
        if path.is_file():
            files += 1
            try:
                total += path.stat().st_size
            except OSError:
                pass
    config = root / "config.json"
    config_data: Dict[str, Any] = {}
    if config.is_file():
        try:
            loaded = json.loads(config.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config_data = loaded
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    return {
        "status": "PRESENT_UNVERIFIED",
        "path": str(root),
        "bytes": total,
        "file_count": files,
        "config_sha256": _sha256(config) if config.is_file() else None,
        "config_model_type": config_data.get("model_type"),
        "revision_verified": False,
        "weights_verified": False,
    }


def model_lake_plan() -> Dict[str, Any]:
    root = Path("/Volumes/corpdrive/hawking-modellake")
    mounted = root.is_dir()
    free = None
    if mounted:
        try:
            free = shutil.disk_usage(root).free
        except OSError:
            free = None
    return {
        "status": "READY_FOR_EXPLICIT_ACQUIRE" if mounted and (free is None or free > EXPECTED_BYTES) else "REFUSED_NOT_READY",
        "root": str(root),
        "mounted": mounted,
        "expected_bytes": EXPECTED_BYTES,
        "free_bytes_observed": free,
        "tier": 2,
        "resumable": True,
        "hash_verified": False,
        "hash_status": "NOT_RUN",
        "hash_policy": "verify every acquired shard before atomic publish",
        "atomic_publish": True,
        "command": [
            "hf", "download", REPO_ID, "--revision", PINNED_REVISION,
            "--local-dir", "<ModelLake partial destination>",
        ],
        "human_confirmation_required": True,
        "download_performed": False,
        "reason": "weights are not downloaded by identity census",
    }


def flash_next_profile(local_root: Optional[str | os.PathLike[str]] = None) -> Dict[str, Any]:
    root_value = local_root or os.environ.get("HCLI_FLASH_NEXT_ROOT")
    local = _local_observation(Path(root_value) if root_value else None)
    profile = ResidentProfile(
        profile_id=f"hf:{REPO_ID}@{PINNED_REVISION[:12]}",
        provider="huggingface",
        model_id=REPO_ID,
        artifact={
            "source_repo": REPO_ID,
            "requested_revision": REQUESTED_REVISION,
            "pinned_revision": PINNED_REVISION,
            "format": "transformers",
            "expected_bytes": EXPECTED_BYTES,
            "expected_safetensor_shards": EXPECTED_SAFETENSOR_SHARDS,
            "local": local,
        },
        tokenizer={"status": "UPSTREAM_DECLARED", "source": MODEL_PAGE, "authoritative_after_acquire": True},
        runtime={
            "status": "NOT_COMPILED",
            "compatible_runtimes": ["transformers", "vllm", "sglang"],
            "native_hawking_resident": "NOT_IMPLEMENTED",
        },
        compiler={"status": "NOT_RUN", "required_before_promotion": True},
        representation={"status": "BF16_MODEL_LAKE_TARGET", "organs": list(ORGANS)},
        capabilities=CapabilityContract.from_mapping({
            "schema": "hcli.provider.capabilities.v1",
            "features": {
                "vision": {"state": "unknown", "enforcement": "unknown", "source": "identity only"},
                "tool_calling": {"state": "unknown", "enforcement": "unknown", "source": "identity only"},
                "streaming": {"state": "unknown", "enforcement": "unknown", "source": "identity only"},
            },
        }),
        prompt_contract={"status": "UPSTREAM_CHAT_TEMPLATE_NOT_EXECUTED", "source": MODEL_PAGE},
        generation={"status": "UPSTREAM_GENERATION_CONFIG_NOT_EXECUTED", "source": MODEL_PAGE},
        limits={"context_length": 262144, "native_extensible_context_length": 1000000},
        fallbacks=["no local weights", "no native resident", "no capability qualification"],
        hot_bytes=None,
        machine_genome={},
        receipts=[],
        qualification={"status": "IDENTITY_ONLY", "promotion_allowed": False, "reason": "no local weights or runtime qualification"},
        metadata={
            "model_page": MODEL_PAGE,
            "model_tree": MODEL_TREE,
            "architecture": {
                "model_type": "qwen4_exp",
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "hidden_size": 2560,
                "layers": 48,
                "experts": 512,
                "routed_experts_per_token": 10,
                "vocab_size": 248320,
                "ngram_vocab_size": 20000000,
            },
            "observed_at": time.time(),
        },
    )
    return {
        "schema": SCHEMA,
        "profile": profile.to_dict(),
        "model_lake": model_lake_plan(),
        "transfer_ledger": {"source": MODEL_PAGE, "revision": PINNED_REVISION, "bytes_transferred": 0, "verified_bytes": 0},
        "organ_census": list(ORGANS),
        "download_performed": False,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root")
    parser.add_argument("--emit")
    args = parser.parse_args(argv)
    report = flash_next_profile(args.local_root)
    if args.emit:
        atomic_write_json(Path(args.emit).expanduser(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_BYTES",
    "EXPECTED_SAFETENSOR_SHARDS",
    "MODEL_PAGE",
    "ORGANS",
    "PINNED_REVISION",
    "REPO_ID",
    "flash_next_profile",
    "main",
    "model_lake_plan",
]
