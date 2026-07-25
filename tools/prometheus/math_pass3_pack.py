#!/usr/bin/env python3.12
"""Math-Preserve PASS 3: profile-conditioned packing from PASS 2's frozen manifest.

Per-shard loop: VERIFY -> PACK (frozen coalition promoted to native, everything else
exactly as General-R0 already packs it) -> EVICT. Re-streams the source PASS 1
already evicted -- the user's own spec names this explicitly ("Re-stream every
required source shard"), so this is not a design mistake to route around: PASS 1
measures and evicts per window before PASS 2 can see the whole model, so a second
pass over the source is structural, not accidental.

Byte-matching honesty: this build promotes each layer's coalition experts to native
precision (full source bytes) and leaves every remainder expert at the same R0 rate
General-R0 already uses -- it does NOT yet demote the remainder to compensate, so
Math-Preserve.gravity as produced here is NOT byte-matched to General-R0's total.
Solving the remainder's rate to hit a matched budget is architecture.
equal_budget_solver's job (it already does exactly this kind of bisection for the
four Claim-A arms); wiring it in is the next increment, not done by this module.
That is reported explicitly in this run's receipt, not silently assumed away.

    python3.12 tools/prometheus/math_pass3_pack.py status
    python3.12 tools/prometheus/math_pass3_pack.py run [--limit-shards N]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONDENSE = REPO / "tools/condense"
for _p in (HERE, CONDENSE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MANIFEST = REPO / "PROMETHEUS_MATH_ALLOCATION_MANIFEST.json"
GRAPH = REPO / "GLM52_SHARD_DEPENDENCY_GRAPH.json"
OFFICIAL_MANIFEST = REPO / "GLM52_OFFICIAL_MANIFEST.json"

STATE_DIR = Path(
    "/Users/scammermike/Library/Application Support/Hawking/GLM52MathPrometheus/pass3"
)
SOURCE_ROOT = STATE_DIR / "source"
LEDGER = STATE_DIR / "PASS3_LEDGER.jsonl"
PROGRESS = STATE_DIR / "progress.json"
LOCK = STATE_DIR / "pass3.lock"
COMPACT = Path(
    "/Users/scammermike/Library/Application Support/Hawking/Models/GLM-5.2/"
    "b4734de4facf877f85769a911abafc5283eab3d9/Math-Preserve-PASS3"
)

os.environ.setdefault("HF_HOME", str(STATE_DIR / "hf_home"))
os.environ.setdefault("HF_HUB_CACHE", str(STATE_DIR / "hf_cache"))
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

DISK_FLOOR_BYTES = int(os.environ.get("GLM52_PASS3_DISK_FLOOR_BYTES", 75 * 10**9))
HASH_CHUNK = 16 << 20


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _append_ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _ledger_rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def verified_shards() -> set[str]:
    return {r["shard"] for r in _ledger_rows() if r.get("status") == "VERIFIED"}


def packed_shards() -> set[str]:
    if not COMPACT.exists():
        return set()
    return {p.stem + ".safetensors" for p in COMPACT.glob("*.gravity")}


def manifest_ready() -> tuple[bool, dict | None, str]:
    if not MANIFEST.exists():
        return False, None, f"{MANIFEST} does not exist yet -- run PASS 2's `freeze`"
    manifest = json.loads(MANIFEST.read_text())
    if not manifest.get("complete"):
        missing = manifest.get("sparse_layers_missing_evidence", [])
        return False, manifest, (
            f"{MANIFEST} exists but complete=false ({len(missing)} sparse layers "
            "still missing capsule evidence) -- PASS 2 should have refused to "
            "write this; do not pack against it"
        )
    return True, manifest, ""


def coalition_rate_override(manifest: dict) -> dict[tuple[int, int], str]:
    """Every coalition member across every layer, mapped to 'native'. Remainder
    experts get no entry -- absent from the map, glm52_pack.pack_shard falls
    through to its default production_rung (R0), unchanged from General-R0."""
    override: dict[tuple[int, int], str] = {}
    for layer_str, data in manifest["per_layer"].items():
        layer = int(layer_str)
        for expert in data["coalition_expert_ids"]:
            override[(layer, expert)] = "native"
    return override


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _free_bytes() -> int:
    import shutil

    return shutil.disk_usage(str(SOURCE_ROOT if SOURCE_ROOT.exists() else STATE_DIR)).free


def _fetch_one(row: dict, repo: str, revision: str) -> dict:
    from huggingface_hub import hf_hub_download

    name = row["path"]
    started = time.time()
    got_path = Path(hf_hub_download(
        repo_id=repo, filename=name, revision=revision,
        local_dir=str(SOURCE_ROOT), token=False,
    ))
    elapsed = max(time.time() - started, 1e-6)
    size = got_path.stat().st_size
    if size != row["logical_bytes"]:
        quarantine = got_path.with_suffix(got_path.suffix + ".badsize")
        os.replace(got_path, quarantine)
        return {"shard": name, "status": "SIZE_MISMATCH", "at": _now()}
    observed = _sha256_file(got_path)
    if observed != row["lfs_sha256"]:
        quarantine = got_path.with_suffix(got_path.suffix + ".badhash")
        os.replace(got_path, quarantine)
        return {"shard": name, "status": "HASH_MISMATCH", "at": _now()}
    return {
        "shard": name, "status": "VERIFIED", "bytes": size, "sha256": observed,
        "seconds": round(elapsed, 2),
        "megabits_per_second": round(size * 8 / elapsed / 1e6, 1), "at": _now(),
    }


def run(*, limit_shards: int | None = None) -> int:
    import fcntl

    import glm52_pack as pack

    ready, manifest, reason = manifest_ready()
    if not ready:
        raise SystemExit(reason)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(LOCK), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.stderr.write("another PASS3 run holds the lock; exiting\n")
        return 0

    override = coalition_rate_override(manifest)
    official = _read_json(OFFICIAL_MANIFEST)
    graph = _read_json(GRAPH)
    repo, revision = official["repo"], official["revision"]
    by_path = {f["path"]: f for f in official["files"] if f.get("is_weight")}
    tensors_by_shard: dict[str, list[dict]] = {}
    for tensor in graph["tensors"]:
        tensors_by_shard.setdefault(tensor["shard"], []).append(tensor)

    shard_names = sorted(by_path)
    if limit_shards is not None:
        shard_names = shard_names[:limit_shards]

    already_packed = packed_shards()
    for name in shard_names:
        if name in already_packed:
            continue
        if _free_bytes() < DISK_FLOOR_BYTES:
            _append_ledger({"event": "DISK_FLOOR_STOP", "shard": name, "at": _now()})
            sys.stderr.write(f"disk floor reached before {name}; stopping\n")
            break

        if name not in verified_shards():
            result = _fetch_one(by_path[name], repo, revision)
            _append_ledger(result)
            if result["status"] != "VERIFIED":
                sys.stderr.write(f"PASS3 fetch failed: {result}\n")
                continue

        rows = tensors_by_shard.get(name, [])
        receipt = pack.pack_shard(
            SOURCE_ROOT / name, rows, COMPACT, rate_override=override,
        )
        _append_ledger({"event": "PACKED", "shard": name,
                        "compact_bytes": receipt["compact_bytes"],
                        "complete_bpw": receipt["complete_bpw"], "at": _now()})

        target = SOURCE_ROOT / name
        if target.exists():
            size = target.stat().st_size
            target.unlink()
            _append_ledger({"event": "EVICT", "shard": name, "bytes": size, "at": _now()})

        _write_json(PROGRESS, {
            "shards_packed": len(packed_shards()), "shards_total": len(shard_names),
            "last_shard": name, "at": _now(),
        })

    return 0


def status() -> dict:
    ready, manifest, reason = manifest_ready()
    return {
        "schema": "hawking.prometheus.math_pass3_status.v1",
        "at": _now(),
        "ready_to_pack": ready,
        "reason": reason if not ready else "manifest complete; run `run` to pack",
        "manifest_layers": len(manifest["per_layer"]) if manifest else 0,
        "coalition_tensors_total": (
            sum(len(d["coalition_expert_ids"]) for d in manifest["per_layer"].values()) * 3
            if manifest else 0
        ),
        "shards_packed": len(packed_shards()),
        "shards_total": 282,
        "byte_matched_to_general_r0": False,
        "byte_matching_note": "coalition promoted to native, remainder left at R0's rate "
                               "unchanged -- not yet demoted to match General-R0's total "
                               "bytes; see module docstring",
    }


def selftest() -> None:
    """No manifest required: proves the override-building logic against a synthetic
    one, independent of whatever PASS 2 has produced so far."""
    synthetic = {
        "per_layer": {
            "3": {"coalition_expert_ids": [7, 200, 15]},
            "10": {"coalition_expert_ids": [0, 255]},
        },
    }
    override = coalition_rate_override(synthetic)
    assert override == {
        (3, 7): "native", (3, 200): "native", (3, 15): "native",
        (10, 0): "native", (10, 255): "native",
    }, override
    assert (3, 8) not in override, "an expert not in the coalition must have no entry"
    assert (4, 7) not in override, "the same expert id in a different layer is a different key"
    print("math_pass3_pack selftest PASS")


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "status"
    if command == "status":
        print(json.dumps(status(), indent=2, sort_keys=True))
        return 0
    if command == "selftest":
        selftest()
        return 0
    if command == "run":
        limit = None
        if "--limit-shards" in argv:
            limit = int(argv[argv.index("--limit-shards") + 1])
        return run(limit_shards=limit)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
