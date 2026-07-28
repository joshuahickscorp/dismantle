#!/usr/bin/env python3.12
"""GLM-5.2 route-population census from retained teacher top-k metadata.

Route-metadata census only. Does not fetch or read parent weight bodies,
does not rehydrate experts, and does not claim representation capability.

Reads only ``layer_NN/topk_indices.npy`` members from the 33 retained capsule
``.npz`` archives via member-selective ``zipfile`` access. Whole archives are
never decompressed and whole capsule files are never hashed in this module
(sealed sidecar hashes are bound instead).

    python3.12 tools/condense/glm52_route_population_census.py selftest
    python3.12 tools/condense/glm52_route_population_census.py census
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import glm52_activation_aware_pack_v2 as v2  # noqa: E402

# ---------------------------------------------------------------------------
# Identity / sealed constants
# ---------------------------------------------------------------------------
SCHEMA = "hawking.glm52.route_population_census.v1"
FORMAT_VERSION = 1
SEED = 0xC2A5_C0DE

DEFAULT_CAPSULE_DIR = Path(
    "/Users/scammermike/Library/Application Support/"
    "Hawking/GLM52Gravity/source_fetch/teacher/capsules"
)
SOURCE_HEADERS = v2.SOURCE_HEADERS
PILOT_RECEIPT = v2.PILOT_RECEIPT
CONTROLLER_RESEAL = v2.CONTROLLER_RESEAL
V2_FEASIBILITY = REPO / "GLM52_V2_PROGRAM_FEASIBILITY.json"
DEFAULT_OUT_JSON = REPO / "GLM52_ROUTE_POPULATION_CENSUS.json"
DEFAULT_OUT_MD = REPO / "GLM52_ROUTE_POPULATION_CENSUS.md"

N_CAPTURE_ROWS = 4096  # 16 * 256
TOP_K = 8
N_EXPERTS = 256
CAPTURE_SHAPE = (16, 256, 8)
FLAT_SHAPE = (4096, 8)
EXPECTED_ROUTE_SUM = N_CAPTURE_ROWS * TOP_K  # 32768

DENSE_LAYERS = frozenset(range(0, 3))  # 0-2; outside routed-expert census
ROUTED_LAYER_FIRST = 3
ROUTED_LAYER_LAST = 78  # inclusive; full MoE span
EXPECTED_COVERED_FIRST = 3
EXPECTED_COVERED_LAST = 77  # layer 78 missing from retained capsules
UNOBSERVED_LAYER = 78

# Sealed five-shard basis pilot route-count anchors (not quality labels).
PROMOTION_PANEL_MIN_ROUTE = 2577
LOW_TRAFFIC_DIAGNOSTIC_ROUTE = 205

BAND_PROMOTION = "PROMOTION_PANEL_ROUTE_RANGE"
BAND_BETWEEN = "BETWEEN_PILOT_ANCHORS"
BAND_BELOW = "BELOW_LOW_TRAFFIC_ANCHOR"
BAND_ZERO = "ZERO_ROUTE"
BAND_UNOBSERVED = "UNOBSERVED"

TOPK_MEMBER_RE = re.compile(r"^layer_(\d{2})/topk_indices\.npy$")
LAYER_FROM_MEMBER_RE = re.compile(r"layer_(\d+)/topk_indices\.npy$")
# Sealed sidecar array hashes are lowercase 64-hex SHA-256 digests.
SEALED_ARRAY_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_N_LOADED_TOPK_MEMBERS = 87

SAFETY_FENCES: dict[str, bool] = {
    "RAMANUJAN_RESEARCH_AUTHORIZED": False,
    "HIDE_KERNEL_TURN": False,
    "ODYSSEY_LAUNCH_AUTHORIZED": False,
    "full_parent_traversal_started": False,
    "full_traversal_authorized": False,
    "capable_artifact_claimed": False,
    "MOP_touched": False,
}


class CensusError(RuntimeError):
    """Hard census failure (fail closed)."""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(blob.encode("utf-8"))


def sha256_array(arr: np.ndarray) -> str:
    """Stable hash of a numpy array (dtype + shape + contiguous bytes)."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(b"|")
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(b"|")
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_array_raw_bytes(arr: np.ndarray) -> str:
    """SHA-256 of C-contiguous raw array bytes only (no dtype/shape prefix).

    Matches sealed teacher-capsule ``array_sha256`` values for top-k members.
    Reshape that preserves C-order does not change the digest.
    """
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Capsule inventory + member-selective top-k loading
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TopkMemberRef:
    capsule_path: Path
    capsule_filename: str
    member_name: str
    layer: int


def list_capsule_npz_files(capsule_dir: Path) -> list[Path]:
    if not capsule_dir.is_dir():
        raise CensusError(f"capsule dir missing: {capsule_dir}")
    files = sorted(capsule_dir.glob("*.npz"))
    if len(files) != 33:
        raise CensusError(
            f"expected 33 retained capsule .npz files, found {len(files)} in {capsule_dir}"
        )
    return files


def enumerate_topk_members(capsule_dir: Path) -> list[TopkMemberRef]:
    """Enumerate every ``layer_NN/topk_indices.npy`` across the 33 capsules.

    Only lists member names from the zip central directory; does not decompress
    archive bodies.
    """
    refs: list[TopkMemberRef] = []
    for path in list_capsule_npz_files(capsule_dir):
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                m = TOPK_MEMBER_RE.match(name) or LAYER_FROM_MEMBER_RE.search(name)
                if m is None:
                    # Accept only exact topk_indices.npy under layer_NN/
                    if name.endswith("/topk_indices.npy") and name.startswith("layer_"):
                        raise CensusError(f"unparseable topk member {name!r} in {path.name}")
                    continue
                if not name.endswith("topk_indices.npy"):
                    continue
                # Prefer strict layer_NN/ form
                mm = TOPK_MEMBER_RE.match(name)
                if mm is None:
                    raise CensusError(f"topk member must be layer_NN/topk_indices.npy, got {name!r}")
                layer = int(mm.group(1))
                refs.append(
                    TopkMemberRef(
                        capsule_path=path,
                        capsule_filename=path.name,
                        member_name=name,
                        layer=layer,
                    )
                )
    refs.sort(key=lambda r: (r.layer, r.capsule_filename, r.member_name))
    return refs


def load_topk_member_bytes(
    capsule_path: Path,
    member_name: str,
    *,
    zip_open: Callable[[Path], zipfile.ZipFile] | None = None,
) -> bytes:
    """Read exactly one zip member stream; never whole-archive decompress.

    Direct calls accept only the exact ``layer_NN/topk_indices.npy`` pattern
    (two-digit layer). An arbitrary path that merely ends in
    ``topk_indices.npy`` is refused.
    """
    if TOPK_MEMBER_RE.match(member_name) is None:
        raise CensusError(f"refusing non-topk member load: {member_name!r}")
    opener = zip_open or (lambda p: zipfile.ZipFile(p, "r"))
    with opener(capsule_path) as zf:
        # Fail closed if caller asks for anything else via ZipFile API misuse:
        names = set(zf.namelist())
        if member_name not in names:
            raise CensusError(f"member {member_name!r} absent from {capsule_path.name}")
        return zf.read(member_name)


def load_topk_array(
    capsule_path: Path,
    member_name: str,
    *,
    zip_open: Callable[[Path], zipfile.ZipFile] | None = None,
) -> tuple[np.ndarray, bytes]:
    raw = load_topk_member_bytes(capsule_path, member_name, zip_open=zip_open)
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    return arr, raw


def normalize_topk(arr: np.ndarray) -> np.ndarray:
    """Normalize capture shape to ``[4096, 8]``; reject all other shapes.

    Accepts ``[16, 256, 8]`` or already-flat ``[4096, 8]``. Validates integer
    dtype, expert IDs in ``[0, 255]``, 4096 rows, top-k width 8, and no
    duplicate expert ID within a row.
    """
    a = np.asarray(arr)
    if a.shape == CAPTURE_SHAPE:
        flat = a.reshape(N_CAPTURE_ROWS, TOP_K)
    elif a.shape == FLAT_SHAPE:
        flat = a
    else:
        raise CensusError(
            f"invalid topk shape {a.shape}; accept only {CAPTURE_SHAPE} or {FLAT_SHAPE}"
        )
    if not np.issubdtype(flat.dtype, np.integer):
        raise CensusError(f"topk dtype must be integer, got {flat.dtype}")
    if flat.shape != FLAT_SHAPE:
        raise CensusError(f"normalized shape must be {FLAT_SHAPE}, got {flat.shape}")
    if flat.min() < 0 or flat.max() > 255:
        raise CensusError(
            f"expert IDs must be in [0, 255], got min={int(flat.min())} max={int(flat.max())}"
        )
    # Per-row uniqueness: each of 8 slots distinct.
    # Sort along k and check adjacent equality (vectorized).
    sorted_rows = np.sort(flat, axis=1)
    if bool(np.any(sorted_rows[:, 1:] == sorted_rows[:, :-1])):
        raise CensusError("duplicate expert ID within a top-k row; fail closed")
    return np.ascontiguousarray(flat)


def bind_sealed_capsule_hash(capsule_path: Path) -> dict[str, Any]:
    """Bind already-sealed capsule hash from sidecar JSON; do not hash the npz."""
    sidecar = capsule_path.with_suffix(".json")
    out: dict[str, Any] = {
        "capsule_filename": capsule_path.name,
        "sidecar_json": sidecar.name if sidecar.exists() else None,
        "capsule_sha256_sealed": None,
        "seal_sha256": None,
        "array_sha256_topk_indices_sealed": {},
        "whole_capsule_hash_recomputed": False,
    }
    if not sidecar.exists():
        return out
    with open(sidecar, encoding="utf-8") as f:
        doc = json.load(f)
    out["capsule_sha256_sealed"] = doc.get("capsule_sha256")
    out["seal_sha256"] = doc.get("seal_sha256")
    arr_hashes = doc.get("array_sha256") or {}
    for k, v in arr_hashes.items():
        if k.endswith("topk_indices") or k.endswith("topk_indices.npy"):
            out["array_sha256_topk_indices_sealed"][k] = v
    return out


def resolve_sealed_topk_array_hash(
    array_sha256_map: dict[str, Any],
    member_name: str,
) -> tuple[str, str]:
    """Resolve sealed sidecar key for a loaded top-k member.

    Accepts the established ``layer_NN/topk_indices`` form and an optional
    ``.npy`` suffix. Requires a present lowercase 64-hex digest; fail closed
    on absence or malformation.
    """
    if TOPK_MEMBER_RE.match(member_name) is None:
        raise CensusError(
            f"cannot resolve sealed array hash for non-exact topk member {member_name!r}"
        )
    base = member_name[: -len(".npy")]  # layer_NN/topk_indices
    # Prefer established no-suffix key; also accept the .npy form.
    candidates = (base, member_name)
    key: str | None = None
    sealed: Any = None
    for candidate in candidates:
        if candidate in array_sha256_map:
            key = candidate
            sealed = array_sha256_map[candidate]
            break
    if key is None:
        raise CensusError(
            f"missing sealed array_sha256 for top-k member {member_name!r} "
            f"(tried {list(candidates)})"
        )
    if not isinstance(sealed, str) or not SEALED_ARRAY_HASH_RE.fullmatch(sealed):
        raise CensusError(
            f"malformed sealed array_sha256 for {key!r}: {sealed!r} "
            f"(require lowercase 64-hex)"
        )
    return key, sealed


def verify_loaded_topk_against_sealed(
    arr: np.ndarray,
    *,
    member_name: str,
    sealed_capsule: dict[str, Any],
) -> dict[str, Any]:
    """Prove loaded top-k raw bytes match the sealed sidecar array hash.

    Hashes only the in-memory C-contiguous array body — never the whole capsule
    file and never other members.
    """
    sealed_map = sealed_capsule.get("array_sha256_topk_indices_sealed") or {}
    sidecar_key, sealed_hex = resolve_sealed_topk_array_hash(sealed_map, member_name)
    computed = sha256_array_raw_bytes(arr)
    if computed != sealed_hex:
        raise CensusError(
            f"loaded top-k array bytes sha256 mismatch for {member_name!r}: "
            f"computed={computed} sealed={sealed_hex} (sidecar_key={sidecar_key!r})"
        )
    return {
        "sealed_array_sidecar_key": sidecar_key,
        "sealed_array_sha256": sealed_hex,
        "computed_array_bytes_sha256": computed,
        "sealed_array_hash_verified": True,
    }


# ---------------------------------------------------------------------------
# Validation / canonical selection
# ---------------------------------------------------------------------------
@dataclass
class LoadedTopk:
    ref: TopkMemberRef
    original_shape: list[int]
    dtype: str
    normalized: np.ndarray
    member_sha256: str
    normalized_sha256: str
    sealed_capsule: dict[str, Any]
    sealed_array_sha256: str
    computed_array_bytes_sha256: str
    sealed_array_hash_verified: bool
    sealed_array_sidecar_key: str


def load_validated_topk(
    ref: TopkMemberRef,
    *,
    zip_open: Callable[[Path], zipfile.ZipFile] | None = None,
) -> LoadedTopk:
    arr, raw = load_topk_array(ref.capsule_path, ref.member_name, zip_open=zip_open)
    original_shape = list(arr.shape)
    dtype = str(arr.dtype)
    sealed_capsule = bind_sealed_capsule_hash(ref.capsule_path)
    # Verify raw loaded array bytes against sealed sidecar before normalize.
    proof = verify_loaded_topk_against_sealed(
        arr, member_name=ref.member_name, sealed_capsule=sealed_capsule
    )
    normalized = normalize_topk(arr)
    return LoadedTopk(
        ref=ref,
        original_shape=original_shape,
        dtype=dtype,
        normalized=normalized,
        member_sha256=sha256_bytes(raw),
        normalized_sha256=sha256_array(normalized),
        sealed_capsule=sealed_capsule,
        sealed_array_sha256=proof["sealed_array_sha256"],
        computed_array_bytes_sha256=proof["computed_array_bytes_sha256"],
        sealed_array_hash_verified=bool(proof["sealed_array_hash_verified"]),
        sealed_array_sidecar_key=proof["sealed_array_sidecar_key"],
    )


def group_by_layer(loaded: Sequence[LoadedTopk]) -> dict[int, list[LoadedTopk]]:
    by: dict[int, list[LoadedTopk]] = defaultdict(list)
    for item in loaded:
        by[item.ref.layer].append(item)
    return dict(by)


def check_duplicate_agreement(by_layer: dict[int, list[LoadedTopk]]) -> dict[str, Any]:
    """Overlapping copies must be byte-identical on normalized top-k; else fail closed."""
    conflicts: list[dict[str, Any]] = []
    agreements: list[dict[str, Any]] = []
    for layer in sorted(by_layer):
        items = by_layer[layer]
        if len(items) < 2:
            continue
        hashes = {it.normalized_sha256 for it in items}
        rec = {
            "layer": layer,
            "n_copies": len(items),
            "members": [
                {
                    "capsule_filename": it.ref.capsule_filename,
                    "member_name": it.ref.member_name,
                    "normalized_sha256": it.normalized_sha256,
                }
                for it in sorted(
                    items, key=lambda x: (x.ref.capsule_filename, x.ref.member_name)
                )
            ],
        }
        if len(hashes) != 1:
            conflicts.append(rec)
        else:
            agreements.append({**rec, "normalized_sha256": next(iter(hashes))})
    if conflicts:
        raise CensusError(
            f"duplicate layer top-k conflict (fail closed): layers "
            f"{[c['layer'] for c in conflicts]}"
        )
    return {
        "n_layers_with_duplicates": len(agreements),
        "agreements": agreements,
        "conflicts": conflicts,
        "all_duplicates_byte_identical": True,
    }


def select_canonical_members(
    by_layer: dict[int, list[LoadedTopk]],
) -> dict[int, LoadedTopk]:
    """Deterministic canonical member per layer: min (capsule_filename, member_name)."""
    canonical: dict[int, LoadedTopk] = {}
    for layer, items in by_layer.items():
        chosen = min(
            items,
            key=lambda it: (it.ref.capsule_filename, it.ref.member_name),
        )
        canonical[layer] = chosen
    return canonical


# ---------------------------------------------------------------------------
# Route counts + evidence bands
# ---------------------------------------------------------------------------
def classify_evidence_band(route_count: int | None, *, unobserved: bool = False) -> str:
    if unobserved:
        return BAND_UNOBSERVED
    assert route_count is not None
    c = int(route_count)
    if c >= PROMOTION_PANEL_MIN_ROUTE:
        return BAND_PROMOTION
    if c >= LOW_TRAFFIC_DIAGNOSTIC_ROUTE:
        return BAND_BETWEEN
    if c >= 1:
        return BAND_BELOW
    if c == 0:
        return BAND_ZERO
    raise CensusError(f"invalid route_count {c}")


def count_routes_for_layer(topk: np.ndarray) -> np.ndarray:
    """Return route_count[expert_id] over 256 experts for one [4096, 8] topk."""
    flat = normalize_topk(topk)
    if flat.shape != FLAT_SHAPE:
        raise CensusError("internal: expected flat topk")
    # bincount over all selected expert ids
    counts = np.bincount(flat.reshape(-1), minlength=N_EXPERTS)
    if counts.shape[0] != N_EXPERTS:
        # if max id < 255, bincount may be shorter only if minlength not used
        counts = counts[:N_EXPERTS]
    if int(counts.sum()) != EXPECTED_ROUTE_SUM:
        raise CensusError(
            f"route conservation failed: sum={int(counts.sum())} != {EXPECTED_ROUTE_SUM}"
        )
    return counts.astype(np.int64)


def build_expert_records(
    canonical: dict[int, LoadedTopk],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build 19,200 covered records + 256 unobserved layer-78 records."""
    records: list[dict[str, Any]] = []
    per_layer_sums: dict[int, int] = {}
    covered_layers = sorted(
        L for L in canonical if EXPECTED_COVERED_FIRST <= L <= EXPECTED_COVERED_LAST
    )
    dense_topk_layers = sorted(L for L in canonical if L in DENSE_LAYERS)

    for layer in covered_layers:
        item = canonical[layer]
        counts = count_routes_for_layer(item.normalized)
        per_layer_sums[layer] = int(counts.sum())
        for eid in range(N_EXPERTS):
            rc = int(counts[eid])
            band = classify_evidence_band(rc)
            records.append(
                {
                    "layer": layer,
                    "expert_id": eid,
                    "route_count": rc,
                    "route_fraction": rc / N_CAPTURE_ROWS,
                    "zero_route": rc == 0,
                    "evidence_band": band,
                    "observed": True,
                    "canonical_capsule_filename": item.ref.capsule_filename,
                    "canonical_member_name": item.ref.member_name,
                    "normalized_topk_sha256": item.normalized_sha256,
                }
            )

    # Layer 78: explicitly UNOBSERVED, never imputed.
    for eid in range(N_EXPERTS):
        records.append(
            {
                "layer": UNOBSERVED_LAYER,
                "expert_id": eid,
                "route_count": None,
                "route_fraction": None,
                "zero_route": None,
                "evidence_band": BAND_UNOBSERVED,
                "observed": False,
                "canonical_capsule_filename": None,
                "canonical_member_name": None,
                "normalized_topk_sha256": None,
            }
        )

    # Sort for determinism: layer, expert_id
    records.sort(key=lambda r: (int(r["layer"]), int(r["expert_id"])))

    covered_n = sum(1 for r in records if r["observed"])
    unobserved_n = sum(1 for r in records if not r["observed"])
    if covered_n != 75 * N_EXPERTS:
        raise CensusError(f"expected 19200 covered records, got {covered_n}")
    if unobserved_n != N_EXPERTS:
        raise CensusError(f"expected 256 unobserved records, got {unobserved_n}")

    # Exact conservation across covered layers
    if any(s != EXPECTED_ROUTE_SUM for s in per_layer_sums.values()):
        bad = {L: s for L, s in per_layer_sums.items() if s != EXPECTED_ROUTE_SUM}
        raise CensusError(f"per-layer route conservation failed: {bad}")

    coverage = {
        "dense_layers_outside_routed_census": sorted(DENSE_LAYERS),
        "dense_layers_with_topk_members": dense_topk_layers,
        "routed_moe_layers_inclusive": [ROUTED_LAYER_FIRST, ROUTED_LAYER_LAST],
        "expected_covered_layers_inclusive": [
            EXPECTED_COVERED_FIRST,
            EXPECTED_COVERED_LAST,
        ],
        "covered_layers": covered_layers,
        "missing_routed_layers": [
            L
            for L in range(ROUTED_LAYER_FIRST, ROUTED_LAYER_LAST + 1)
            if L not in covered_layers
        ],
        "unobserved_layer": UNOBSERVED_LAYER,
        "n_covered_layers": len(covered_layers),
        "n_covered_expert_records": covered_n,
        "n_unobserved_expert_records": unobserved_n,
        "per_layer_route_sum": {str(k): v for k, v in sorted(per_layer_sums.items())},
        "expected_per_layer_route_sum": EXPECTED_ROUTE_SUM,
        "all_covered_layers_route_sum_ok": all(
            s == EXPECTED_ROUTE_SUM for s in per_layer_sums.values()
        ),
        "layer_78_imputed": False,
    }
    if coverage["missing_routed_layers"] != [UNOBSERVED_LAYER]:
        raise CensusError(
            f"expected only layer 78 missing from routed coverage, got "
            f"{coverage['missing_routed_layers']}"
        )
    if covered_layers != list(range(EXPECTED_COVERED_FIRST, EXPECTED_COVERED_LAST + 1)):
        raise CensusError(
            f"covered layers must be exactly 3..77, got {covered_layers[0]}..{covered_layers[-1]} "
            f"(n={len(covered_layers)})"
        )
    return records, coverage


def quantiles(values: Sequence[int], qs: Sequence[float] | None = None) -> dict[str, float]:
    if not values:
        return {}
    if qs is None:
        qs = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
    arr = np.asarray(list(values), dtype=np.float64)
    out: dict[str, float] = {}
    for q in qs:
        out[f"p{int(round(q * 100)):02d}"] = float(np.quantile(arr, q))
    out["mean"] = float(arr.mean())
    out["std"] = float(arr.std())
    out["min"] = float(arr.min())
    out["max"] = float(arr.max())
    out["n"] = int(arr.size)
    return out


def summarize_bands(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    covered = [r for r in records if r["observed"]]
    counts = Counter(r["evidence_band"] for r in covered)
    unobserved = sum(1 for r in records if r["evidence_band"] == BAND_UNOBSERVED)
    route_vals = [int(r["route_count"]) for r in covered]
    # Exact threshold counts (preregistered anchors + boundary probes).
    thresholds = {
        "eq_0": sum(1 for c in route_vals if c == 0),
        "eq_1": sum(1 for c in route_vals if c == 1),
        "eq_204": sum(1 for c in route_vals if c == 204),
        "eq_205": sum(1 for c in route_vals if c == 205),
        "eq_2576": sum(1 for c in route_vals if c == 2576),
        "eq_2577": sum(1 for c in route_vals if c == 2577),
        "ge_1": sum(1 for c in route_vals if c >= 1),
        "ge_205": sum(1 for c in route_vals if c >= 205),
        "ge_2577": sum(1 for c in route_vals if c >= 2577),
        "lt_205": sum(1 for c in route_vals if c < 205),
        "lt_2577": sum(1 for c in route_vals if c < 2577),
    }
    # Compact histogram of route counts (value -> frequency), sorted keys as str.
    hist_counter = Counter(route_vals)
    # Full histogram is large; keep both full and binned.
    bins = [0, 1, 10, 50, 100, 205, 500, 1000, 2577, 4096, 4097]
    bin_labels = []
    bin_counts = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            lab = "0"
            n = sum(1 for c in route_vals if c == 0)
        elif i == 1:
            lab = "1-9"
            n = sum(1 for c in route_vals if 1 <= c < 10)
        else:
            lab = f"{lo}-{hi - 1}"
            n = sum(1 for c in route_vals if lo <= c < hi)
        bin_labels.append(lab)
        bin_counts.append(n)

    per_layer: dict[str, Any] = {}
    by_layer: dict[int, list[int]] = defaultdict(list)
    for r in covered:
        by_layer[int(r["layer"])].append(int(r["route_count"]))
    for layer, vals in sorted(by_layer.items()):
        band_c = Counter(
            classify_evidence_band(c) for c in vals
        )
        per_layer[str(layer)] = {
            "quantiles": quantiles(vals),
            "band_counts": dict(band_c),
            "n_zero_route": sum(1 for c in vals if c == 0),
            "n_promotion_range": band_c.get(BAND_PROMOTION, 0),
            "n_between_anchors": band_c.get(BAND_BETWEEN, 0),
            "n_below_anchor": band_c.get(BAND_BELOW, 0),
            "route_sum": int(sum(vals)),
        }

    return {
        "anchors": {
            "promotion_grade_rank64_min_route_count": PROMOTION_PANEL_MIN_ROUTE,
            "low_traffic_diagnostic_rank128_route_count": LOW_TRAFFIC_DIAGNOSTIC_ROUTE,
            "note": (
                "Sealed five-shard basis pilot anchors only. Band names are "
                "arithmetic/evidence classes, not quality labels for rank 64/128."
            ),
        },
        "band_counts_covered": {
            BAND_PROMOTION: counts.get(BAND_PROMOTION, 0),
            BAND_BETWEEN: counts.get(BAND_BETWEEN, 0),
            BAND_BELOW: counts.get(BAND_BELOW, 0),
            BAND_ZERO: counts.get(BAND_ZERO, 0),
        },
        "band_counts_including_unobserved": {
            BAND_PROMOTION: counts.get(BAND_PROMOTION, 0),
            BAND_BETWEEN: counts.get(BAND_BETWEEN, 0),
            BAND_BELOW: counts.get(BAND_BELOW, 0),
            BAND_ZERO: counts.get(BAND_ZERO, 0),
            BAND_UNOBSERVED: unobserved,
        },
        "threshold_counts": thresholds,
        "global_quantiles": quantiles(route_vals),
        "histogram_binned": {"labels": bin_labels, "counts": bin_counts},
        "histogram_exact_n_unique_route_counts": len(hist_counter),
        "histogram_exact": {str(k): int(v) for k, v in sorted(hist_counter.items())},
        "per_layer": per_layer,
        "not_quality_labels": True,
        "no_smooth_quality_law_invented": True,
    }


# ---------------------------------------------------------------------------
# Byte scenarios (reuse v2 target-local ledger arithmetic)
# ---------------------------------------------------------------------------
def _ledger_summary(ledger: v2.BasisLedger, **extra: Any) -> dict[str, Any]:
    d = ledger.as_dict()
    bpw = ledger.complete_bpw()
    out = {
        **d,
        "scope": "target_local",
        "complete_bpw_exact": f"{bpw.numerator}/{bpw.denominator}",
        "complete_bpw_float": float(bpw),
        "within_target_bpw": bool(bpw <= v2.TARGET_BPW),
        "itemization_reconciles": ledger.reconciles(),
        "ranks_never_reduced_to_fit": True,
    }
    out.update(extra)
    return out


def build_partial_encoded_ledger(
    tensors: Sequence[v2.TensorClass],
    *,
    rank_by_expert: dict[tuple[int, int], int],
) -> tuple[v2.BasisLedger, dict[str, int]]:
    """Encode only experts present in rank_by_expert; other routed experts skipped.

    Non-routed program organs still billed at preregistered ranks so the partial
    total is comparable on the shared non-routed base. Unresolved routed experts
    contribute zero encoded bytes in this incomplete scenario.
    """
    ledger = v2.BasisLedger()
    n_encoded_experts = 0
    n_skipped_routed = 0
    seen_skip: set[tuple[int, int]] = set()
    seen_enc: set[tuple[int, int]] = set()
    for tc in tensors:
        if tc.program_group == "routed_experts":
            assert tc.layer is not None and tc.expert_id is not None
            key = (int(tc.layer), int(tc.expert_id))
            if key not in rank_by_expert:
                if key not in seen_skip:
                    seen_skip.add(key)
                    n_skipped_routed += 1
                continue
            r = int(rank_by_expert[key])
            v2._encode_tensor_into_ledger(
                tc,
                ledger,
                scope="target_local",
                rank_override=r,
                basis_authorizing=False,
            )
            if key not in seen_enc:
                seen_enc.add(key)
                n_encoded_experts += 1
        else:
            v2._encode_tensor_into_ledger(
                tc,
                ledger,
                scope="target_local",
                basis_authorizing=False,
            )
    stats = {
        "n_encoded_routed_experts": n_encoded_experts,
        "n_skipped_routed_experts": n_skipped_routed,
    }
    return ledger, stats


def build_native_mixture_ledger(
    tensors: Sequence[v2.TensorClass],
    *,
    rank_by_expert: dict[tuple[int, int], int],
    native_experts: set[tuple[int, int]],
) -> v2.BasisLedger:
    """Promotion/between at assigned ranks; unresolved gate/up/down at native BF16."""
    ledger = v2.BasisLedger()
    for tc in tensors:
        if tc.program_group == "routed_experts":
            assert tc.layer is not None and tc.expert_id is not None
            key = (int(tc.layer), int(tc.expert_id))
            if key in native_experts:
                ledger.add_native(tc.payload_bytes)
                continue
            if key not in rank_by_expert:
                raise CensusError(f"routed expert {key} missing rank and not native")
            v2._encode_tensor_into_ledger(
                tc,
                ledger,
                scope="target_local",
                rank_override=int(rank_by_expert[key]),
                basis_authorizing=False,
            )
        else:
            v2._encode_tensor_into_ledger(
                tc,
                ledger,
                scope="target_local",
                basis_authorizing=False,
            )
    return ledger


def build_byte_scenarios(
    tensors: Sequence[v2.TensorClass],
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    promotion: set[tuple[int, int]] = set()
    between: set[tuple[int, int]] = set()
    below: set[tuple[int, int]] = set()
    zero: set[tuple[int, int]] = set()
    unobserved: set[tuple[int, int]] = set()
    for r in records:
        key = (int(r["layer"]), int(r["expert_id"]))
        band = r["evidence_band"]
        if band == BAND_PROMOTION:
            promotion.add(key)
        elif band == BAND_BETWEEN:
            between.add(key)
        elif band == BAND_BELOW:
            below.add(key)
        elif band == BAND_ZERO:
            zero.add(key)
        elif band == BAND_UNOBSERVED:
            unobserved.add(key)
        else:
            raise CensusError(f"unknown band {band}")

    all_routed = set(v2.list_routed_experts(tensors))
    # Sanity: unobserved must be exactly layer 78's experts present in census.
    expected_unobs = {(UNOBSERVED_LAYER, e) for e in range(N_EXPERTS)}
    if unobserved != expected_unobs:
        raise CensusError(
            f"unobserved set mismatch: got {len(unobserved)} expected {len(expected_unobs)}"
        )
    if not expected_unobs <= all_routed:
        raise CensusError("layer 78 experts missing from weight census tensors")

    unresolved = below | zero | unobserved
    known_rank = promotion | between

    # 1) anchor_assignment_scenario (incomplete; never authorizing)
    rank_map_anchor = {k: 64 for k in promotion}
    rank_map_anchor.update({k: 128 for k in between})
    partial_led, partial_stats = build_partial_encoded_ledger(
        tensors, rank_by_expert=rank_map_anchor
    )
    anchor_scenario = {
        "name": "anchor_assignment_scenario",
        "authorizing": False,
        "incomplete": True,
        "description": (
            "PROMOTION_PANEL_ROUTE_RANGE experts at rank 64; BETWEEN_PILOT_ANCHORS "
            "at rank 128; BELOW_LOW_TRAFFIC_ANCHOR, ZERO_ROUTE, and UNOBSERVED left "
            "unresolved. Incomplete assignment — never authorizing."
        ),
        "assignment": {
            "rank_64": BAND_PROMOTION,
            "rank_128": BAND_BETWEEN,
            "unresolved": [BAND_BELOW, BAND_ZERO, BAND_UNOBSERVED],
        },
        "n_rank_64_experts": len(promotion),
        "n_rank_128_experts": len(between),
        "n_unresolved_experts": len(unresolved),
        "n_unresolved_below": len(below),
        "n_unresolved_zero": len(zero),
        "n_unresolved_unobserved": len(unobserved),
        "known_rank_encoded": _ledger_summary(
            partial_led,
            authorizing=False,
            incomplete=True,
            known_rank_only=True,
            **partial_stats,
        ),
        "note": (
            "known_rank_encoded bills non-routed organs plus only experts with an "
            "anchor-assigned rank. Unresolved experts contribute no encoded bytes "
            "here. Not a complete model byte total and not authorization."
        ),
    }

    # 2) rank128_for_all_nonpromotion_bound
    rank128_experts = all_routed - promotion
    mix_led = v2.build_routed_mixture_ledger(
        tensors, rank128_experts=rank128_experts
    )
    # Guard: ranks present must be only 64 and/or 128 for routed.
    routed_ranks = {
        int(b["rank"])
        for b in mix_led.bases.values()
        if "|E" in str(b["identity"])
    }
    if not routed_ranks <= {64, 128}:
        raise CensusError(f"unexpected routed ranks in mixture: {routed_ranks}")
    rank128_bound = {
        "name": "rank128_for_all_nonpromotion_bound",
        "authorizing": False,
        "is_uncertainty_bound": True,
        "is_quality_proof": False,
        "description": (
            "Promotion-range covered experts at rank 64; every other expert "
            "including layer 78 at rank 128. Byte uncertainty bound only — not "
            "quality proof. Ranks never reduced to fit target BPW."
        ),
        "assignment": {
            "rank_64": BAND_PROMOTION,
            "rank_128": "all_other_routed_experts_including_unobserved_layer_78",
        },
        "n_rank_64_experts": len(promotion),
        "n_rank_128_experts": len(rank128_experts),
        "ledger": _ledger_summary(
            mix_led,
            authorizing=False,
            is_uncertainty_bound=True,
            routed_ranks_present=sorted(routed_ranks),
        ),
    }

    # 3) native_for_unresolved_bound
    rank_map_native = {k: 64 for k in promotion}
    rank_map_native.update({k: 128 for k in between})
    native_led = build_native_mixture_ledger(
        tensors,
        rank_by_expert=rank_map_native,
        native_experts=unresolved,
    )
    native_bound = {
        "name": "native_for_unresolved_bound",
        "authorizing": False,
        "description": (
            "Promotion-range at rank 64; between-anchor at rank 128; below-anchor, "
            "zero-route, and unobserved expert gate/up/down triplets billed at "
            "sealed native BF16 payload width."
        ),
        "assignment": {
            "rank_64": BAND_PROMOTION,
            "rank_128": BAND_BETWEEN,
            "native_bf16": [BAND_BELOW, BAND_ZERO, BAND_UNOBSERVED],
        },
        "n_rank_64_experts": len(promotion),
        "n_rank_128_experts": len(between),
        "n_native_experts": len(unresolved),
        "ledger": _ledger_summary(
            native_led,
            authorizing=False,
            component_reconciliation=native_led.component_totals(),
            total_bytes=native_led.total_bytes(),
        ),
    }
    # Explicit component reconciliation expose
    comps = native_led.component_totals()
    native_bound["component_reconciliation"] = {
        **comps,
        "sum_components": sum(comps.values()),
        "total_bytes": native_led.total_bytes(),
        "reconciles": sum(comps.values()) == native_led.total_bytes(),
    }

    # Compare with sealed max 6583 rank-128 under 49/50 BPW
    sensitivity_ref = {
        "max_rank128_experts_under_target_bpw": 6583,
        "max_rank128_fraction_under_target_bpw_exact": "6583/19456",
        "target_bpw": f"{v2.TARGET_BPW.numerator}/{v2.TARGET_BPW.denominator}",
        "source": "GLM52_V2_PROGRAM_FEASIBILITY.json route_population_sensitivity",
        "note": (
            "Sealed arithmetic sensitivity threshold. Not a traffic classification. "
            "This census does not promote even if a scenario fits under that count."
        ),
    }
    n_r128_if_all_nonpromo = len(rank128_experts)
    comparison = {
        "sealed_max_rank128_experts_under_49_50_bpw": 6583,
        "rank128_for_all_nonpromotion_n_rank128": n_r128_if_all_nonpromo,
        "rank128_for_all_nonpromotion_exceeds_sealed_max": n_r128_if_all_nonpromo > 6583,
        "anchor_n_rank128": len(between),
        "anchor_n_rank64": len(promotion),
        "native_bound_within_target_bpw": native_bound["ledger"]["within_target_bpw"],
        "rank128_nonpromotion_within_target_bpw": rank128_bound["ledger"][
            "within_target_bpw"
        ],
        "even_if_bytes_fit_do_not_promote": True,
        "route_count_is_not_tensor_quality": True,
    }

    return {
        "anchor_assignment_scenario": anchor_scenario,
        "rank128_for_all_nonpromotion_bound": rank128_bound,
        "native_for_unresolved_bound": native_bound,
        "sealed_rank128_capacity_reference": sensitivity_ref,
        "scenario_comparison": comparison,
        "band_set_sizes": {
            BAND_PROMOTION: len(promotion),
            BAND_BETWEEN: len(between),
            BAND_BELOW: len(below),
            BAND_ZERO: len(zero),
            BAND_UNOBSERVED: len(unobserved),
        },
    }


# ---------------------------------------------------------------------------
# Representative experts for a later bounded pilot (selection only; no rehydrate)
# ---------------------------------------------------------------------------
def select_next_pilot_representatives(
    records: Sequence[dict[str, Any]],
    *,
    per_band_per_depth: int = 2,
) -> dict[str, Any]:
    """Select representatives across bands and early/middle/late layers.

    Selection only — does not rehydrate weights or claim quality.
    """
    covered = [r for r in records if r["observed"]]
    depths = {
        "early": range(3, 28),
        "middle": range(28, 53),
        "late": range(53, 78),
    }
    bands_for_pilot = (BAND_BELOW, BAND_BETWEEN, BAND_ZERO, BAND_PROMOTION)
    selected: list[dict[str, Any]] = []
    for band in bands_for_pilot:
        for depth_name, layer_range in depths.items():
            pool = [
                r
                for r in covered
                if r["evidence_band"] == band and int(r["layer"]) in layer_range
            ]
            # Deterministic: sort by (route_count, layer, expert_id)
            pool.sort(
                key=lambda r: (
                    int(r["route_count"]),
                    int(r["layer"]),
                    int(r["expert_id"]),
                )
            )
            # Spread: pick evenly spaced indices
            if not pool:
                continue
            if len(pool) <= per_band_per_depth:
                picks = pool
            else:
                idxs = [
                    int(round(i * (len(pool) - 1) / (per_band_per_depth - 1)))
                    for i in range(per_band_per_depth)
                ]
                picks = [pool[i] for i in idxs]
            for r in picks:
                selected.append(
                    {
                        "layer": r["layer"],
                        "expert_id": r["expert_id"],
                        "route_count": r["route_count"],
                        "evidence_band": r["evidence_band"],
                        "depth_bucket": depth_name,
                        "rehydrated": False,
                    }
                )
    # Require multi-band multi-depth coverage intent in the note
    return {
        "purpose": (
            "Representative experts for a later bounded real-weight pilot. "
            "Not rehydrated by this census."
        ),
        "requirements_for_next_pilot": [
            "Cover multiple BELOW_LOW_TRAFFIC_ANCHOR experts",
            "Cover multiple BETWEEN_PILOT_ANCHORS experts",
            "Span early/middle/late layers",
            "Test a representation designed for zero/rare routes",
            "No full traversal authorized by this census",
        ],
        "n_selected": len(selected),
        "selected": selected,
        "rehydrated": False,
    }


# ---------------------------------------------------------------------------
# Coverage proof assembly
# ---------------------------------------------------------------------------
def build_canonical_binding(canonical: dict[int, LoadedTopk]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer in sorted(canonical):
        it = canonical[layer]
        rows.append(
            {
                "layer": layer,
                "capsule_filename": it.ref.capsule_filename,
                "member_name": it.ref.member_name,
                "normalized_array_sha256": it.normalized_sha256,
                "member_bytes_sha256": it.member_sha256,
                "dtype": it.dtype,
                "original_shape": it.original_shape,
                "normalized_shape": list(FLAT_SHAPE),
                "sealed_capsule_sha256": it.sealed_capsule.get("capsule_sha256_sealed"),
                "sealed_array_sha256": it.sealed_array_sha256,
                "computed_array_bytes_sha256": it.computed_array_bytes_sha256,
                "sealed_array_hash_verified": it.sealed_array_hash_verified,
                "sealed_array_sidecar_key": it.sealed_array_sidecar_key,
                "routed_census_member": EXPECTED_COVERED_FIRST
                <= layer
                <= EXPECTED_COVERED_LAST,
            }
        )
    return rows


def summarize_sealed_array_hash_proof(
    loaded: Sequence[LoadedTopk],
) -> dict[str, Any]:
    """Aggregate proof that every loaded top-k matches its sealed array hash."""
    n = len(loaded)
    n_present = 0
    n_verified = 0
    rows: list[dict[str, Any]] = []
    for it in sorted(
        loaded, key=lambda x: (x.ref.layer, x.ref.capsule_filename, x.ref.member_name)
    ):
        present = bool(it.sealed_array_sha256) and bool(
            SEALED_ARRAY_HASH_RE.fullmatch(it.sealed_array_sha256)
        )
        verified = bool(it.sealed_array_hash_verified) and (
            it.computed_array_bytes_sha256 == it.sealed_array_sha256
        )
        if present:
            n_present += 1
        if verified:
            n_verified += 1
        rows.append(
            {
                "layer": it.ref.layer,
                "capsule_filename": it.ref.capsule_filename,
                "member_name": it.ref.member_name,
                "sealed_array_sidecar_key": it.sealed_array_sidecar_key,
                "sealed_array_sha256": it.sealed_array_sha256,
                "computed_array_bytes_sha256": it.computed_array_bytes_sha256,
                "sealed_array_hash_verified": verified,
            }
        )
    if n_present != n or n_verified != n or not all(
        r["sealed_array_hash_verified"] for r in rows
    ):
        raise CensusError(
            f"sealed array hash proof incomplete: loaded={n} present={n_present} "
            f"verified={n_verified}"
        )
    return {
        "n_loaded_topk_members": n,
        "n_sealed_array_hashes_present": n_present,
        "n_sealed_array_hashes_verified": n_verified,
        "all_loaded_topk_match_sealed_array_hashes": True,
        "whole_capsule_hash_recomputed": False,
        "hash_scope": "loaded_topk_array_c_contiguous_raw_bytes_only",
        "members": rows,
    }


def run_census(
    *,
    capsule_dir: Path = DEFAULT_CAPSULE_DIR,
    headers_path: Path = SOURCE_HEADERS,
    zip_open: Callable[[Path], zipfile.ZipFile] | None = None,
) -> dict[str, Any]:
    refs = enumerate_topk_members(capsule_dir)
    if not refs:
        raise CensusError("no topk_indices.npy members found")

    # Dense capsules may lack topk; report inventory.
    all_capsules = [p.name for p in list_capsule_npz_files(capsule_dir)]
    loaded: list[LoadedTopk] = []
    for ref in refs:
        loaded.append(load_validated_topk(ref, zip_open=zip_open))

    by_layer = group_by_layer(loaded)
    dup = check_duplicate_agreement(by_layer)
    canonical = select_canonical_members(by_layer)
    records, coverage = build_expert_records(canonical)
    bands = summarize_bands(records)
    sealed_array_proof = summarize_sealed_array_hash_proof(loaded)

    # Source weight census for byte scenarios
    entries = v2.load_source_headers(headers_path)
    weight_census = v2.build_census(entries)
    tensors: list[v2.TensorClass] = weight_census["tensors"]
    byte_scenarios = build_byte_scenarios(tensors, records)
    reps = select_next_pilot_representatives(records)

    sealed_capsules = []
    seen_caps: set[str] = set()
    for it in loaded:
        fn = it.ref.capsule_filename
        if fn in seen_caps:
            continue
        seen_caps.add(fn)
        sealed_capsules.append(it.sealed_capsule)

    inventory = {
        "capsule_dir": str(capsule_dir),
        "n_capsule_npz_files": len(all_capsules),
        "capsule_filenames": all_capsules,
        "n_topk_members": len(refs),
        "topk_members": [
            {
                "capsule_filename": r.capsule_filename,
                "member_name": r.member_name,
                "layer": r.layer,
            }
            for r in refs
        ],
        "layers_with_topk": sorted(by_layer),
        "member_selective_loading": True,
        "whole_archive_decompressed": False,
        "whole_capsule_files_hashed": False,
        "whole_capsule_hash_recomputed": False,
    }

    code_hashes = {
        "glm52_route_population_census_py_sha256": sha256_file(Path(__file__)),
        "glm52_activation_aware_pack_v2_py_sha256": sha256_file(
            HERE / "glm52_activation_aware_pack_v2.py"
        ),
    }
    test_path = HERE / "tests" / "test_glm52_route_population_census.py"
    if test_path.exists():
        code_hashes["test_glm52_route_population_census_py_sha256"] = sha256_file(
            test_path
        )

    source_hashes: dict[str, str] = {}
    if headers_path.exists():
        source_hashes["GLM52_SOURCE_SHARD_HEADERS_sha256"] = sha256_file(headers_path)
    if PILOT_RECEIPT.exists():
        source_hashes["GLM52_BASIS_PILOT_RECEIPT_sha256"] = sha256_file(PILOT_RECEIPT)
    if CONTROLLER_RESEAL.exists():
        source_hashes["GLM52_BASIS_PILOT_CONTROLLER_RESEAL_sha256"] = sha256_file(
            CONTROLLER_RESEAL
        )
    if V2_FEASIBILITY.exists():
        source_hashes["GLM52_V2_PROGRAM_FEASIBILITY_sha256"] = sha256_file(V2_FEASIBILITY)

    # Top-level authorization/evidence remain false regardless of byte fit.
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "format_version": FORMAT_VERSION,
        "purpose": (
            "Route-population census from retained teacher top-k metadata. "
            "Resolves rank-64 vs rank-128 population uncertainty only as far as "
            "route counts allow. Not a compression run and not a capability claim."
        ),
        "seed": SEED,
        "target_bpw": f"{v2.TARGET_BPW.numerator}/{v2.TARGET_BPW.denominator}",
        "source_hashes": source_hashes,
        "code_hashes": code_hashes,
        "capsule_inventory": inventory,
        "sealed_capsule_hashes_bound": sealed_capsules,
        "duplicate_agreement": dup,
        "canonical_members": build_canonical_binding(canonical),
        "sealed_array_hash_proof": sealed_array_proof,
        "n_loaded_topk_members": sealed_array_proof["n_loaded_topk_members"],
        "n_sealed_array_hashes_present": sealed_array_proof[
            "n_sealed_array_hashes_present"
        ],
        "n_sealed_array_hashes_verified": sealed_array_proof[
            "n_sealed_array_hashes_verified"
        ],
        "all_loaded_topk_match_sealed_array_hashes": sealed_array_proof[
            "all_loaded_topk_match_sealed_array_hashes"
        ],
        "whole_capsule_hash_recomputed": False,
        "coverage": coverage,
        "evidence_band_summary": bands,
        "expert_records": records,
        "n_expert_records": len(records),
        "byte_scenarios": byte_scenarios,
        "next_pilot_representatives": reps,
        "route_population_evidence_sufficient_for_rank_assignment": False,
        "within_target_bpw_for_proven_complete_assignment": False,
        "full_traversal_authorized": False,
        "full_parent_traversal_started": False,
        "capable_artifact_claimed": False,
        "safety": dict(SAFETY_FENCES),
        "remaining_uncertainties": [
            "Route count is not tensor quality; band names are arithmetic anchors only.",
            "Layer 78 (256 experts) is UNOBSERVED; never imputed.",
            "Below-anchor and zero-route experts have no bounded tensor-quality exemplar.",
            "No smooth quality-versus-route-count law may be invented from five pilot tensors.",
            "anchor_assignment_scenario is incomplete and never authorizing.",
            "rank128_for_all_nonpromotion_bound is a byte envelope, not quality proof.",
            "Even if a byte scenario fits under 49/50 BPW, do not promote.",
            "No full parent traversal, HIDE, Odyssey, or Ramanujan research is authorized.",
        ],
        "non_claims": [
            "Does not claim representation capability for any rank assignment.",
            "Does not rehydrate experts or read parent weight bodies.",
            "Does not authorize full traversal from route counts alone.",
            "Does not treat PROMOTION_PANEL_ROUTE_RANGE as proven rank-64 quality for every member.",
            "Does not treat BETWEEN_PILOT_ANCHORS as proven rank-128 quality for every member.",
        ],
        "next_safe_action": (
            "Run a bounded real-weight pilot on selected representatives spanning "
            "BELOW_LOW_TRAFFIC_ANCHOR and BETWEEN_PILOT_ANCHORS experts across "
            "early/middle/late layers, and test a representation designed for "
            "zero/rare routes. Do not rehydrate from this census alone. "
            "No full traversal is authorized."
        ),
    }
    # Deterministic content hash excluding the hash field itself.
    receipt["receipt_sha256"] = sha256_json(receipt)
    return receipt


def census_markdown(receipt: dict[str, Any]) -> str:
    cov = receipt["coverage"]
    bands = receipt["evidence_band_summary"]
    bc = bands["band_counts_including_unobserved"]
    scen = receipt["byte_scenarios"]
    cmp_ = scen["scenario_comparison"]
    gq = bands["global_quantiles"]
    lines = [
        "# GLM-5.2 route-population census",
        "",
        "Route-metadata census from retained teacher top-k indices only. "
        "Not a compression run. Not a capability claim.",
        "",
        "## Top-level fences (all remain false)",
        "",
        f"- `route_population_evidence_sufficient_for_rank_assignment`: "
        f"**{receipt['route_population_evidence_sufficient_for_rank_assignment']}**",
        f"- `within_target_bpw_for_proven_complete_assignment`: "
        f"**{receipt['within_target_bpw_for_proven_complete_assignment']}**",
        f"- `full_traversal_authorized`: **{receipt['full_traversal_authorized']}**",
        "",
        "## Coverage",
        "",
        f"- Covered routed layers: **{cov['covered_layers'][0]}–{cov['covered_layers'][-1]}** "
        f"({cov['n_covered_layers']} layers × 256 = {cov['n_covered_expert_records']} experts)",
        f"- Missing: **layer {cov['unobserved_layer']}** "
        f"({cov['n_unobserved_expert_records']} experts UNOBSERVED, never imputed)",
        f"- Per-layer route sum: **{cov['expected_per_layer_route_sum']}** "
        f"({'OK' if cov['all_covered_layers_route_sum_ok'] else 'FAIL'})",
        f"- Duplicate overlapping copies: "
        f"**{receipt['duplicate_agreement']['n_layers_with_duplicates']}** layers, "
        f"all byte-identical = "
        f"**{receipt['duplicate_agreement']['all_duplicates_byte_identical']}**",
        f"- Loaded top-k members: **{receipt['n_loaded_topk_members']}**; "
        f"sealed array hashes present/verified: "
        f"**{receipt['n_sealed_array_hashes_present']}/"
        f"{receipt['n_sealed_array_hashes_verified']}**; "
        f"all match = **{receipt['all_loaded_topk_match_sealed_array_hashes']}**",
        f"- Whole capsule hash recomputed: "
        f"**{receipt['whole_capsule_hash_recomputed']}** "
        f"(bind sealed capsule hashes only)",
        "",
        "## Evidence bands (arithmetic anchors, not quality labels)",
        "",
        f"- Anchors: promotion-grade min route **{PROMOTION_PANEL_MIN_ROUTE}**; "
        f"low-traffic diagnostic **{LOW_TRAFFIC_DIAGNOSTIC_ROUTE}**",
        f"- `{BAND_PROMOTION}`: **{bc[BAND_PROMOTION]}**",
        f"- `{BAND_BETWEEN}`: **{bc[BAND_BETWEEN]}**",
        f"- `{BAND_BELOW}`: **{bc[BAND_BELOW]}**",
        f"- `{BAND_ZERO}`: **{bc[BAND_ZERO]}**",
        f"- `{BAND_UNOBSERVED}`: **{bc[BAND_UNOBSERVED]}**",
        "",
        "## Global route-count quantiles (covered only)",
        "",
        f"- min/p50/max: **{gq.get('min')} / {gq.get('p50')} / {gq.get('max')}**",
        f"- mean±std: **{gq.get('mean'):.2f} ± {gq.get('std'):.2f}**" if gq else "- n/a",
        "",
        "## Byte scenarios",
        "",
        "### anchor_assignment_scenario (incomplete, never authorizing)",
        "",
        f"- rank-64 experts: **{scen['anchor_assignment_scenario']['n_rank_64_experts']}**",
        f"- rank-128 experts: **{scen['anchor_assignment_scenario']['n_rank_128_experts']}**",
        f"- unresolved experts: **{scen['anchor_assignment_scenario']['n_unresolved_experts']}**",
        f"- known-rank total bytes: "
        f"**{scen['anchor_assignment_scenario']['known_rank_encoded']['total_bytes']:,}**",
        "",
        "### rank128_for_all_nonpromotion_bound (byte envelope, not quality)",
        "",
        f"- rank-64 / rank-128: "
        f"**{scen['rank128_for_all_nonpromotion_bound']['n_rank_64_experts']} / "
        f"{scen['rank128_for_all_nonpromotion_bound']['n_rank_128_experts']}**",
        f"- total bytes: "
        f"**{scen['rank128_for_all_nonpromotion_bound']['ledger']['total_bytes']:,}**",
        f"- complete BPW: "
        f"**{scen['rank128_for_all_nonpromotion_bound']['ledger']['complete_bpw_exact']}** "
        f"({scen['rank128_for_all_nonpromotion_bound']['ledger']['complete_bpw_float']:.6f})",
        f"- within 49/50: "
        f"**{scen['rank128_for_all_nonpromotion_bound']['ledger']['within_target_bpw']}**",
        "",
        "### native_for_unresolved_bound",
        "",
        f"- rank-64 / rank-128 / native: "
        f"**{scen['native_for_unresolved_bound']['n_rank_64_experts']} / "
        f"{scen['native_for_unresolved_bound']['n_rank_128_experts']} / "
        f"{scen['native_for_unresolved_bound']['n_native_experts']}**",
        f"- total bytes: "
        f"**{scen['native_for_unresolved_bound']['ledger']['total_bytes']:,}**",
        f"- complete BPW: "
        f"**{scen['native_for_unresolved_bound']['ledger']['complete_bpw_exact']}** "
        f"({scen['native_for_unresolved_bound']['ledger']['complete_bpw_float']:.6f})",
        f"- within 49/50: "
        f"**{scen['native_for_unresolved_bound']['ledger']['within_target_bpw']}**",
        f"- components reconcile: "
        f"**{scen['native_for_unresolved_bound']['component_reconciliation']['reconciles']}**",
        "",
        "## Comparison with sealed max rank-128 under 49/50",
        "",
        f"- Sealed max rank-128 experts: **{cmp_['sealed_max_rank128_experts_under_49_50_bpw']}**",
        f"- Non-promotion bound n_rank128: **{cmp_['rank128_for_all_nonpromotion_n_rank128']}** "
        f"(exceeds sealed max: **{cmp_['rank128_for_all_nonpromotion_exceeds_sealed_max']}**)",
        "",
        "## Next safe action",
        "",
        receipt["next_safe_action"],
        "",
        "## Safety",
        "",
    ]
    for k, v in receipt["safety"].items():
        lines.append(f"- `{k}`: **{v}**")
    lines += [
        "",
        f"Receipt sha256: `{receipt['receipt_sha256']}`",
    ]
    return "\n".join(lines) + "\n"


def write_census(
    *,
    capsule_dir: Path = DEFAULT_CAPSULE_DIR,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
) -> dict[str, Any]:
    receipt = run_census(capsule_dir=capsule_dir)
    out_json.write_text(
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    out_md.write_text(census_markdown(receipt), encoding="utf-8")
    return receipt


# ---------------------------------------------------------------------------
# Selftest (no real multi-GB dependency beyond optional smoke)
# ---------------------------------------------------------------------------
def _make_fake_topk(
    *,
    shape: tuple[int, ...] = CAPTURE_SHAPE,
    seed: int = 0,
    force_dup_row: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_rows = int(np.prod(shape[:-1]))
    k = shape[-1]
    rows = []
    for i in range(n_rows):
        # sample k unique experts
        row = rng.choice(N_EXPERTS, size=k, replace=False).astype(np.int32)
        rows.append(row)
    arr = np.stack(rows, axis=0).reshape(shape)
    if force_dup_row:
        flat = arr.reshape(n_rows, k)
        flat[0, 1] = flat[0, 0]
        arr = flat.reshape(shape)
    return arr


def _write_fake_capsule(
    path: Path,
    members: dict[str, np.ndarray],
    *,
    extra_members: dict[str, bytes] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in members.items():
            buf = io.BytesIO()
            np.save(buf, arr)
            zf.writestr(name, buf.getvalue())
        if extra_members:
            for name, blob in extra_members.items():
                zf.writestr(name, blob)


def selftest() -> int:
    """Lightweight internal checks (unit tests are authoritative)."""
    # normalize shapes
    a = _make_fake_topk(shape=CAPTURE_SHAPE, seed=1)
    b = normalize_topk(a)
    assert b.shape == FLAT_SHAPE
    c = normalize_topk(b)
    assert c.shape == FLAT_SHAPE
    assert sha256_array(b) == sha256_array(c)

    # raw-byte hash is reshape-stable and distinct from dtype+shape hash
    raw_h = sha256_array_raw_bytes(a)
    assert raw_h == sha256_array_raw_bytes(a.reshape(FLAT_SHAPE))
    assert raw_h != sha256_array(a)

    # exact member path only
    try:
        load_topk_member_bytes(Path("x.npz"), "prefix/layer_03/topk_indices.npy")
        raise AssertionError("expected non-exact path refusal")
    except CensusError:
        pass

    # sealed key resolve + match
    sealed_map = {"layer_03/topk_indices": raw_h}
    k, s = resolve_sealed_topk_array_hash(sealed_map, "layer_03/topk_indices.npy")
    assert k == "layer_03/topk_indices" and s == raw_h
    sealed_map_npy = {"layer_03/topk_indices.npy": raw_h}
    k2, s2 = resolve_sealed_topk_array_hash(
        sealed_map_npy, "layer_03/topk_indices.npy"
    )
    assert k2 == "layer_03/topk_indices.npy" and s2 == raw_h
    try:
        resolve_sealed_topk_array_hash({}, "layer_03/topk_indices.npy")
        raise AssertionError("expected missing sealed hash")
    except CensusError:
        pass
    try:
        resolve_sealed_topk_array_hash(
            {"layer_03/topk_indices": "deadbeef"}, "layer_03/topk_indices.npy"
        )
        raise AssertionError("expected malformed sealed hash")
    except CensusError:
        pass
    try:
        verify_loaded_topk_against_sealed(
            a,
            member_name="layer_03/topk_indices.npy",
            sealed_capsule={
                "array_sha256_topk_indices_sealed": {
                    "layer_03/topk_indices": "0" * 64
                }
            },
        )
        raise AssertionError("expected mismatch")
    except CensusError:
        pass

    # bad shape
    try:
        normalize_topk(np.zeros((10, 8), dtype=np.int32))
        raise AssertionError("expected shape failure")
    except CensusError:
        pass

    # band boundaries
    assert classify_evidence_band(0) == BAND_ZERO
    assert classify_evidence_band(1) == BAND_BELOW
    assert classify_evidence_band(204) == BAND_BELOW
    assert classify_evidence_band(205) == BAND_BETWEEN
    assert classify_evidence_band(2576) == BAND_BETWEEN
    assert classify_evidence_band(2577) == BAND_PROMOTION
    assert classify_evidence_band(None, unobserved=True) == BAND_UNOBSERVED

    # fences
    assert all(v is False for v in SAFETY_FENCES.values())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        nargs="?",
        default="census",
        choices=("selftest", "census"),
    )
    p.add_argument("--capsule-dir", type=Path, default=DEFAULT_CAPSULE_DIR)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = p.parse_args(argv)
    if args.command == "selftest":
        rc = selftest()
        print("selftest_ok" if rc == 0 else "selftest_fail")
        return rc
    receipt = write_census(
        capsule_dir=args.capsule_dir,
        out_json=args.out_json,
        out_md=args.out_md,
    )
    bands = receipt["evidence_band_summary"]["band_counts_including_unobserved"]
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
                "receipt_sha256": receipt["receipt_sha256"],
                "n_expert_records": receipt["n_expert_records"],
                "band_counts": bands,
                "route_population_evidence_sufficient_for_rank_assignment": receipt[
                    "route_population_evidence_sufficient_for_rank_assignment"
                ],
                "within_target_bpw_for_proven_complete_assignment": receipt[
                    "within_target_bpw_for_proven_complete_assignment"
                ],
                "full_traversal_authorized": receipt["full_traversal_authorized"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
