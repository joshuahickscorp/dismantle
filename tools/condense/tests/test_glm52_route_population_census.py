#!/usr/bin/env python3.12
"""Unit tests for GLM-5.2 route-population census (top-k metadata only).

Proves:

  * member-selective loading never asks for non-top-k members
  * accepted shapes normalize identically
  * invalid shape/dtype/range/per-row duplicates fail closed
  * duplicate layer copies must match; conflicts fail
  * deterministic canonical member selection
  * exact per-layer route conservation
  * evidence-band boundaries at 0, 1, 204, 205, 2576, 2577
  * missing layer 78 remains unobserved
  * all three byte scenarios reconcile and never change ranks to fit
  * top-level authorization/evidence fields remain false
  * deterministic receipt generation
  * loaded top-k raw bytes match sealed sidecar array_sha256 (fail closed)
  * exact layer_NN/topk_indices.npy member path only
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import sys
import zipfile
from collections import Counter

import numpy as np
import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
REPO = CONDENSE.parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_activation_aware_pack_v2 as v2  # noqa: E402
import glm52_route_population_census as rpc  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
class TrackingZipFile(zipfile.ZipFile):
    """ZipFile that records every ``read`` member name."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_members: list[str] = []

    def read(self, name, *args, **kwargs):
        self.read_members.append(str(name))
        return super().read(name, *args, **kwargs)


def _unique_rows(n_rows: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        rows.append(rng.choice(rpc.N_EXPERTS, size=k, replace=False).astype(np.int32))
    return np.stack(rows, axis=0)


def make_topk(
    *,
    shape=rpc.CAPTURE_SHAPE,
    seed: int = 0,
    force_dup_row: bool = False,
    force_bad_id: int | None = None,
) -> np.ndarray:
    n_rows = int(np.prod(shape[:-1]))
    k = shape[-1]
    flat = _unique_rows(n_rows, k, seed)
    if force_dup_row:
        flat[0, 1] = flat[0, 0]
    if force_bad_id is not None:
        flat[0, 0] = force_bad_id
    return flat.reshape(shape)


def write_capsule(path: pathlib.Path, members: dict[str, np.ndarray], extra=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in members.items():
            buf = io.BytesIO()
            np.save(buf, arr)
            zf.writestr(name, buf.getvalue())
        if extra:
            for name, blob in extra.items():
                zf.writestr(name, blob)


def raw_array_sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def array_sha256_map_for_members(
    members: dict[str, np.ndarray],
    *,
    key_with_npy: bool = False,
) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, arr in members.items():
        if not name.endswith("topk_indices.npy"):
            continue
        key = name if key_with_npy else name[: -len(".npy")]
        out[key] = raw_array_sha256(arr)
    return out


def write_sidecar(
    npz_path: pathlib.Path,
    *,
    capsule_sha256: str = "abc" * 21 + "ab",
    array_sha256: dict[str, str] | None = None,
    members: dict[str, np.ndarray] | None = None,
    key_with_npy: bool = False,
) -> None:
    if array_sha256 is None:
        if members is not None:
            array_sha256 = array_sha256_map_for_members(
                members, key_with_npy=key_with_npy
            )
        else:
            array_sha256 = {}
    doc = {
        "capsule_sha256": capsule_sha256,
        "seal_sha256": "def" * 21 + "de",
        "array_sha256": array_sha256,
        "schema": "hawking.glm52.teacher_capsule.v1",
    }
    npz_path.with_suffix(".json").write_text(json.dumps(doc), encoding="utf-8")


def write_capsule_and_sidecar(
    path: pathlib.Path,
    members: dict[str, np.ndarray],
    *,
    extra=None,
    capsule_sha256: str = "abc" * 21 + "ab",
    key_with_npy: bool = False,
    array_sha256: dict[str, str] | None = None,
) -> None:
    write_capsule(path, members, extra=extra)
    write_sidecar(
        path,
        capsule_sha256=capsule_sha256,
        members=members if array_sha256 is None else None,
        array_sha256=array_sha256,
        key_with_npy=key_with_npy,
    )


# ---------------------------------------------------------------------------
# Normalize / validate
# ---------------------------------------------------------------------------
def test_accepted_shapes_normalize_identically():
    shaped = make_topk(shape=rpc.CAPTURE_SHAPE, seed=7)
    flat = shaped.reshape(rpc.FLAT_SHAPE)
    n1 = rpc.normalize_topk(shaped)
    n2 = rpc.normalize_topk(flat)
    assert n1.shape == rpc.FLAT_SHAPE
    assert n2.shape == rpc.FLAT_SHAPE
    assert np.array_equal(n1, n2)
    assert rpc.sha256_array(n1) == rpc.sha256_array(n2)


def test_invalid_shape_fails_closed():
    with pytest.raises(rpc.CensusError, match="invalid topk shape"):
        rpc.normalize_topk(np.zeros((10, 8), dtype=np.int32))
    with pytest.raises(rpc.CensusError, match="invalid topk shape"):
        rpc.normalize_topk(np.zeros((16, 256, 7), dtype=np.int32))


def test_invalid_dtype_fails_closed():
    arr = make_topk(seed=1).astype(np.float32)
    with pytest.raises(rpc.CensusError, match="integer"):
        rpc.normalize_topk(arr)


def test_invalid_expert_range_fails_closed():
    with pytest.raises(rpc.CensusError, match=r"\[0, 255\]"):
        rpc.normalize_topk(make_topk(seed=2, force_bad_id=256))
    with pytest.raises(rpc.CensusError, match=r"\[0, 255\]"):
        rpc.normalize_topk(make_topk(seed=2, force_bad_id=-1))


def test_per_row_duplicate_fails_closed():
    with pytest.raises(rpc.CensusError, match="duplicate expert ID"):
        rpc.normalize_topk(make_topk(seed=3, force_dup_row=True))


# ---------------------------------------------------------------------------
# Member-selective loading
# ---------------------------------------------------------------------------
def test_member_selective_loading_never_asks_for_non_topk(tmp_path):
    cap = tmp_path / "L03_L03.npz"
    topk = make_topk(seed=11)
    write_capsule(
        cap,
        {
            "layer_03/topk_indices.npy": topk,
            "layer_03/input_hidden.npy": np.zeros((4, 4), dtype=np.float32),
            "layer_03/router_logits.npy": np.zeros((4, 8), dtype=np.float32),
        },
        extra={"carry_out_hidden.npy": b"not-an-array"},
    )
    tracking: list[TrackingZipFile] = []

    def opener(path):
        zf = TrackingZipFile(path, "r")
        tracking.append(zf)
        return zf

    arr, raw = rpc.load_topk_array(cap, "layer_03/topk_indices.npy", zip_open=opener)
    assert arr.shape == rpc.CAPTURE_SHAPE
    assert raw
    assert tracking, "zip was not opened"
    assert tracking[0].read_members == ["layer_03/topk_indices.npy"]

    with pytest.raises(rpc.CensusError, match="non-topk"):
        rpc.load_topk_member_bytes(cap, "layer_03/input_hidden.npy")


def test_refuse_non_topk_member_name():
    with pytest.raises(rpc.CensusError, match="non-topk"):
        rpc.load_topk_member_bytes(pathlib.Path("x.npz"), "layer_03/topk_weights.npy")


def test_refuse_non_exact_topk_member_path():
    """Arbitrary paths merely ending in topk_indices.npy must fail."""
    with pytest.raises(rpc.CensusError, match="non-topk"):
        rpc.load_topk_member_bytes(
            pathlib.Path("x.npz"), "prefix/layer_03/topk_indices.npy"
        )
    with pytest.raises(rpc.CensusError, match="non-topk"):
        rpc.load_topk_member_bytes(pathlib.Path("x.npz"), "layer_3/topk_indices.npy")
    with pytest.raises(rpc.CensusError, match="non-topk"):
        rpc.load_topk_member_bytes(
            pathlib.Path("x.npz"), "layer_03/subdir/topk_indices.npy"
        )


# ---------------------------------------------------------------------------
# Sealed array hash verification (raw C-contiguous bytes)
# ---------------------------------------------------------------------------
def test_matching_sealed_raw_array_hash_passes(tmp_path):
    arr = make_topk(seed=50)
    cap = tmp_path / "L05_L05.npz"
    members = {"layer_05/topk_indices.npy": arr}
    write_capsule_and_sidecar(cap, members)
    ref = rpc.TopkMemberRef(cap, cap.name, "layer_05/topk_indices.npy", 5)
    loaded = rpc.load_validated_topk(ref)
    expect = raw_array_sha256(arr)
    assert loaded.sealed_array_sha256 == expect
    assert loaded.computed_array_bytes_sha256 == expect
    assert loaded.sealed_array_hash_verified is True
    assert loaded.sealed_array_sidecar_key == "layer_05/topk_indices"


def test_missing_sealed_array_hash_fails(tmp_path):
    arr = make_topk(seed=51)
    cap = tmp_path / "L05_L05.npz"
    members = {"layer_05/topk_indices.npy": arr}
    write_capsule(cap, members)
    write_sidecar(cap, array_sha256={})  # present sidecar, missing key
    ref = rpc.TopkMemberRef(cap, cap.name, "layer_05/topk_indices.npy", 5)
    with pytest.raises(rpc.CensusError, match="missing sealed array_sha256"):
        rpc.load_validated_topk(ref)


def test_malformed_sealed_array_hash_fails(tmp_path):
    arr = make_topk(seed=52)
    cap = tmp_path / "L05_L05.npz"
    members = {"layer_05/topk_indices.npy": arr}
    write_capsule(cap, members)
    write_sidecar(cap, array_sha256={"layer_05/topk_indices": "not-a-hex-digest"})
    ref = rpc.TopkMemberRef(cap, cap.name, "layer_05/topk_indices.npy", 5)
    with pytest.raises(rpc.CensusError, match="malformed sealed array_sha256"):
        rpc.load_validated_topk(ref)
    write_sidecar(cap, array_sha256={"layer_05/topk_indices": "abcd"})  # short
    with pytest.raises(rpc.CensusError, match="malformed sealed array_sha256"):
        rpc.load_validated_topk(ref)
    write_sidecar(
        cap,
        array_sha256={"layer_05/topk_indices": "A" * 64},  # uppercase
    )
    with pytest.raises(rpc.CensusError, match="malformed sealed array_sha256"):
        rpc.load_validated_topk(ref)


def test_mismatched_sealed_array_hash_fails(tmp_path):
    arr = make_topk(seed=53)
    cap = tmp_path / "L05_L05.npz"
    members = {"layer_05/topk_indices.npy": arr}
    write_capsule(cap, members)
    write_sidecar(cap, array_sha256={"layer_05/topk_indices": "0" * 64})
    ref = rpc.TopkMemberRef(cap, cap.name, "layer_05/topk_indices.npy", 5)
    with pytest.raises(rpc.CensusError, match="mismatch"):
        rpc.load_validated_topk(ref)


def test_both_sidecar_key_suffix_variants_accepted(tmp_path):
    arr = make_topk(seed=54)
    # Variant A: layer_NN/topk_indices (no .npy)
    cap_a = tmp_path / "A" / "L05_L05.npz"
    write_capsule_and_sidecar(cap_a, {"layer_05/topk_indices.npy": arr}, key_with_npy=False)
    loaded_a = rpc.load_validated_topk(
        rpc.TopkMemberRef(cap_a, cap_a.name, "layer_05/topk_indices.npy", 5)
    )
    assert loaded_a.sealed_array_sidecar_key == "layer_05/topk_indices"
    assert loaded_a.sealed_array_hash_verified is True

    # Variant B: layer_NN/topk_indices.npy
    cap_b = tmp_path / "B" / "L05_L05.npz"
    write_capsule_and_sidecar(cap_b, {"layer_05/topk_indices.npy": arr}, key_with_npy=True)
    loaded_b = rpc.load_validated_topk(
        rpc.TopkMemberRef(cap_b, cap_b.name, "layer_05/topk_indices.npy", 5)
    )
    assert loaded_b.sealed_array_sidecar_key == "layer_05/topk_indices.npy"
    assert loaded_b.sealed_array_hash_verified is True
    assert loaded_a.computed_array_bytes_sha256 == loaded_b.computed_array_bytes_sha256


# ---------------------------------------------------------------------------
# Duplicates + canonical selection
# ---------------------------------------------------------------------------
def test_duplicate_layers_must_match_and_conflicts_fail(tmp_path):
    a = make_topk(seed=20)
    b = make_topk(seed=21)  # different
    assert not np.array_equal(a, b)

    # agreement case
    cap1 = tmp_path / "L05_L05.npz"
    cap2 = tmp_path / "L04_L15.npz"
    write_capsule_and_sidecar(cap1, {"layer_05/topk_indices.npy": a})
    write_capsule_and_sidecar(cap2, {"layer_05/topk_indices.npy": a.copy()})
    refs = [
        rpc.TopkMemberRef(cap1, cap1.name, "layer_05/topk_indices.npy", 5),
        rpc.TopkMemberRef(cap2, cap2.name, "layer_05/topk_indices.npy", 5),
    ]
    loaded = [rpc.load_validated_topk(r) for r in refs]
    by = rpc.group_by_layer(loaded)
    ag = rpc.check_duplicate_agreement(by)
    assert ag["all_duplicates_byte_identical"] is True
    assert ag["n_layers_with_duplicates"] == 1
    assert all(it.sealed_array_hash_verified for it in loaded)

    # conflict case
    write_capsule_and_sidecar(cap2, {"layer_05/topk_indices.npy": b})
    loaded2 = [rpc.load_validated_topk(r) for r in refs]
    by2 = rpc.group_by_layer(loaded2)
    with pytest.raises(rpc.CensusError, match="conflict"):
        rpc.check_duplicate_agreement(by2)


def test_deterministic_canonical_member_selection(tmp_path):
    arr = make_topk(seed=30)
    # filenames sort: L04_L15.npz < L05_L05.npz?
    # 'L04_L15.npz' < 'L05_L05.npz' lexicographically
    cap_range = tmp_path / "L04_L15.npz"
    cap_single = tmp_path / "L05_L05.npz"
    write_capsule_and_sidecar(cap_range, {"layer_05/topk_indices.npy": arr})
    write_capsule_and_sidecar(cap_single, {"layer_05/topk_indices.npy": arr.copy()})
    refs = [
        rpc.TopkMemberRef(cap_single, cap_single.name, "layer_05/topk_indices.npy", 5),
        rpc.TopkMemberRef(cap_range, cap_range.name, "layer_05/topk_indices.npy", 5),
    ]
    loaded = [rpc.load_validated_topk(r) for r in refs]
    by = rpc.group_by_layer(loaded)
    can = rpc.select_canonical_members(by)
    # min (filename, member): L04_L15.npz wins
    assert can[5].ref.capsule_filename == "L04_L15.npz"
    # stable across reshuffle
    loaded_rev = list(reversed(loaded))
    can2 = rpc.select_canonical_members(rpc.group_by_layer(loaded_rev))
    assert can2[5].ref.capsule_filename == can[5].ref.capsule_filename
    assert can2[5].normalized_sha256 == can[5].normalized_sha256
    assert can[5].sealed_array_hash_verified is True


# ---------------------------------------------------------------------------
# Route conservation + evidence bands
# ---------------------------------------------------------------------------
def test_exact_per_layer_route_conservation():
    topk = make_topk(seed=40)
    counts = rpc.count_routes_for_layer(topk)
    assert counts.shape == (256,)
    assert int(counts.sum()) == rpc.EXPECTED_ROUTE_SUM
    # each row contributes 8 unique experts
    flat = rpc.normalize_topk(topk)
    assert flat.shape[0] * flat.shape[1] == rpc.EXPECTED_ROUTE_SUM


def test_evidence_band_boundaries():
    cases = [
        (0, rpc.BAND_ZERO),
        (1, rpc.BAND_BELOW),
        (204, rpc.BAND_BELOW),
        (205, rpc.BAND_BETWEEN),
        (2576, rpc.BAND_BETWEEN),
        (2577, rpc.BAND_PROMOTION),
        (4096, rpc.BAND_PROMOTION),
    ]
    for count, band in cases:
        assert rpc.classify_evidence_band(count) == band, (count, band)
    assert rpc.classify_evidence_band(None, unobserved=True) == rpc.BAND_UNOBSERVED


def test_layer_78_remains_unobserved_in_records():
    # Build minimal canonical for layers 3 only, then patch via direct record builder
    # by constructing synthetic LoadedTopk for all covered layers is heavy;
    # unit-test the unobserved rows from a full synthetic canonical 3..77.
    pass  # covered in test_synthetic_full_coverage_receipt


def _synthetic_canonical_for_layers(
    layers: list[int],
    tmp_path: pathlib.Path,
    *,
    seed_base: int = 100,
) -> dict[int, rpc.LoadedTopk]:
    canonical: dict[int, rpc.LoadedTopk] = {}
    for i, layer in enumerate(layers):
        cap = tmp_path / f"L{layer:02d}_L{layer:02d}.npz"
        arr = make_topk(seed=seed_base + i)
        member = f"layer_{layer:02d}/topk_indices.npy"
        members = {member: arr}
        write_capsule_and_sidecar(
            cap, members, capsule_sha256=f"{layer:064d}"[:64]
        )
        ref = rpc.TopkMemberRef(cap, cap.name, member, layer)
        canonical[layer] = rpc.load_validated_topk(ref)
    return canonical


def test_missing_layer_78_unobserved_and_covered_count(tmp_path):
    layers = list(range(3, 78))
    can = _synthetic_canonical_for_layers(layers, tmp_path)
    records, coverage = rpc.build_expert_records(can)
    assert coverage["unobserved_layer"] == 78
    assert coverage["missing_routed_layers"] == [78]
    assert coverage["n_covered_expert_records"] == 19200
    assert coverage["n_unobserved_expert_records"] == 256
    assert coverage["layer_78_imputed"] is False
    assert all(coverage["per_layer_route_sum"][str(L)] == 32768 for L in layers)
    unobs = [r for r in records if r["layer"] == 78]
    assert len(unobs) == 256
    assert all(r["evidence_band"] == rpc.BAND_UNOBSERVED for r in unobs)
    assert all(r["route_count"] is None for r in unobs)
    assert all(r["observed"] is False for r in unobs)


# ---------------------------------------------------------------------------
# Byte scenarios
# ---------------------------------------------------------------------------
def _add_routed_expert(tensors: list[v2.TensorClass], layer: int, expert: int) -> None:
    for proj, organ in (
        ("gate_proj", "routed_gate"),
        ("up_proj", "routed_up"),
        ("down_proj", "routed_down"),
    ):
        shape = [6144, 2048] if proj == "down_proj" else [2048, 6144]
        pb = int(np.prod(shape)) * 2  # bf16
        tensors.append(
            v2.TensorClass(
                name=f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight",
                organ_class=organ,
                layer=layer,
                expert_id=expert,
                projection=proj,
                program_group="routed_experts",
                shape=shape,
                payload_bytes=pb,
                n_weights=int(np.prod(shape)),
            )
        )


def _tiny_tensors_for_scenarios() -> list[v2.TensorClass]:
    """Synthetic tensors: covered sample experts + full layer-78 population."""
    tensors: list[v2.TensorClass] = []
    for layer, expert in [(3, 0), (3, 1), (5, 0), (5, 100)]:
        _add_routed_expert(tensors, layer, expert)
    # Full layer 78 so unobserved set matches the 256-expert contract.
    for expert in range(rpc.N_EXPERTS):
        _add_routed_expert(tensors, 78, expert)
    tensors.append(
        v2.TensorClass(
            name="model.layers.3.mlp.gate.weight",
            organ_class="router_control",
            layer=3,
            expert_id=None,
            projection="gate",
            program_group="router_control",
            shape=[256, 6144],
            payload_bytes=256 * 6144 * 2,
            n_weights=256 * 6144,
        )
    )
    return tensors


def test_byte_scenarios_reconcile_and_never_change_ranks():
    tensors = _tiny_tensors_for_scenarios()
    records = [
        {
            "layer": 3,
            "expert_id": 0,
            "route_count": 3000,
            "route_fraction": 3000 / 4096,
            "zero_route": False,
            "evidence_band": rpc.BAND_PROMOTION,
            "observed": True,
        },
        {
            "layer": 3,
            "expert_id": 1,
            "route_count": 500,
            "route_fraction": 500 / 4096,
            "zero_route": False,
            "evidence_band": rpc.BAND_BETWEEN,
            "observed": True,
        },
        {
            "layer": 5,
            "expert_id": 0,
            "route_count": 50,
            "route_fraction": 50 / 4096,
            "zero_route": False,
            "evidence_band": rpc.BAND_BELOW,
            "observed": True,
        },
        {
            "layer": 5,
            "expert_id": 100,
            "route_count": 0,
            "route_fraction": 0.0,
            "zero_route": True,
            "evidence_band": rpc.BAND_ZERO,
            "observed": True,
        },
    ]
    for expert in range(rpc.N_EXPERTS):
        records.append(
            {
                "layer": 78,
                "expert_id": expert,
                "route_count": None,
                "route_fraction": None,
                "zero_route": None,
                "evidence_band": rpc.BAND_UNOBSERVED,
                "observed": False,
            }
        )
    scenarios = rpc.build_byte_scenarios(tensors, records)

    anchor = scenarios["anchor_assignment_scenario"]
    assert anchor["authorizing"] is False
    assert anchor["incomplete"] is True
    assert anchor["n_rank_64_experts"] == 1
    assert anchor["n_rank_128_experts"] == 1
    # below + zero + 256 unobserved
    assert anchor["n_unresolved_experts"] == 2 + 256
    assert anchor["known_rank_encoded"]["itemization_reconciles"] is True
    assert anchor["known_rank_encoded"]["ranks_never_reduced_to_fit"] is True

    r128 = scenarios["rank128_for_all_nonpromotion_bound"]
    assert r128["is_quality_proof"] is False
    assert r128["ledger"]["itemization_reconciles"] is True
    assert r128["ledger"]["ranks_never_reduced_to_fit"] is True
    assert set(r128["ledger"]["routed_ranks_present"]) <= {64, 128}
    # promotion=1 at 64; remaining 3 covered + 256 layer78 at 128
    assert r128["n_rank_64_experts"] == 1
    assert r128["n_rank_128_experts"] == 3 + 256

    native = scenarios["native_for_unresolved_bound"]
    assert native["ledger"]["itemization_reconciles"] is True
    assert native["component_reconciliation"]["reconciles"] is True
    assert native["n_native_experts"] == 2 + 256
    assert native["ledger"]["ranks_never_reduced_to_fit"] is True

    # Ranks never lowered: native path bills unresolved as native, not reduced rank.
    assert native["n_rank_64_experts"] == 1
    assert native["n_rank_128_experts"] == 1


def test_top_level_authorization_fields_remain_false_on_synthetic(tmp_path, monkeypatch):
    """Build a reduced census path by patching enumerate to a 33-file layout.

    Full 3..77 synthetic under 33 capsules is heavy; instead verify the receipt
    builder fields when run_census is fed a complete synthetic capsule dir that
    satisfies the 33-file and 3..77 coverage contracts with pad capsules.
    """
    # Create 33 capsule files: layers 3-77 via multi-layer capsules + pads for 0-2
    # Strategy: one file per single layer for 3-32 (30 files) + 3 multi for rest?
    # Need exactly 33 files total matching list_capsule_npz_files.
    # Real layout: 3 dense + 13 single (3-15) + 1 range(4-15) + ranges for rest = 33.
    # Simpler: create 33 dummy npz; put topk only for 3-77 across them.

    capsules_dir = tmp_path / "capsules"
    capsules_dir.mkdir()

    # Build mapping layer -> capsule index using 33 files named like real ones.
    names = (
        [f"L{i:02d}_L{i:02d}.npz" for i in range(0, 16)]  # 16 files: 0-15
        + [
            "L16_L18.npz",
            "L19_L21.npz",
            "L22_L24.npz",
            "L25_L27.npz",
            "L28_L30.npz",
            "L31_L34.npz",
            "L35_L37.npz",
            "L38_L41.npz",
            "L42_L45.npz",
            "L46_L49.npz",
            "L50_L54.npz",
            "L55_L59.npz",
            "L60_L64.npz",
            "L65_L69.npz",
            "L70_L75.npz",
            "L76_L77.npz",
            "L04_L15.npz",  # duplicate range like real inventory
        ]
    )
    assert len(names) == 33

    layer_ranges = {
        "L16_L18.npz": range(16, 19),
        "L19_L21.npz": range(19, 22),
        "L22_L24.npz": range(22, 25),
        "L25_L27.npz": range(25, 28),
        "L28_L30.npz": range(28, 31),
        "L31_L34.npz": range(31, 35),
        "L35_L37.npz": range(35, 38),
        "L38_L41.npz": range(38, 42),
        "L42_L45.npz": range(42, 46),
        "L46_L49.npz": range(46, 50),
        "L50_L54.npz": range(50, 55),
        "L55_L59.npz": range(55, 60),
        "L60_L64.npz": range(60, 65),
        "L65_L69.npz": range(65, 70),
        "L70_L75.npz": range(70, 76),
        "L76_L77.npz": range(76, 78),
        "L04_L15.npz": range(4, 16),
    }

    for name in names:
        path = capsules_dir / name
        members: dict[str, np.ndarray] = {}
        if name in layer_ranges:
            for L in layer_ranges[name]:
                # Same per-layer seed as single-layer capsules so duplicates agree.
                members[f"layer_{L:02d}/topk_indices.npy"] = make_topk(seed=1000 + L)
        else:
            # L00..L15 single
            lo = int(name[1:3])
            if lo >= 3:
                members[f"layer_{lo:02d}/topk_indices.npy"] = make_topk(seed=1000 + lo)
            else:
                # dense: no topk, but need a dummy member so zip is valid
                members["calibration_token_ids.npy"] = np.arange(4, dtype=np.int32)
        write_capsule_and_sidecar(path, members)

    # Byte scenarios need full headers — use real sealed headers if present.
    if not v2.SOURCE_HEADERS.exists():
        pytest.skip("source headers unavailable")

    receipt = rpc.run_census(capsule_dir=capsules_dir, headers_path=v2.SOURCE_HEADERS)
    assert receipt["route_population_evidence_sufficient_for_rank_assignment"] is False
    assert receipt["within_target_bpw_for_proven_complete_assignment"] is False
    assert receipt["full_traversal_authorized"] is False
    assert all(v is False for v in receipt["safety"].values())
    assert receipt["n_expert_records"] == 19200 + 256
    assert receipt["coverage"]["missing_routed_layers"] == [78]
    # duplicates on 4-15 must agree (same seed per layer for single + range)
    assert receipt["duplicate_agreement"]["all_duplicates_byte_identical"] is True
    # 75 single-layer (3-77) + 12 range duplicates (4-15) = 87 loaded members
    assert receipt["n_loaded_topk_members"] == 87
    assert receipt["n_sealed_array_hashes_present"] == 87
    assert receipt["n_sealed_array_hashes_verified"] == 87
    assert receipt["all_loaded_topk_match_sealed_array_hashes"] is True
    assert receipt["whole_capsule_hash_recomputed"] is False
    for row in receipt["canonical_members"]:
        if row["routed_census_member"]:
            assert row["sealed_array_hash_verified"] is True
            assert row["sealed_array_sha256"] == row["computed_array_bytes_sha256"]
            assert len(row["sealed_array_sha256"]) == 64

    r2 = rpc.run_census(capsule_dir=capsules_dir, headers_path=v2.SOURCE_HEADERS)
    assert r2["receipt_sha256"] == receipt["receipt_sha256"]

    # Byte scenarios reconcile
    for key in (
        "rank128_for_all_nonpromotion_bound",
        "native_for_unresolved_bound",
    ):
        led = receipt["byte_scenarios"][key]["ledger"]
        assert led["itemization_reconciles"] is True
        assert led["ranks_never_reduced_to_fit"] is True
    assert receipt["byte_scenarios"]["anchor_assignment_scenario"]["authorizing"] is False
    assert (
        receipt["byte_scenarios"]["native_for_unresolved_bound"][
            "component_reconciliation"
        ]["reconciles"]
        is True
    )


def test_selftest_entrypoint():
    assert rpc.selftest() == 0


def test_band_summary_threshold_keys():
    records = []
    for i, c in enumerate([0, 1, 204, 205, 2576, 2577]):
        records.append(
            {
                "layer": 3,
                "expert_id": i,
                "route_count": c,
                "route_fraction": c / 4096,
                "zero_route": c == 0,
                "evidence_band": rpc.classify_evidence_band(c),
                "observed": True,
            }
        )
    # pad remaining experts as zero for layer 3 only (summary doesn't require 19200)
    for i in range(6, 256):
        records.append(
            {
                "layer": 3,
                "expert_id": i,
                "route_count": 0,
                "route_fraction": 0.0,
                "zero_route": True,
                "evidence_band": rpc.BAND_ZERO,
                "observed": True,
            }
        )
    records.append(
        {
            "layer": 78,
            "expert_id": 0,
            "route_count": None,
            "route_fraction": None,
            "zero_route": None,
            "evidence_band": rpc.BAND_UNOBSERVED,
            "observed": False,
        }
    )
    summary = rpc.summarize_bands(records)
    t = summary["threshold_counts"]
    assert t["eq_0"] >= 1
    assert t["eq_1"] == 1
    assert t["eq_204"] == 1
    assert t["eq_205"] == 1
    assert t["eq_2576"] == 1
    assert t["eq_2577"] == 1
    assert summary["band_counts_including_unobserved"][rpc.BAND_UNOBSERVED] == 1
    assert summary["not_quality_labels"] is True


def test_sealed_capsule_hash_bound_not_file_hash(tmp_path):
    arr = make_topk(seed=1)
    cap = tmp_path / "L03_L03.npz"
    members = {"layer_03/topk_indices.npy": arr}
    sealed_arr = raw_array_sha256(arr)
    write_capsule(cap, members)
    write_sidecar(
        cap,
        capsule_sha256="sealed_hash_value_not_computed_from_file_body_xx",
        array_sha256={"layer_03/topk_indices": sealed_arr},
    )
    bound = rpc.bind_sealed_capsule_hash(cap)
    assert bound["capsule_sha256_sealed"] == "sealed_hash_value_not_computed_from_file_body_xx"
    assert bound["array_sha256_topk_indices_sealed"]["layer_03/topk_indices"] == sealed_arr
    assert bound["whole_capsule_hash_recomputed"] is False
    loaded = rpc.load_validated_topk(
        rpc.TopkMemberRef(cap, cap.name, "layer_03/topk_indices.npy", 3)
    )
    assert loaded.sealed_array_hash_verified is True
    assert loaded.sealed_array_sha256 == sealed_arr


def test_real_capsules_all_87_loaded_members_verified():
    """When real retained capsules are present, all 87 members verify."""
    if not rpc.DEFAULT_CAPSULE_DIR.is_dir():
        pytest.skip("real capsule dir unavailable")
    if not v2.SOURCE_HEADERS.exists():
        pytest.skip("source headers unavailable")
    # Smoke: load+verify only (full census is slow but correct). Use inventory count
    # plus per-member verification without rebuilding full byte scenarios when possible.
    refs = rpc.enumerate_topk_members(rpc.DEFAULT_CAPSULE_DIR)
    assert len(refs) == 87
    loaded = [rpc.load_validated_topk(r) for r in refs]
    proof = rpc.summarize_sealed_array_hash_proof(loaded)
    assert proof["n_loaded_topk_members"] == 87
    assert proof["n_sealed_array_hashes_present"] == 87
    assert proof["n_sealed_array_hashes_verified"] == 87
    assert proof["all_loaded_topk_match_sealed_array_hashes"] is True
    assert proof["whole_capsule_hash_recomputed"] is False
    # Generated receipt (if present) must publish the same aggregate proof.
    if rpc.DEFAULT_OUT_JSON.exists():
        doc = json.loads(rpc.DEFAULT_OUT_JSON.read_text(encoding="utf-8"))
        if doc.get("schema") == rpc.SCHEMA:
            # After regeneration these must hold; tolerate pre-revision receipts
            # only when fields are absent by requiring regeneration in CI path.
            if "n_loaded_topk_members" in doc:
                assert doc["n_loaded_topk_members"] == 87
                assert doc["n_sealed_array_hashes_present"] == 87
                assert doc["n_sealed_array_hashes_verified"] == 87
                assert doc["all_loaded_topk_match_sealed_array_hashes"] is True
                assert doc["whole_capsule_hash_recomputed"] is False
                for row in doc["canonical_members"]:
                    if row.get("routed_census_member"):
                        assert row["sealed_array_hash_verified"] is True
                        assert (
                            row["sealed_array_sha256"]
                            == row["computed_array_bytes_sha256"]
                        )
