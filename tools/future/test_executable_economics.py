"""Pins for tools/future/executable_economics.py.

A ratio without bytes_added is not a candidate. A 100% MLP-byte removal
that also adds 100 extra FLOPs per output element is scored on both
terms, not on bytes alone. The S020 §20 bar is 1% of complete token
time, or a reusable family, or a high-information falsifier.
"""
from __future__ import annotations

import json

import pytest

from tools.future import executable_economics as ee
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
)
from tools.future.physical_primitives import ATLAS_PRIMITIVES


def test_bytes_removed_without_bytes_added_is_refused():
    """A compression ratio without executable economics is not a candidate."""
    with pytest.raises(ee.IncompleteEconomics, match="bytes_added"):
        ee.score(bytes_removed=1_000_000)
    with pytest.raises(ee.IncompleteEconomics, match="executable economics"):
        ee.score(bytes_removed=ee.MLP_ACTIVE_BYTES, bytes_added=None)
    with pytest.raises(ee.IncompleteEconomics, match="bytes_added"):
        ee.score(bytes_removed=0)


def test_explicit_zero_bytes_added_is_a_complete_claim():
    row = ee.score(
        bytes_removed=1_000_000,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
    )
    assert row["ok"] is True
    assert row["bytes_added_supplied"] is True
    assert row["bytes_added"]["total"] == 0
    assert row["bytes_removed"] == 1_000_000
    assert row["terms"]["byte_ms_delta"] < 0.0


def test_mlp_full_removal_with_extra_flops_scores_both_terms():
    """Removing 100% of MLP bytes AND adding 100 extra FLOPs per output
    element must move the score on both terms, not on bytes alone.
    """
    bytes_only = ee.score(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added=0,
        extra_flops_per_output_element=0.0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
    )
    both = ee.score(
        bytes_removed=ee.MLP_ACTIVE_BYTES,
        bytes_added=0,
        extra_flops_per_output_element=100.0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
    )
    assert bytes_only["terms"]["byte_ms_delta"] < 0.0
    assert bytes_only["terms"]["flop_ms_delta"] == 0.0
    assert both["terms"]["flop_ms_delta"] > 0.0
    assert both["terms"]["byte_ms_delta"] == pytest.approx(
        bytes_only["terms"]["byte_ms_delta"]
    )
    assert both["n_output_elements"] == ee.ORGAN_OUTPUT_ELEMENTS["mlp"]
    assert both["n_output_elements"] == 2_555_904
    # Extra FLOPs eat into the byte save; they are not dropped.
    assert both["predicted_ms_delta"] > bytes_only["predicted_ms_delta"]
    assert both["predicted_token_ms"] > bytes_only["predicted_token_ms"]
    assert both["predicted_ms_delta"] != both["terms"]["byte_ms_delta"]
    assert both["predicted_ms_delta"] == pytest.approx(
        both["terms"]["byte_ms_delta"] + both["terms"]["flop_ms_delta"]
    )
    # The byte term is the whole MLP organ at its own measured rate.
    assert bytes_only["terms"]["byte_ms_delta"] == pytest.approx(
        -ee.bytes_to_ms(ee.MLP_ACTIVE_BYTES, ee.MLP_GB_S)
    )
    assert bytes_only["terms"]["byte_ms_delta"] == pytest.approx(-ee.MLP_MS, abs=1e-3)


def test_bytes_added_five_fields_reduce_the_save():
    removed = 534_773_760  # quantize_aux_u8
    no_add = ee.score(
        bytes_removed=removed,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
    )
    with_add = ee.score(
        bytes_removed=removed,
        bytes_added={
            "generator": 10_000_000,
            "embeddings": 1_000_000,
            "residuals": 2_000_000,
            "metadata": 3_000_000,
            "state": 4_000_000,
        },
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
    )
    assert with_add["bytes_added"]["total"] == 20_000_000
    assert with_add["net_bytes"] == removed * -1 + 20_000_000
    assert with_add["terms"]["byte_added_ms"] > 0.0
    assert with_add["predicted_ms_saved"] < no_add["predicted_ms_saved"]
    for key in ee.BYTES_ADDED_FIELDS:
        assert key in with_add["bytes_added"]


def test_one_percent_bar_and_family_override():
    bar_bytes = int(ee.S020_SECTION_20_BAR_MS / 1000.0 * ee.MLP_GB_S * 1e9)
    # Comfortably under the bar.
    small = ee.score(
        bytes_removed=bar_bytes // 4,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        candidate_id="tiny_header_pack",
    )
    assert small["s020_section_20"]["clears_time_bar"] is False
    assert small["verdict"] == "IMMATERIAL"

    over = ee.score(
        bytes_removed=int(bar_bytes * 1.5),
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
    )
    assert over["s020_section_20"]["clears_time_bar"] is True
    assert over["verdict"] == "MATERIAL"
    assert "clears_s020_section_20_time_bar" in over["verdict_reasons"]

    family = ee.score(
        bytes_removed=bar_bytes // 4,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        reusable_family=True,
        candidate_id="a_family",
    )
    assert family["s020_section_20"]["clears_time_bar"] is False
    assert family["verdict"] == "MATERIAL"
    assert "reusable_representation_family" in family["verdict_reasons"]

    falsifier = ee.score(
        bytes_removed=0,
        bytes_added=0,
        high_information_falsifier=True,
        candidate_id="a_falsifier",
    )
    assert falsifier["verdict"] == "MATERIAL"
    assert "high_information_falsifier" in falsifier["verdict_reasons"]


def test_dead_status_is_immaterial_even_if_the_byte_save_was_large():
    row = ee.score(
        bytes_removed=ee.MLP_CODE_BYTES,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        reusable_family=True,
        status="MEASURED_NEGATIVE",
    )
    assert row["live"] is False
    assert row["verdict"] == "IMMATERIAL"
    assert row["predicted_ms_saved"] > ee.S020_SECTION_20_BAR_MS


def test_new_representation_bandwidth_is_an_assumption_with_a_range():
    row = ee.score(
        bytes_removed=100_000_000,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="TiledProjection",
    )
    assume = row["assumptions"]
    assert assume["bandwidth_regime"] == "unknown_new_representation"
    assert assume["bandwidth_is_assumption"] is True
    lo, hi = assume["bandwidth_gb_s_range"]
    assert lo < hi
    assert lo == pytest.approx(ee.AFFINE_Q2_GB_S_AT_5MB)
    assert hi == pytest.approx(ee.LM_HEAD_GB_S)
    # The clean roof is excluded from the packed-representation range.
    assert hi < ee.CLEAN_GEMV_GB_S
    dlo, dhi = row["predicted_ms_delta_range"]
    assert dlo < dhi
    assert row["predicted_tps_range"][0] < row["predicted_tps_range"][1]
    assert "ASSUMPTION" in assume["bandwidth_note"]


def test_affine_q2_is_not_dressed_as_497():
    row = ee.score(
        bytes_removed=534_773_760,
        bytes_added=0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
    )
    assume = row["assumptions"]
    assert assume["bandwidth_regime"] == "affine_q2_family"
    lo, hi = assume["bandwidth_gb_s_range"]
    assert hi == pytest.approx(ee.AFFINE_Q2_SATURATED_GB_S)
    assert hi < ee.LM_HEAD_GB_S
    assert assume["bandwidth_gb_s_nominal"] == pytest.approx(ee.MLP_GB_S)


def test_dispatch_class_is_class_dependent():
    mlp = ee.score(
        bytes_removed=0,
        bytes_added=0,
        dispatch_delta=-48,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
        dispatch_class="mlp_gqa_norm_fusion",
    )
    dn = ee.score(
        bytes_removed=0,
        bytes_added=0,
        dispatch_delta=-48,
        organ="deltanet",
        consuming_primitive="FusedDecodeCompute",
        dispatch_class="deltanet_ba",
    )
    assert mlp["terms"]["dispatch_ms_delta"] == pytest.approx(-48 * 6.25 / 1000.0)
    assert dn["terms"]["dispatch_ms_delta"] == pytest.approx(-48 * 2.884 / 1000.0)
    assert mlp["terms"]["dispatch_ms_delta"] != dn["terms"]["dispatch_ms_delta"]


def test_unknown_primitive_is_still_scored_with_a_range():
    row = ee.score(
        bytes_removed=10_000,
        bytes_added={"metadata": 100},
        organ="mlp",
        consuming_primitive="NotAnAtlasPrimitive",
    )
    assert row["ok"] is True
    assert row["assumptions"]["bandwidth_is_assumption"] is True
    assert row["consuming_primitive"] == "NotAnAtlasPrimitive"


def test_bytes_removed_past_the_organ_is_refused():
    with pytest.raises(ee.EconomicsRefuse, match="exceeds"):
        ee.score(
            bytes_removed=ee.MLP_ACTIVE_BYTES + 1,
            bytes_added=0,
            organ="mlp",
        )


def test_recorded_catalog_covers_aux_code_deltanet_and_dispatch_size():
    ranked = ee.rank_recorded()
    ids = [r["id"] for r in ranked]
    assert len(ids) == len(set(ids))

    aux = json.loads((RECEIPTS / "MLP_AUXILIARY_INFORMATION.json").read_text())
    code = json.loads((RECEIPTS / "MLP_CODE_INFORMATION.json").read_text())
    dn = json.loads((RECEIPTS / "DELTANET_REPRESENTATION.json").read_text())
    for src in (aux, code, dn):
        for cand in src["candidates"]:
            assert cand["id"] in ids, cand["id"]

    assert "surviving_dispatch_size_amortize_sub20mb" in ids
    assert "dispatch_size_concat_to_lm_head_mb" in ids
    assert "group_size_256" in ids
    assert "group_size_512" in ids
    assert "group_size_1024" in ids
    assert "quantize_aux_u8" in ids
    assert "entropy_coded_code_stream" in ids
    assert "pack_headers" in ids
    assert "function_replacement" in ids
    assert "fused_update_consume" in ids

    by_id = {r["id"]: r for r in ranked}
    assert by_id["quantize_aux_u8"]["verdict"] == "MATERIAL"
    assert by_id["group_size_1024"]["verdict"] == "MATERIAL"
    assert by_id["entropy_coded_code_stream"]["verdict"] == "MATERIAL"
    assert by_id["pack_headers"]["verdict"] == "IMMATERIAL"
    assert by_id["pack_headers"]["predicted_ms_saved"] < ee.S020_SECTION_20_BAR_MS
    assert by_id["surviving_dispatch_size_amortize_sub20mb"]["verdict"] == "MATERIAL"
    assert "reusable_representation_family" in by_id[
        "surviving_dispatch_size_amortize_sub20mb"
    ]["verdict_reasons"] or "high_information_falsifier" in by_id[
        "surviving_dispatch_size_amortize_sub20mb"
    ]["verdict_reasons"]
    assert by_id["dispatch_size_concat_to_lm_head_mb"]["live"] is False
    assert by_id["dispatch_size_concat_to_lm_head_mb"]["verdict"] == "IMMATERIAL"

    # Rank is by predicted save, largest first.
    saves = [r["predicted_ms_saved"] for r in ranked]
    assert saves == sorted(saves, reverse=True)

    # fused_update_consume is the BA-delta class, not the 6.25 us class.
    fused = by_id["fused_update_consume"]
    assert fused["dispatch_delta"] == -48
    assert fused["assumptions"]["dispatch_class"] == "deltanet_ba"
    assert fused["terms"]["dispatch_ms_delta"] == pytest.approx(
        -48 * 2.884 / 1000.0, abs=1e-3
    )


def test_group_size_1024_matches_causal_budget_arithmetic():
    by_id = {r["id"]: r for r in ee.rank_recorded()}
    row = by_id["group_size_1024"]
    # 1.002700800 GB at the MLP's own 344.1 GB/s.
    expected = 1_002_700_800 / 344.1 * 1e-6
    assert row["predicted_ms_saved"] == pytest.approx(expected, rel=1e-4)
    assert row["bytes_removed"] == 1_002_700_800
    assert row["bytes_added_total"] == 0


def test_rejected_dense_remat_is_scored_on_added_bytes_not_removal_alone():
    by_id = {r["id"]: r for r in ee.rank_recorded()}
    gen = by_id["generated_tensors"]
    assert gen["status"] == "REJECTED_DENSE_REMAT"
    assert gen["bytes_added_total"] == ee.MLP_PARAMS * ee.F16_BYTES
    assert gen["net_bytes"] > 0
    assert gen["predicted_ms_saved"] < 0.0
    assert gen["verdict"] == "IMMATERIAL"


def test_atlas_primitives_are_the_vocabulary():
    for name in (
        "FusedDecodeCompute",
        "TiledProjection",
        "LocalStateMachine",
        "LayoutTransform",
        "MoveOrRecompute",
        "DirectRoutedAccumulate",
        "ConditionalPhysicalProgram",
    ):
        assert name in ATLAS_PRIMITIVES


def test_build_emits_sealed_static_only_receipt():
    path = ee.record()
    assert path.parent == RECEIPTS
    assert path.name == "EXECUTABLE_ECONOMICS.json"
    doc = json.loads(path.read_text())
    assert doc["schema"] == ee.SCHEMA
    assert doc["version"] == 1
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["seal_sha256"]
    assert doc["guards"]["bytes_removed_without_bytes_added_refused"] is True
    assert doc["guards"]["mlp_full_removal_scores_both_terms"] is True
    assert doc["guards"]["mlp_full_removal_flop_ms_at_100"] > 0.0
    assert doc["n_candidates"] == len(doc["candidates_ranked"])
    assert doc["n_live_material"] >= 8
    assert "quantize_aux_u8" in doc["live_material_ranked"]
    assert "group_size_1024" in doc["live_material_ranked"]
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        # Nested keys of these names would have already raised. Pin the
        # predicted_* spellings so a closer cannot rename them back.
        assert key not in doc
    # Affine-Q2 saturation is in the cited constants; 497 is not a default.
    cited = doc["measured_constants_cited"]
    assert cited["affine_q2_saturated_gb_s"] == pytest.approx(376.7)
    assert cited["lm_head_gb_s"] == pytest.approx(497.4)
    assert cited["gpu_reduction_for_71_at_current_executor"] == pytest.approx(
        0.528, abs=0.005
    )


def test_every_ranked_row_carries_a_bandwidth_range():
    for row in ee.rank_recorded():
        lo, hi = row["assumptions"]["bandwidth_gb_s_range"]
        assert lo > 0.0 and hi > 0.0
        assert lo <= hi
        assert "bandwidth_regime" in row["assumptions"]
        assert "bandwidth_note" in row["assumptions"]
        dlo, dhi = row["predicted_ms_delta_range"]
        assert dlo <= dhi
