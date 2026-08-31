"""The 71 ledger must read its numbers, not remember them.

G072: "the ledger recomputes from landed receipts rather than from hand-entered
numbers, and a test proves it refuses to sum overlapping levers."

Every constant used to be a literal with a receipt PATH beside it. That is a
citation, not a recomputation: when the receipt moved, the literal stayed and
nothing said so. One drift was already sitting there - the ledger carried
was_worth_tps 1.24 while WALL_GPU_RECONCILIATION.derived says 1.214.
"""
from __future__ import annotations

import json

import pytest

from tools.future import causal_budget_71 as cb


def test_every_citation_resolves_from_disk():
    got = cb.resolve_all()
    assert len(got) == len(cb.CITATIONS)
    assert got["host_gap_ms"] == pytest.approx(0.9894, abs=1e-9)
    assert got["group_size_1024_bytes"] == 1_002_700_800


def test_drift_raises_rather_than_keeping_the_literal():
    with pytest.raises(cb.CitationDrift):
        cb.resolve(
            "receipts/future/WALL_GPU_RECONCILIATION.json",
            ["derived", "host_gap_ms_per_token"],
            0.5,
        )


def test_missing_receipt_raises_and_never_falls_back():
    with pytest.raises(cb.CitationMissing):
        cb.resolve("receipts/future/NO_SUCH_RECEIPT.json", ["x"], 1.0)


def test_missing_field_raises():
    with pytest.raises(cb.CitationMissing):
        cb.resolve(
            "receipts/future/WALL_GPU_RECONCILIATION.json",
            ["derived", "not_a_field"],
            1.0,
        )


def test_selector_must_match_exactly_one_row():
    with pytest.raises(cb.CitationMissing):
        cb.resolve(
            "receipts/future/MLP_AUXILIARY_INFORMATION.json",
            ["group_size_curve", {"group_size": 999999}, "bytes_eliminated_vs_incumbent"],
            0.0,
        )


def test_non_numeric_citation_is_refused():
    with pytest.raises(cb.CitationMissing):
        cb.resolve("receipts/future/MLP_AUXILIARY_INFORMATION.json", ["schema"], 1.0)


def test_build_cannot_emit_without_resolving_citations():
    d = cb.build()
    assert set(d["citations_resolved"]) == {c["id"] for c in cb.CITATIONS}
    assert len(d["citations"]) == len(cb.CITATIONS)
    json.dumps(d)  # the receipt must stay serialisable


def test_the_drift_that_was_actually_there_is_gone():
    """was_worth_tps 1.24 was never in the receipt; it says 1.214."""
    row = next(r for r in cb.REFUTED_LEVERS if r["id"] == "eliminate_all_host_gap")
    assert row["was_worth_tps"] == 1.214
    assert cb.resolve_all()["host_gap_worth_tps"] == pytest.approx(1.214, abs=1e-9)


def test_the_residual_is_the_same_run_remainder_not_a_cross_receipt_subtraction():
    """0.321 was 0.095 of remainder plus 0.226 of trace overhead and variation.

    I computed the residual by subtracting ORGAN_BANDWIDTH's covered 27.733 from
    WALL_GPU_RECONCILIATION's 28.054 and called the difference unattributed GPU
    time. Different runs, and the organ run has the region trace ON at a measured
    1.8% of GPU time. The honest remainder comes from the run that measured both
    the parts and the total.
    """
    r = cb.causal_residual()
    assert r["gpu_residual_ms"] == pytest.approx(0.095, abs=1e-6)
    assert cb.UNATTRIBUTED_GPU_MS == pytest.approx(r["gpu_residual_ms"], abs=1e-9)
    # same-run identity: covered + remainder == the traced total
    assert r["organ_ms_covered_in_receipt"] + r["gpu_residual_ms"] == pytest.approx(
        r["traced_gpu_ms"], abs=1e-6
    )


def test_the_traced_untraced_gap_is_labelled_overhead_not_physics():
    r = cb.causal_residual()
    assert r["traced_vs_untraced_ms"] == pytest.approx(0.226, abs=1e-3)
    assert r["trace_overhead_pct"] == pytest.approx(1.8, abs=1e-6)
    assert r["gpu_residual_ms"] < r["traced_vs_untraced_ms"], (
        "if the remainder ever exceeds the run-to-run gap, the comparison is "
        "no longer dominated by overhead and this framing must be revisited"
    )


def test_the_remainder_is_named_not_miscellaneous():
    """G072: the residual must not stay a miscellaneous bucket."""
    r = cb.causal_residual()
    named = r["gpu_residual_is_named_not_mysterious"]
    for part in ("norms", "embedding row", "A_log", "dt_bias"):
        assert part in named, f"the remainder does not name {part}"


def test_host_gap_is_still_a_subtraction():
    r = cb.causal_residual()
    assert r["wall_ms"] - r["untraced_gpu_ms"] == pytest.approx(r["host_gap_ms"], abs=1e-9)


def test_the_ladder_baseline_moved_to_the_measured_current_body():
    """G072: when a win lands, the baseline MOVES. It has."""
    b = cb.causal_residual()["baseline_moved"]
    assert b["current_body_ms"] == pytest.approx(27.2896, abs=1e-3)
    assert b["current_body_tps"] == pytest.approx(36.644, abs=1e-2)
    assert b["ms_removed_since"] > 1.0
    # and it must say what did NOT move with it
    assert "PER-ORGAN ms in ORGANS are from the pre-widen_f4" in b[
        "what_is_still_from_the_old_census"
    ]


def test_organs_do_not_explain_all_of_traced_gpu_time():
    r = cb.causal_residual()
    assert r["organs_explain_of_traced_gpu"] < 1.0, "zero would mean fitted, not measured"
    assert r["organs_explain_of_traced_gpu"] > 0.99, "coverage is 99.97% of bytes"


def test_the_next_prey_is_not_the_remainder():
    """ORGAN_BANDWIDTH's finding is that the loss is uniform, not localised."""
    prey = cb.causal_residual()["next_prey"]
    assert "no hot organ" in prey
    assert "341.9" in prey and "360.0" in prey
