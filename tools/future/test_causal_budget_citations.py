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


def test_causal_residual_is_gpu_minus_organs_not_wall_minus_parts():
    """The reassuring zero would have come from the subtraction, not the measurement."""
    r = cb.causal_residual()
    assert r["gpu_residual_ms"] == pytest.approx(0.321, abs=5e-4)
    assert 0.010 < r["gpu_residual_frac_of_wall"] < 0.012
    # host_gap is wall - gpu exactly, which is why it is not the residual
    # This identity IS the point: host_gap is derived, so it can never surprise.
    # Assert the arithmetic, not the prose - a test that greps its own docstring
    # passes when the sentence is reworded and fails when it is improved.
    assert r["wall_ms"] - r["gpu_ms"] == pytest.approx(r["host_gap_ms"], abs=1e-9)
    assert r["gpu_residual_ms"] != pytest.approx(r["wall_ms"] - r["gpu_ms"], abs=1e-3)


def test_residual_refuses_to_pretend_the_baseline_is_current():
    r = cb.causal_residual()
    assert r["baseline_is_stale"]["current_body_ms"] == "UNKNOWN_UNTIL_G075"


def test_organs_do_not_explain_all_of_gpu_time():
    r = cb.causal_residual()
    assert r["organs_explain_of_gpu"] < 1.0, "a residual of zero would mean fitted, not measured"
    assert r["gpu_residual_is_unattributed_not_absent"] is True


def test_reconstruction_matches_the_measured_wall():
    """Summing only the organs you can name reports a body that never ran.

    token_ms() rebuilt the token from four organs plus the host gap and landed at
    28.722 ms while the measured wall is 29.0434 - every rung in the ladder was
    0.321 ms optimistic. The unattributed GPU term is now part of the
    reconstruction, so this identity holds or something moved.
    """
    r = cb.causal_residual()
    assert cb.token_ms() == pytest.approx(r["wall_ms"], abs=1e-3)


def test_the_unattributed_term_is_the_gpu_residual():
    r = cb.causal_residual()
    assert cb.UNATTRIBUTED_GPU_MS == pytest.approx(r["gpu_residual_ms"], abs=5e-4)


def test_host_gap_lever_is_derived_not_remembered():
    """1.214 must fall out of the arithmetic, not be typed in."""
    now = cb.token_ms()
    assert cb.tps(now - cb.HOST_GAP_MS) - cb.tps(now) == pytest.approx(1.214, abs=5e-3)
