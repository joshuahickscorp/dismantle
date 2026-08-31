"""The hoist is built, measured, and its bit-identity is an artifact.

A speed number for a kernel nobody checked is a faster way to be wrong, so the
output comparison came first - and then the comparison itself was checked with a
planted error before its result was believed.
"""
from __future__ import annotations

import pytest

from tools.future import dequant_hoist_ab as ab


def test_all_three_arms_ran_or_it_is_not_a_matched_comparison():
    m = ab.measured()
    assert m["production_gb_s"] > 0
    assert m["hoist_gb_s"] > m["production_gb_s"]
    assert m["arm_a_stripped_gb_s"] > m["hoist_gb_s"]


def test_a_missing_arm_refuses(monkeypatch):
    monkeypatch.setattr(ab, "RAW_REL", "receipts/future/NO_SUCH.json")
    with pytest.raises(ab.AbRefused, match="not on disk"):
        ab.measured()


def test_the_span_recovery_is_the_load_robust_number():
    """Every ratio in a contended run is inflated together; the span is not."""
    m = ab.measured()
    assert m["span_recovered"] == pytest.approx(0.405, abs=0.01)
    assert m["arm_a_over_production"] > 2.0, "this run really is contended"
    assert "inflated together" in m["why_span_and_not_ratio"]


def test_the_measurement_disagrees_with_the_arithmetic_prediction():
    """1.6667 FMA/B was predicted; it behaves like k6 at 2.0."""
    p = ab.prediction_vs_measurement()
    assert p["measured_span_recovery"] == pytest.approx(
        p["k6_span_recovery_on_the_earlier_ladder"], abs=0.03
    )
    assert "adds a LOAD" in p["reading"]
    assert "an arithmetic-only model could not have said so" in p["reading"]


def test_the_bit_identity_is_flagged_as_an_artifact_not_a_result():
    b = ab.bit_identity()
    assert b["all_exact"] is True
    warn = b["MUST_NOT_BE_QUOTED_AS_BIT_IDENTICAL"]
    assert "dyadic" in warn and "four mantissa bits" in warn
    assert "LEAST likely to expose" in warn
    assert "UNTESTED" in warn


def test_the_comparison_was_itself_checked_with_a_planted_error():
    """An output check nobody falsified is not a check."""
    b = ab.bit_identity()
    assert "1e-6" in b["the_comparison_is_sound"]
    assert "39936 to 0" in b["the_comparison_is_sound"]
    assert "640 chunks" in b["the_comparison_is_sound"]


def test_it_says_what_would_settle_the_fp_question():
    b = ab.bit_identity()
    assert "real captured activation" in b["what_would_settle_it"]


def test_the_window_is_declared_contaminated():
    cb = ab.build()["claim_boundary"]
    assert "CONTAMINATED WINDOW" in cb
    assert "not promotable" in cb
    assert "says nothing about FP behaviour on real activations" in cb
