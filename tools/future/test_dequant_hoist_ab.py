"""The hoist is refuted on real activations, and the input is why.

The earlier receipt claimed 1.63x. It compared the hoist against a production
arm the synthetic input had slowed down. These tests pin the control that makes
that attribution legitimate - ARM A holding across the two runs - because
without it the correction is just a different number.
"""
from __future__ import annotations

import pytest

from tools.future import dequant_hoist_ab as ab


def test_all_three_arms_ran_in_both_inputs_or_it_is_not_a_matched_comparison():
    m = ab.measured()
    for run in (m["real_activation_run"], m["synthetic_control_run"]):
        assert run["production_gb_s"] > 0
        assert run["hoist_gb_s"] > 0
        assert run["arm_a_stripped_gb_s"] > run["hoist_gb_s"]


def test_a_missing_arm_refuses(monkeypatch):
    monkeypatch.setattr(ab, "REAL_REL", "receipts/future/NO_SUCH.json")
    with pytest.raises(ab.AbRefused, match="not on disk"):
        ab.measured()


def test_arm_a_is_the_control_and_a_drifting_control_voids_the_conclusion(monkeypatch):
    """Without this the production change could just be contention."""
    m = ab.measured()
    assert m["arm_a_holds_across_runs"]["relative_drift"] < 0.01

    real = ab._raw(ab.REAL_REL)
    drifted = {**real, "mlp": {**real["mlp"], "arm_a_stripped": {
        **real["mlp"]["arm_a_stripped"], "effective_gb_s": 700.0}}}
    monkeypatch.setattr(ab, "_raw", lambda rel: drifted if rel == ab.REAL_REL
                        else ab.__dict__["json"].loads(
                            (ab.REPO / rel).read_text()))
    with pytest.raises(ab.AbRefused, match="machine state is not held fixed"):
        ab.measured()


def test_the_hoist_is_not_faster_on_real_activations():
    m = ab.measured()
    assert m["verdict"] == "NO_SPEEDUP_ON_REAL_ACTIVATIONS"
    assert m["headline_hoist_over_production"] == pytest.approx(1.0, abs=0.01)
    assert ab.build()["ms_per_token_saved"] == 0.0


def test_the_input_moved_production_and_left_the_hoist_alone():
    d = ab.measured()["the_input_moved_production_not_the_hoist"]
    assert d["production_real_over_synth"] > 1.5, "production is input-sensitive"
    assert d["hoist_real_over_synth"] == pytest.approx(1.0, abs=0.02), \
        "the hoist is input-insensitive, which is what makes the ratio a fiction"


def test_the_mechanism_is_labelled_a_hypothesis_not_a_finding():
    m = ab.measured()["mechanism_is_not_established"]
    assert "hypothesis" in m
    assert "does not rest on it" in m


def test_the_fp_question_is_now_answered_on_real_activations():
    b = ab.bit_identity()
    assert b["measured_on"] == "real_captured_activation"
    assert b["all_exact"] is False, "a reassociating transform should not be exact"
    assert b["max_rel_fro"] < 1e-6, "and the error should sit at f32 epsilon"
    assert "rejected on speed, not on accuracy" in b["reading"]


def test_the_ladder_prediction_is_refuted_not_quietly_dropped():
    p = ab.prediction_vs_measurement()
    assert p["inside_bracket"] is False
    assert p["direction"] == "BELOW"
    assert "was not on the critical resource" in p["reading"]
    assert "does not kill the wider decode-tax target" in p["what_this_kills"]


def test_the_instrument_scar_is_emitted_and_scoped():
    c = ab.instrument_consequence()
    assert c["scar"] == "FILL_F32_IS_NOT_A_NEUTRAL_INPUT_FOR_THE_PRODUCTION_DEQUANT_ARM"
    assert "1.62x difference from the input alone" in c["statement"]
    assert "does not retroactively void the ALU roofline" in c["what_is_NOT_claimed"]
    assert "329.6" in c["what_is_NOT_claimed"], "the check that exonerates it"


def test_the_correction_is_stated_not_withdrawn_quietly():
    corr = ab.build()["correction"]
    assert "1.6338x" in corr
    assert "not withdrawn quietly" in corr
