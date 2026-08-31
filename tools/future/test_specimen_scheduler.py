"""Selection is computed, and a dead hypothesis loads nothing.

Two acceptance points from S027: the order must not be FIFO, and a scar query
must be PROVEN to suppress a load rather than merely consulted.
"""
from __future__ import annotations

import pytest

from tools.future import specimen_scheduler as ss


def test_architecture_distance_does_not_call_a_near_neighbour_far():
    """The bug this test exists to prevent recurring."""
    assert ss.distance("qwen4_exp") == "NEAR"
    assert ss.distance("qwen3") == "MID", "qwen3 is a near neighbour, not an adversary"
    assert ss.distance("qwen3_moe") == "MID"
    assert ss.distance("deepseek_v4") == "FAR"
    assert ss.distance("mistral3") == "FAR"


def test_an_unknown_architecture_is_far_not_near():
    assert ss.distance(None) == "FAR"
    assert ss.distance("") == "FAR"


def test_the_stem_is_the_alphabetic_lineage():
    assert ss._stem("qwen3_vl_moe") == "qwen"
    assert ss._stem("deepseek_v4") == "deepseek"
    assert ss._stem("mistral3") == "mistral"


def test_a_ranking_needs_a_question():
    with pytest.raises(ss.ScheduleRefused, match="rank FOR"):
        ss.rank(hypothesis_family="")


def test_the_order_is_not_the_registry_order():
    p = ss.not_fifo_proof()
    assert p["differs"] is True
    assert p["ranked_order"] != p["registry_order"]
    assert "not by registry or arrival order" in \
        ss.rank(hypothesis_family="x")["selection_is_not_fifo"]


def test_a_dead_hypothesis_suppresses_every_load():
    p = ss.scar_suppression_proof()
    assert p["all_suppressed_when_dead"] is True
    assert p["n_suppressed_for_dead_family"] > 0
    assert "NOTHING IS LOADED" in p["statement"]


def test_a_live_hypothesis_suppresses_none():
    p = ss.scar_suppression_proof()
    assert p["none_suppressed_when_live"] is True
    assert p["n_suppressed_for_live_family"] == 0


def test_a_suppressed_specimen_scores_zero():
    r = ss.rank(hypothesis_family="low_rank")
    assert all(x["score"] == 0.0 for x in r["ranked"] if x["suppressed"])


def test_cost_is_the_measured_load_time_not_a_nominal_one():
    r = ss.rank(hypothesis_family="x_none")
    for x in r["ranked"]:
        assert x["measured_cold_load_minutes"] > 0
    biggest = max(r["ranked"], key=lambda x: x["source_gb"])
    assert biggest["measured_cold_load_minutes"] > 60, \
        "the 360 GB incumbent source really is over an hour of cold read"


def test_memory_admission_is_carried_into_the_ranking():
    r = ss.rank(hypothesis_family="x_none")
    assert r["n_refused_by_memory"] > 0, "most sealed specimens do not fit"
    assert any(x["exceeds_total_memory"] for x in r["ranked"])


def test_a_far_architecture_outranks_a_near_one_at_equal_cost():
    """S027 §49: the informative specimen is the one that discriminates."""
    assert ss.ROLE_WEIGHT["FAR"] > ss.ROLE_WEIGHT["MID"] > ss.ROLE_WEIGHT["NEAR"]


def test_information_gain_is_an_input_not_an_invention():
    w = ss.build()["what_this_does_not_do"]
    assert "it does not estimate information gain per specimen" in w
    assert "ranking on a number nobody measured" in w
