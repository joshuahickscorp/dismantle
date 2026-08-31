"""Divergence has a denominator, and it is not the number of asks.

The first honest 30m model-bearing run asked choose() 56 times and diverged 0
times, which reads as "the model agreed 56 independent times". Only FIVE of
those asks were distinct - one prompt fired 52 times - and those 5 distinct
prompts produced 5 distinct replies. At temperature 0 a deterministic body
re-asked a byte-identical question answers it identically by construction. The
repeat count is a fact about the frontier; reporting it as agreement makes it
look like a fact about the model's judgment, 11x overstated.
"""
from __future__ import annotations

from tools.future import model_bearing as mb


def _choose(digest: str, *, diverged: bool = False):
    return {
        "kind": "choose",
        "cognition": mb.AVAILABLE,
        "prompt_sha256": digest,
        "diverged_from_fixed_policy": diverged,
        "chose": {"id": "WU.X"} if diverged else None,
        "counts_as_decision": True,
        "reason": "because",
        "model_decided": "WU.X",
    }


def test_repeated_asks_do_not_inflate_the_denominator():
    log = [_choose("aaa") for _ in range(52)] + [_choose(f"d{i}") for i in range(4)]
    got = mb.materially_participated(log)
    scope = got["divergence_scope"]
    assert scope["n_choose_asks"] == 56
    assert scope["n_distinct_choose_asks"] == 5
    assert scope["n_repeated_choose_asks"] == 51


def test_the_finding_states_the_distinct_count_not_the_total():
    log = [_choose("aaa") for _ in range(52)] + [_choose(f"d{i}") for i in range(4)]
    finding = mb.materially_participated(log)["publishable_finding"]
    assert "5 DISTINCT" in finding
    assert "56 total" in finding


def test_distinct_asks_still_counted_when_all_differ():
    log = [_choose(f"h{i}") for i in range(7)]
    scope = mb.materially_participated(log)["divergence_scope"]
    assert scope["n_distinct_choose_asks"] == 7
    assert scope["n_repeated_choose_asks"] == 0


def test_a_real_divergence_still_counts():
    log = [_choose("a"), _choose("b", diverged=True)]
    got = mb.materially_participated(log)
    assert got["divergence_count"] == 1
    assert got["publishable_finding"] is None
