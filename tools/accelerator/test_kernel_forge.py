"""Kernel Forge pins, including the bandit."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
import kernel_forge as kf  # noqa: E402


class Fake:
    def __init__(self, name, t, iqr=1.0):
        self.name, self.t, self.iqr = name, t, iqr
    def __str__(self):
        return self.name


def measure_of(cands):
    def m(c, reps):
        return c.t, c.iqr
    return m


def test_successive_halving_spends_less_than_flat_measurement():
    cands = [Fake(f"c{i}", 1.0 + i * 0.1) for i in range(16)]
    res = kf.successive_halving(cands, measure=measure_of(cands), rounds=3, base_reps=5)
    assert res["total_reps_spent"] < kf.exhaustive_cost(len(cands), 40)


def test_it_finds_the_clear_winner_when_the_landscape_is_not_flat():
    cands = [Fake("slow", 10.0), Fake("mid", 5.0), Fake("FAST", 1.0), Fake("slower", 20.0)]
    res = kf.successive_halving(cands, measure=measure_of(cands), rounds=3, base_reps=5)
    assert res["champion"].name == "FAST"


def test_an_unreliable_candidate_cannot_be_champion():
    """The bug that actually happened: the first bandit ranked on median alone and
    crowned a candidate that had failed the forge's own 10% IQR gate."""
    cands = [Fake("fast_but_jittery", 1.0, iqr=30.0), Fake("steady", 2.0, iqr=2.0)]
    res = kf.successive_halving(cands, measure=measure_of(cands), rounds=1, base_reps=5)
    assert res["champion"].name == "steady"
    # Check the BEHAVIOUR, not the mechanism. An earlier version of this test also
    # required the jittery candidate to appear in rejected_for_unreliability, which
    # pinned it to being rejected at the FINAL gate. The better fix sinks it during
    # elimination instead, so it never reaches that gate -- and the over-specific
    # assertion failed on a change that improved the code.
    eliminated_early = any("fast_but_jittery" in r["eliminated"] for r in res["rounds"])
    rejected_late = any(r["candidate"] == "fast_but_jittery"
                        for r in res["rejected_for_unreliability"])
    assert eliminated_early or rejected_late


def test_no_champion_when_every_survivor_is_unreliable():
    cands = [Fake("a", 1.0, iqr=40.0), Fake("b", 2.0, iqr=50.0)]
    res = kf.successive_halving(cands, measure=measure_of(cands), rounds=1, base_reps=5)
    assert res["champion"] is None
    assert len(res["rejected_for_unreliability"]) >= 1


def test_each_round_keeps_half_and_doubles_the_reps():
    cands = [Fake(f"c{i}", float(i)) for i in range(8)]
    res = kf.successive_halving(cands, measure=measure_of(cands), rounds=3, base_reps=5)
    reps = [r["reps_each"] for r in res["rounds"]]
    assert reps == [5, 10, 20]
    assert [r["kept"] for r in res["rounds"]] == [4, 2, 1]


def test_a_jittery_candidate_cannot_knock_out_steady_ones_during_elimination():
    """The failure this actually had: gating only at the END let a fast jittery
    candidate win round one, eliminate every steady candidate, and then be rejected
    itself -- leaving no champion at all. The gate fired correctly and the answer was
    still wrong."""
    cands = [Fake("jitter1", 0.1, iqr=40.0), Fake("jitter2", 0.2, iqr=40.0),
             Fake("steady", 5.0, iqr=1.0), Fake("steady2", 6.0, iqr=1.0)]
    res = kf.successive_halving(cands, measure=measure_of(cands), rounds=2, base_reps=5)
    assert res["champion"] is not None, "a reliable candidate existed and must survive"
    assert res["champion"].name == "steady"
