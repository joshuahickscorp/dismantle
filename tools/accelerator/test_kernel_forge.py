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


def test_bench_flags_a_reliability_verdict_taken_from_too_few_reps():
    """Measured: the same kernel's IQR ranged 1.84-13.38% across eight independent
    20-rep runs, so the gate verdict flipped run to run. A verdict below the stable
    threshold must say so rather than presenting itself as a property of the kernel."""
    import bench
    calls = {"n": 0}
    def work():
        calls["n"] += 1
    few = bench.time_arm(work, reps=20, warmup=2)
    assert few["reliability_verdict_is_stable"] is False
    assert "may flip on a rerun" in few["reliability_caveat"]
    many = bench.time_arm(work, reps=bench.STABLE_RELIABILITY_MIN_REPS, warmup=2)
    assert many["reliability_verdict_is_stable"] is True
    assert many["reliability_caveat"] is None


def test_per_tg_is_the_variable_and_the_grid_repeats_itself():
    """Candidates sharing per_tg are the same physical point, so the default grid
    explores far fewer distinct configurations than it has candidates."""
    cands = kf.generate(1 << 20)
    groups = kf.collapse_groups(cands)
    assert len(cands) == 15 and len(groups) == 7
    assert sorted(groups) == [64, 128, 256, 512, 1024, 2048, 4096]
    # the repeats are real: per_tg 256 is reached three different ways
    assert sorted(groups[256]) == ["tg128_ept2", "tg256_ept1", "tg64_ept4"]


def test_the_threadgroup_prior_is_documented_as_primitive_specific():
    """The prior's floor was measured on the fused chain and does NOT transfer -- a
    write-only kernel is still 1.9x off its floor at per_tg 64. Pinning the caveat in
    the source is what stops the tuple being read as a property of the machine."""
    src = Path(kf.__file__).read_text()
    assert "PRIMITIVE-SPECIFIC" in src and "CHAIN-SHAPED PRIOR" in src


def test_a_sweep_that_ran_nothing_is_not_a_passing_sweep():
    """The first shape fuzz in this program reported ZERO FAILURES for four primitives
    it never executed -- one filter demanded every dimension be a multiple of 8 while
    the grid held exactly one such value. A coverage counter is part of the check."""
    with pytest.raises(ValueError, match="ZERO cases"):
        kf.shape_sweep([], lambda c: 0.0, name="empty")


def test_shape_sweep_records_a_raise_as_a_failure_not_a_skip():
    """A shape the kernel REFUSES is a shape the caller needs told about; skipping it
    would let a refusal read as a pass."""
    def check(case):
        if case[0] == 3:
            raise RuntimeError("threadgroup memory exceeded")
        return 0.0
    r = kf.shape_sweep([(1,), (3,), (7,)], check, name="raises")
    assert r["cases_run"] == 3 and r["failures"] == 1 and not r["all_ok"]
    assert "threadgroup memory" in r["failing"][0]["raised"]
    good = kf.shape_sweep([(1,), (7,)], check, name="clean")
    assert good["all_ok"] and good["cases_run"] == 2


def test_shape_sweep_would_have_caught_the_tiled_matmul_launch_bug():
    """The defect: correct whenever m is a multiple of the tile, wrong otherwise. A
    single-shape check at 64x64x64 passes it; the sweep does not."""
    def check(case):
        m, tile = case
        return 0.0 if m % tile == 0 else 3.84      # the measured 60x60x60 error
    assert kf.shape_sweep([(64, 16)], check, name="one")["all_ok"]
    swept = kf.shape_sweep([(m, 16) for m in kf.AWKWARD_SHAPES],
                                     check, name="many")
    assert not swept["all_ok"] and swept["failures"] == len(kf.AWKWARD_SHAPES)


def test_a_sweep_whose_control_never_fails_is_refused():
    """A probabilistic sweep reporting zero failures is exactly the shape of a check
    that cannot fail. If the broken arm passes everywhere, the clean result is
    evidence of nothing and the sweep says so instead of reporting green."""
    with pytest.raises(ValueError, match="broken control passed at every case"):
        kf.swept_with_control([(64,), (256,)], lambda c: 0.0, lambda c: 0.0,
                              name="vacuous")


def test_the_control_may_be_blind_at_some_widths_and_that_is_reported():
    """Stripping one barrier from AirNorm is caught at threadgroups 64 through 1024
    and is EXACT at 32, where the whole threadgroup is one simdgroup. A control run
    only at 32 would have called the broken kernel fine, so the widths where the
    control is blind are part of the result."""
    def broken(case):
        return 0.0 if case[0] == 32 else 1.0      # the measured lockstep behaviour
    r = kf.swept_with_control([(w,) for w in kf.WIDTH_PRIOR],
                              lambda c: 0.0, broken, name="lockstep", repeat=3)
    assert r["all_ok"] and r["failures"] == 0
    assert r["control_blind_at"] == [[32]]
    assert len(r["control_detected_at"]) == 5
    assert r["executions"] == len(kf.WIDTH_PRIOR) * 3 * 2


def test_repeat_sets_the_detection_floor_not_the_case_list():
    """A width-dependent defect is usually a RACE and therefore probabilistic; the
    resolving power comes from repeats, and the floor is reported so nobody reads a
    clean sweep as proof that no race exists."""
    one = kf.swept_with_control([(1,)], lambda c: 0.0, lambda c: 1.0, repeat=1)
    many = kf.swept_with_control([(1,)], lambda c: 0.0, lambda c: 1.0, repeat=8)
    assert one["detection_floor_per_run"] > many["detection_floor_per_run"]
    assert many["detection_floor_per_run"] < 0.10
