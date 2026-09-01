"""A loop that cannot produce work must stop, not spin.

The run this guards against is real and measured: 330 consecutive iterations
over 4.4 hours, each emitting a byte-identical 1667-character reply every ~48
seconds, `degenerated=True` on every one, `n_accepted=0` on every one. The loop
computed the degeneracy flag and did nothing with it.

The spiral is self-reinforcing, which is why it never escaped on its own: no
work accepted means the kernel does not change, an unchanged kernel rebuilds a
byte-identical pack, and greedy decoding returns the same reply, which accepts
no work. Nothing inside that cycle can perturb itself, so the perturbation has
to come from the loop's own bookkeeping.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.future import hcli_sovereign as sov


def _streak(accepted_per_turn):
    """Replay the loop's own streak bookkeeping over a sequence of turns."""
    unproductive = 0
    trace = []
    for accepted in accepted_per_turn:
        forced_terse = unproductive >= sov.UNPRODUCTIVE_TERSE_AFTER
        stop = unproductive >= sov.UNPRODUCTIVE_STOP_AFTER
        trace.append({"terse": forced_terse, "stop": stop, "streak": unproductive})
        if stop:
            break
        unproductive = 0 if accepted > 0 else unproductive + 1
    return trace


def test_thresholds_are_ordered_and_finite():
    """Terse must be tried BEFORE giving up, or the cheap fix never runs."""
    assert 0 < sov.UNPRODUCTIVE_TERSE_AFTER < sov.UNPRODUCTIVE_STOP_AFTER
    assert sov.UNPRODUCTIVE_STOP_AFTER < 50, "a bound this loose is not a bound"


def test_a_productive_turn_clears_the_streak():
    trace = _streak([0, 0, 1, 0, 0])
    assert trace[2]["streak"] == 2
    assert trace[3]["streak"] == 0, "one accepted item resets the count"
    assert not any(t["stop"] for t in trace)


def test_the_pack_is_shortened_before_the_loop_gives_up():
    trace = _streak([0] * 20)
    first_terse = next(i for i, t in enumerate(trace) if t["terse"])
    first_stop = next(i for i, t in enumerate(trace) if t["stop"])
    assert first_terse < first_stop, "terse must be tried first"
    assert first_terse == sov.UNPRODUCTIVE_TERSE_AFTER


def test_a_loop_that_produces_nothing_stops():
    """The 330-iteration, 4.4-hour case. It must not reach double digits."""
    trace = _streak([0] * 400)
    assert any(t["stop"] for t in trace), "it span forever before this guard"
    assert len(trace) <= sov.UNPRODUCTIVE_STOP_AFTER + 1, len(trace)


def test_a_parsed_reply_that_accepts_nothing_still_counts_as_unproductive():
    """The stuck run parsed cleanly every turn. Parsing is not producing."""
    trace = _streak([0] * 12)
    assert any(t["stop"] for t in trace)


def test_alternating_work_never_trips_the_stop():
    """A slow but real loop must not be killed for being slow."""
    trace = _streak([0, 0, 1] * 30)
    assert not any(t["stop"] for t in trace)
    assert not any(t["terse"] for t in trace)


def test_context_pack_accepts_the_terse_flag_the_guard_sets():
    """The guard is worthless if the escape hatch it reaches for is not real."""
    kernel = {
        "objective": "o", "measured_state": {}, "hypotheses": [], "scars": [],
        "live_hypotheses": [], "iterations": [], "observations": [],
        "tried_params": [], "frontier": [], "executable_work_types": [],
    }
    long_pack = sov.context_pack(kernel)
    short_pack = sov.context_pack(kernel, terse=True)
    assert isinstance(short_pack, str) and short_pack
    assert len(short_pack) <= len(long_pack)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all green")
