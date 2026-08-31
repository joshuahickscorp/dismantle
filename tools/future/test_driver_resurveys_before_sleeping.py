"""The 477-second idle must be structurally impossible, not merely detected.

G037: "the driver must consult refill before sleeping on open handles and must
justify any idle it does take; the judge needs a no_idle_while_work_exists
condition whose negative control is the archived 30m timeline itself."

The archived 30m run drained its queue at t=88 and did nothing until t=565, then
ended with next_work_left naming TWELVE frontiers that still held novel work.
eval_never_conversational_wait passed it because that evaluator matches
conversational phrases and event kinds and NEVER an idle interval.

These pin the DRIVER half. emit_idle_justified already refuses the four ways a
wait can be dishonest; what nothing pinned is that the wait loop RE-ASKS rather
than justifying once and sleeping through a frontier that later fills - which is
exactly the shape of a 477-second hole.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.future.autonomy_run import EmitRefused, emit_idle_justified  # noqa: E402


def _doc():
    return {"events": []}


def test_waiting_while_novel_work_exists_is_refused():
    """The core of the obligation: novel work means take it, never wait."""
    with pytest.raises(EmitRefused, match="refill returned novel work"):
        emit_idle_justified(
            _doc(),
            asked=[{"frontier_id": "F.A", "returned": "novel"}],
            waiting_on=[{"job_id": "j1", "pid": 1}],
            t_s=88,
        )


def test_a_wait_with_no_open_handle_is_an_end_not_a_wait():
    with pytest.raises(EmitRefused, match="not a wait; it is an end"):
        emit_idle_justified(_doc(), asked=[{"frontier_id": "F.A"}], waiting_on=[], t_s=88)


def test_idle_without_asking_the_frontier_is_refused():
    """A 477-second gap with no survey is the failure this exists to prevent."""
    with pytest.raises(EmitRefused, match="frontiers were not asked"):
        emit_idle_justified(_doc(), asked=None, waiting_on=[{"job_id": "j1"}], t_s=88)


def test_a_dry_survey_with_an_open_handle_is_a_legitimate_wait():
    doc = emit_idle_justified(
        _doc(),
        asked=[{"frontier_id": "F.A"}, {"frontier_id": "F.B"}],
        waiting_on=[{"job_id": "j1", "pid": 4242}],
        t_s=88,
    )
    ev = doc["events"][-1]
    payload = ev.get("payload") or ev
    assert payload["n_novel"] == 0
    assert payload["n_asked"] == 2
    assert payload["frontiers_asked"] == ["F.A", "F.B"]
    assert payload["waiting_on"][0]["pid"] == 4242


def test_the_survey_is_carried_not_summarised():
    """next_work_left named twelve frontiers holding novel work and the judge
    could not see them. The justification must carry the frontiers it asked, by
    id, so a later reader can check the claim rather than trust it."""
    asked = [{"frontier_id": f"F.{i}"} for i in range(12)]
    doc = emit_idle_justified(
        _doc(), asked=asked, waiting_on=[{"job_id": "j1"}], t_s=88
    )
    payload = doc["events"][-1].get("payload") or doc["events"][-1]
    assert payload["frontiers_asked"] == [f"F.{i}" for i in range(12)]
    assert len(payload["returned"]) == 12


def test_one_novel_frontier_among_many_still_refuses():
    """Eleven dry frontiers do not license sleeping through the twelfth."""
    asked = [{"frontier_id": f"F.{i}"} for i in range(11)]
    asked.append({"frontier_id": "F.11", "returned": "novel"})
    with pytest.raises(EmitRefused, match="refill returned novel work"):
        emit_idle_justified(_doc(), asked=asked, waiting_on=[{"job_id": "j1"}], t_s=88)
