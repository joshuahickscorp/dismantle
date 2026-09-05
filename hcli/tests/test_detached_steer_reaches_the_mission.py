"""A steer injected from a DETACHED process must reach the running mission.

The Odyssey contract treats detached supervision as normal: Claude attaches,
injects a steer, detaches, and HCLI continues. Measured 2026-09-05, before this
fix:

    mission session_id                    3f254d6d-...
    the steer landed in                   c22b3aa4-...   (the INJECTING session)
    file for the mission's session        did not exist

`SteeringQueue` is keyed by session id and the mission polls only its OWN file, so
the steer was orphaned. The CLI printed "✓ Steer queued" either way, which is the
worst version of this bug: a silent success.
"""
from __future__ import annotations

import json
from pathlib import Path

import hcli.controller as ctrl


def _write_mission(root: Path, mission_id: str, session_id: str, phase: str = "running"):
    d = root / ".hcli" / "mission"
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(
        {"id": mission_id, "session_id": session_id, "phase": phase}), encoding="utf-8")


def test_live_mission_identity_is_read_from_disk(tmp_path):
    _write_mission(tmp_path, "M-1", "S-1")
    got = ctrl._live_mission_identity(str(tmp_path))
    assert got == {"id": "M-1", "session_id": "S-1"}


def test_a_finished_mission_is_not_a_steer_target(tmp_path):
    """Negative control: only a LIVE mission may capture a detached steer."""
    _write_mission(tmp_path, "M-1", "S-1", phase="failed")
    assert ctrl._live_mission_identity(str(tmp_path)) is None


def test_no_mission_on_disk_is_not_an_error(tmp_path):
    assert ctrl._live_mission_identity(str(tmp_path)) is None


def test_a_detached_steer_lands_in_the_missions_queue_with_provenance(tmp_path):
    from hcli.steering import SteeringQueue

    _write_mission(tmp_path, "M-9", "S-9")
    live = ctrl._live_mission_identity(str(tmp_path))
    assert live is not None

    q = SteeringQueue(str(tmp_path), live["session_id"])
    q.enqueue("look at the miss path", kind="knowledge",
              mission_id=live["id"], source_session_id="SUPERVISOR-1")

    target = tmp_path / ".hcli" / "steering" / "S-9.json"
    assert target.is_file(), (
        "the steer did not land in the MISSION's queue; the mission polls only its "
        "own session file, so anywhere else is an orphan"
    )
    events = json.loads(target.read_text(encoding="utf-8"))
    assert events[0]["mission_id"] == "M-9", "the steer does not name the mission it steers"
    assert events[0]["source_session_id"] == "SUPERVISOR-1", (
        "the steer does not record who injected it"
    )
