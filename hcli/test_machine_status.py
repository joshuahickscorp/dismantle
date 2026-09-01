"""/status reports the machine, not just this session, and never guesses.

The defect: a fresh CLI session printed ``health=down resident=0`` on a host
that was running a resident supervisor and three ModelLake downloads, because
every field came from the session's own controller.

The rule these tests enforce is that each added value comes from a live
producer -- the resident's own ``state.json``, the ModelLake watcher's lock
file and JSONL -- and that a value which cannot be read prints ``unknown``
rather than a remembered number. In particular a pid that merely exists is
never enough to call a supervisor live.

Runnable two ways:

    python3 -m pytest hcli/test_machine_status.py -q
    python3 hcli/test_machine_status.py
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from hcli.commands import (
    _last_watcher_sample,
    _modellake_status,
    _pid_liveness,
    _resident_status,
    _status_roots,
    format_machine_status,
    format_status,
)

MAX_STATUS_LINE = 80


def fresh() -> Path:
    root = Path(tempfile.mkdtemp(prefix="machine_status_test_"))
    atexit.register(shutil.rmtree, root, True)
    return root


def write_resident(root: Path, **fields) -> Path:
    path = root / ".hcli" / "resident" / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "state": "RUNNING",
        "cycles": 42,
        "last_event": "worker_heartbeat",
        "updated_at": time.time(),
    }
    state.update(fields)
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


def write_watcher(root: Path, *, pid, rows) -> None:
    odyssey = root / "workspace" / "campaign" / "odyssey"
    (odyssey / "downloads").mkdir(parents=True, exist_ok=True)
    if pid is not None:
        (odyssey / ".modellake-watch.lock").write_text(str(pid), encoding="utf-8")
    log = odyssey / "downloads" / "modellake-watch.jsonl"
    log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


# -- liveness ---------------------------------------------------------------


def test_a_pid_alone_is_never_live():
    """os.getpid() is alive, but not with someone else's start token."""
    assert _pid_liveness(os.getpid(), None) == "live"
    assert _pid_liveness(os.getpid(), "not-the-token-this-process-has") == "dead"


def test_absent_and_impossible_pids_are_not_alive():
    assert _pid_liveness(None, None) == "none"
    assert _pid_liveness(0, None) == "none"
    assert _pid_liveness("nonsense", None) == "none"
    # 2**22 is above every default pid_max; nothing can be running there.
    assert _pid_liveness(2**22, None) == "dead"


# -- resident ---------------------------------------------------------------


def test_no_resident_record_reports_absent_not_healthy():
    root = fresh()
    assert _resident_status([root]) is None
    assert format_machine_status({"resident": None, "modellake": None}) == [
        "Machine resident=absent modellake=absent"
    ]


def test_resident_fields_come_from_the_state_file():
    root = fresh()
    write_resident(root, cycles=7, last_event="mission_idle", state="IDLE")
    snap = _resident_status([root])
    assert snap["state"] == "IDLE"
    assert snap["cycles"] == 7
    assert snap["last_event"] == "mission_idle"
    assert snap["age_s"] < 60


def test_a_running_record_with_a_dead_pid_is_not_reported_live():
    """The exact 'never healthy merely because a PID exists' case."""
    root = fresh()
    write_resident(root, state="RUNNING", supervisor_pid=2**22)
    snap = _resident_status([root])
    assert snap["state"] == "RUNNING", "the record is left as written"
    assert snap["supervisor"] == "dead", "the process table is what decides"
    line = format_machine_status({"resident": snap})[0]
    assert "RUNNING" in line and "supervisor=dead" in line


def test_a_recycled_pid_does_not_inherit_the_record():
    root = fresh()
    write_resident(
        root, supervisor_pid=os.getpid(), supervisor_start_token="a-previous-boot"
    )
    assert _resident_status([root])["supervisor"] == "dead"


def test_a_stale_record_still_shows_its_age():
    root = fresh()
    write_resident(root, updated_at=time.time() - 7200)
    snap = _resident_status([root])
    assert 7000 < snap["age_s"] < 8000
    assert "age=7" in format_machine_status({"resident": snap})[0]


def test_unreadable_state_prints_unknown_not_zero():
    root = fresh()
    write_resident(root, cycles=None, last_event=None, state=None)
    line = format_machine_status({"resident": _resident_status([root])})[0]
    assert "cycles=unknown" in line and "event=unknown" in line
    assert "cycles=0" not in line


def test_a_corrupt_state_file_is_skipped_not_guessed():
    root = fresh()
    path = root / ".hcli" / "resident" / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert _resident_status([root]) is None


# -- modellake --------------------------------------------------------------


def test_modellake_counts_the_watchers_own_last_sample():
    root = fresh()
    write_watcher(
        root,
        pid=os.getpid(),
        rows=[
            {"event": "watcher_sample", "active_jobs": ["old"], "ts": "2020-01-01T00:00:00+00:00"},
            {"event": "network_sample", "rx_bytes": 1},
            {
                "event": "watcher_sample",
                "active_jobs": ["a", "b", "c"],
                "active_remaining_bytes": 3 * 1024**3,
                "ts": "2020-01-02T00:00:00+00:00",
            },
            {"event": "network_sample", "rx_bytes": 2},
        ],
    )
    snap = _modellake_status([root])
    assert snap["jobs"] == 3, "the newest watcher_sample wins, not the newest row"
    assert snap["watcher"] == "live"
    assert snap["sample_age_s"] > 0
    line = format_machine_status({"modellake": snap})[0]
    assert "jobs=3" in line and "remaining=3.0GB" in line


def test_a_dead_watcher_is_not_dressed_up_as_activity():
    root = fresh()
    write_watcher(
        root,
        pid=2**22,
        rows=[{"event": "watcher_sample", "active_jobs": ["a"], "ts": "2020-01-01T00:00:00+00:00"}],
    )
    snap = _modellake_status([root])
    assert snap["watcher"] == "dead"
    assert snap["jobs"] == 1, "its last observation is still reported, with its age"
    assert snap["sample_age_s"] > 86400


def test_a_log_with_no_sample_yet_reports_unknown():
    root = fresh()
    write_watcher(root, pid=None, rows=[{"event": "network_sample", "rx_bytes": 1}])
    snap = _modellake_status([root])
    assert snap["jobs"] is None and snap["remaining_bytes"] is None
    line = format_machine_status({"modellake": snap})[0]
    assert "jobs=unknown" in line and "remaining=unknown" in line
    assert "jobs=0" not in line


def test_only_a_bounded_tail_of_the_log_is_read():
    """The live log is 580 MB. Reading it whole would stall every /status."""
    root = fresh()
    filler = [{"event": "network_sample", "rx_bytes": n, "pad": "x" * 512} for n in range(4000)]
    write_watcher(
        root,
        pid=None,
        rows=[{"event": "watcher_sample", "active_jobs": ["buried"], "ts": "2020-01-01T00:00:00+00:00"}]
        + filler
        + [{"event": "watcher_sample", "active_jobs": ["recent"], "ts": "2020-01-02T00:00:00+00:00"}],
    )
    log = root / "workspace" / "campaign" / "odyssey" / "downloads" / "modellake-watch.jsonl"
    assert log.stat().st_size > 2 * 1024 * 1024, "the fixture must exceed the tail"
    assert _last_watcher_sample(log)["active_jobs"] == ["recent"]


def test_a_truncated_first_row_in_the_tail_is_skipped():
    root = fresh()
    odyssey = root / "workspace" / "campaign" / "odyssey" / "downloads"
    odyssey.mkdir(parents=True, exist_ok=True)
    log = odyssey / "modellake-watch.jsonl"
    good = json.dumps({"event": "watcher_sample", "active_jobs": ["ok"]})
    log.write_text('{"event": "watcher_sample", "active_jo\n' + good + "\n", encoding="utf-8")
    assert _last_watcher_sample(log)["active_jobs"] == ["ok"]


# -- the rendered screen ----------------------------------------------------


def test_machine_lines_join_the_one_screen_status():
    root = fresh()
    write_resident(root, supervisor_pid=2**22)
    write_watcher(
        root,
        pid=os.getpid(),
        rows=[{"event": "watcher_sample", "active_jobs": ["a"], "ts": "2020-01-01T00:00:00+00:00"}],
    )
    snap = {
        "mission_id": "m-1",
        "phase": "running",
        "goal": "ship it",
        "resident": _resident_status([root]),
        "modellake": _modellake_status([root]),
    }
    text = format_status(snap)
    assert "Resident RUNNING" in text
    assert "ModelLake watcher=live" in text
    # The house limits on /status: one screen, and no wrapped lines.
    lines = text.splitlines()
    assert len(lines) <= 10, lines
    for line in lines:
        assert len(line) <= MAX_STATUS_LINE, (len(line), line)


def test_a_long_event_name_cannot_widen_the_line():
    root = fresh()
    write_resident(
        root,
        state="STARTING",
        cycles=1234567,
        last_event="evacuation_checkpoint_error",
        supervisor_pid=2**22,
        updated_at=time.time() - 999999,
    )
    line = format_machine_status({"resident": _resident_status([root])})[0]
    assert len(line) <= MAX_STATUS_LINE, (len(line), line)


def test_status_roots_fall_back_to_the_repo():
    """A session opened in a scratch directory must still see the machine."""

    class Elsewhere:
        workspace_root = tempfile.gettempdir()

    roots = _status_roots(Elsewhere())
    assert len(roots) == 2 and roots[0] != roots[1]
    assert (roots[1] / "hcli" / "commands.py").is_file(), roots


if __name__ == "__main__":
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all green")
