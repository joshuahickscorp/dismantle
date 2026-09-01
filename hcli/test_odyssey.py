"""The resident can see and drive the live Odyssey mission through HCLI.

Odyssey (tools/odyssey_ctl.py) is real and mid-flight -- O003 sealed, O006 on
disk, O010-O013 queued -- but HCLI has no odyssey verb at all. hcli/odyssey.py
is the connector. This locks down two things:

1. inspection ingests the *existing* state (no fixture, no re-run) and never
   writes anything -- proven by hashing a patient packet before and after.
2. every verb that can mutate Odyssey state, spawn a subprocess, or start a
   download refuses with PermissionError unless confirm=True, the same gate
   hcli/tool_registry.py already uses for benchmark.run/accelerator.benchmark
   -- and it refuses *before* touching the driver at all (no subprocess is
   spawned to reach the refusal).

Runnable two ways:

    python3 -m pytest hcli/test_odyssey.py -q
    python3 hcli/test_odyssey.py
"""
from __future__ import annotations

import hashlib
import subprocess

import pytest

from hcli import odyssey


def test_status_ingests_real_mid_flight_state():
    """No fixture: this must reflect the sealed patient actually on disk."""
    result = odyssey.status()
    assert result["ok"] is True, result
    assert "HAWKING ODYSSEY-I" in result["stdout"]
    assert "O003" in result["stdout"]
    assert "SEALED" in result["stdout"]


def test_queue_value_economics_are_read_only_and_succeed():
    for call in (odyssey.queue, odyssey.value, odyssey.economics):
        result = call()
        assert result["ok"] is True, (call.__name__, result)


def test_patient_reads_without_writing():
    """Named 'inspect', so a patient read must not perturb the file it reads."""
    before = odyssey.patient("O003")
    assert before["found"] is True, before
    digest_before = hashlib.sha256(open(before["path"], "rb").read()).hexdigest()

    odyssey.patient("O003")  # a second read must not mutate anything either

    digest_after = hashlib.sha256(open(before["path"], "rb").read()).hexdigest()
    assert digest_before == digest_after


def test_patient_reports_absence_for_unknown_oxx_without_writing():
    result = odyssey.patient("O999")
    assert result["found"] is False
    assert result["oxx"] == "O999"


def test_admit_check_is_read_only_despite_the_verb_name():
    """cmd_admit only prints a memgate/worker_gate/disk decision; it never
    writes ODYSSEY_STATE.json, so it belongs with the inspect verbs."""
    result = odyssey.admit_check("hcli-odyssey-probe", 1.0)
    assert result["exit_code"] in (0, 1), result
    assert "slug=hcli-odyssey-probe" in result["stdout"]


def test_dry_run_paths_execute_the_real_driver_and_launch_nothing():
    """dry_run=True needs no confirm because the driver itself launches
    nothing in that mode -- assert the driver's own words say so."""
    result = odyssey.harvest(dry_run=True)
    assert result["ok"] is True, result
    assert "mode=dry-run" in result["stdout"]

    result = odyssey.acquire_next(dry_run=True)
    assert result["ok"] is True, result
    assert "DRY-RUN" in result["stdout"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: odyssey.harvest(dry_run=False),
        lambda: odyssey.write_packet("O003"),
        lambda: odyssey.run(dry_run=False),
        lambda: odyssey.cycle(dry_run=False),
        lambda: odyssey.retire("O013"),
        lambda: odyssey.acquire_next(dry_run=False),
        lambda: odyssey.completions(rebuild=True),
    ],
    ids=["harvest", "write_packet", "run", "cycle", "retire", "acquire_next", "completions_rebuild"],
)
def test_mutating_verbs_refuse_without_confirm(call, monkeypatch):
    """Refusal must happen before the driver is ever invoked -- patch
    subprocess.run to explode if a mutating verb reaches it unconfirmed."""

    def _boom(*args, **kwargs):
        raise AssertionError("mutating verb spawned the driver without confirm=True")

    monkeypatch.setattr(subprocess, "run", _boom)
    with pytest.raises(PermissionError):
        call()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
