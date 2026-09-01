"""Resident-facing bridge into the live, mid-flight Odyssey-I driver.

Odyssey is real and running under ``tools/odyssey_ctl.py`` (~8k lines: queue,
patient packets, harvester, compiler-rule inference, run loop). O003 is
already SEALED. HCLI has no odyssey verb, so a resident cannot see or drive
any of it. This module is the connector, not a rewrite: every function here
shells out to the existing driver's own subcommands. It adds nothing to the
curriculum and encodes none of it.

Two capability classes:

* **inspect** -- ``status``, ``queue``, ``value``, ``economics``,
  ``completions()``, ``patient()``, ``admit_check()``. Read-only, run
  unconditionally, never write anything or spend a resource. ``admit_check``
  is named after the CLI's ``admit`` verb but that verb only prints a
  memgate/worker_gate/disk decision -- it never touches queue state -- so it
  is read-only despite the name.

* **continue** -- ``harvest``, ``write_packet``, ``run``, ``cycle``,
  ``retire``, ``acquire_next``, ``completions(rebuild=True)``. Every one of
  these can mutate persistent Odyssey state, spawn a subprocess, or start a
  download. Each requires ``confirm=True`` or raises ``PermissionError``,
  mirroring the gate ``benchmark.run``/``accelerator.benchmark`` already use
  in hcli/tool_registry.py (``if args.get("confirm") is not True: raise
  PermissionError(...)``). A ``dry_run=True`` path (the default, where the
  driver has one) needs no confirm because the driver itself launches
  nothing in that mode.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
ODYSSEY_CTL = REPO / "tools" / "odyssey_ctl.py"
PATIENTS_DIR = REPO / "workspace" / "campaign" / "odyssey" / "patients"


def _run(args: list[str], timeout_s: float = 60.0) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(ODYSSEY_CTL), *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return {
        "schema": "hcli.odyssey.ctl.v1",
        "argv": args,
        "exit_code": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _require_confirm(confirm: bool, what: str) -> None:
    if confirm is not True:
        raise PermissionError(f"{what} mutates Odyssey state and requires confirm=True")


# --------------------------------------------------------------------------
# inspect -- read-only, no gate
# --------------------------------------------------------------------------

def status() -> dict:
    """Queue + current patient + compiler rules + research counters, in one shot."""
    return _run(["status"])


def queue() -> dict:
    return _run(["queue"])


def value() -> dict:
    """Ranked NEXT-work list with info-value proxies."""
    return _run(["value"])


def economics() -> dict:
    return _run(["economics"])


def completions(rebuild: bool = False, confirm: bool = False, completed_at: Optional[str] = None) -> dict:
    """List recorded completions; ``rebuild=True`` writes a backfill and needs confirm."""
    if not rebuild:
        return _run(["completions"])
    _require_confirm(confirm, "completions --rebuild")
    args = ["completions", "--rebuild"]
    if completed_at:
        args += ["--completed-at", completed_at]
    return _run(args)


def patient(oxx: str) -> dict:
    """Read a patient packet already on disk. Never writes one (see write_packet)."""
    path = PATIENTS_DIR / oxx / f"ODYSSEY_PATIENT_{oxx}.json"
    if not path.is_file():
        return {"schema": "hcli.odyssey.patient.v1", "oxx": oxx, "found": False, "path": str(path)}
    return {
        "schema": "hcli.odyssey.patient.v1",
        "oxx": oxx,
        "found": True,
        "path": str(path),
        "packet": json.loads(path.read_text()),
    }


def admit_check(slug: str, est_gib: float) -> dict:
    """Memgate/worker_gate/disk-floor GO-or-REFUSE check. Prints only; writes nothing."""
    return _run(["admit", slug, str(est_gib)])


# --------------------------------------------------------------------------
# continue -- mutating, confirm=True required (dry-run paths excepted)
# --------------------------------------------------------------------------

def harvest(dry_run: bool = True, confirm: bool = False) -> dict:
    if dry_run:
        return _run(["harvest", "--dry-run"])
    _require_confirm(confirm, "harvest (non-dry-run)")
    return _run(["harvest"])


def write_packet(oxx: str, confirm: bool = False) -> dict:
    _require_confirm(confirm, f"packet {oxx} (writes a patient packet file)")
    return _run(["packet", oxx])


def run(confirm: bool = False, dry_run: bool = True, max_lanes: int = 2, grok_lanes: int = 0) -> dict:
    args = ["run", "--max-lanes", str(max_lanes), "--grok-lanes", str(grok_lanes)]
    if dry_run:
        return _run([*args, "--dry-run"])
    _require_confirm(confirm, "run --go (spawns grok-run lanes)")
    return _run([*args, "--go"], timeout_s=600.0)


def cycle(
    confirm: bool = False,
    dry_run: bool = True,
    max_lanes: int = 2,
    grok_lanes: int = 0,
    loop_secs: Optional[float] = None,
    inner_sleep: float = 3.0,
) -> dict:
    args = ["cycle", "--max-lanes", str(max_lanes), "--grok-lanes", str(grok_lanes)]
    if loop_secs is not None:
        args += ["--loop-secs", str(loop_secs), "--inner-sleep", str(inner_sleep)]
    if dry_run:
        return _run([*args, "--dry-run"])
    _require_confirm(confirm, "cycle --go (harvest/retire/acquire/launch for real)")
    return _run([*args, "--go"], timeout_s=(loop_secs or 60.0) + 120.0)


def retire(oxx: str, confirm: bool = False) -> dict:
    _require_confirm(confirm, f"retire {oxx} (removes a patient from the queue)")
    return _run(["retire", oxx])


def acquire_next(confirm: bool = False, dry_run: bool = True) -> dict:
    if dry_run:
        return _run(["acquire-next", "--dry-run"])
    _require_confirm(confirm, "acquire-next --go (starts a Hugging Face download)")
    return _run(["acquire-next", "--go"])
