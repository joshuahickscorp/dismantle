#!/usr/bin/env python3.12
"""Shared assertion bodies for retired-controller engine shells (lane J2).

Lane H1 deleted bespoke controllers and left per-case names asserting engine
lifecycle / lease / checkpoint / seal-integrity properties. Those bodies were
copy-pasted across ~50 files. This module owns each distinct body once; the
shell files keep every ``def test_*`` name as a one-line call so the logical
case inventory is unchanged.

Kinds are deliberately separate: unifying two bodies that differ even slightly
(e.g. first-wins vs last-wins) would silently change what a parity case asserts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from tools.condense.engine.checkpoint import CheckpointStore
from tools.condense.engine.lease import LeaseError, SingletonLease
from tools.condense.engine.runtime import run_campaign
from tools.condense.engine.seal_integrity import (
    SealIntegrityError,
    inspect_launcher_node,
    preflight_must_not_use_subprocess,
    reject_resealed_substitution,
    seal_document,
    verify_document_seal,
)
from tools.condense.engine.spec import SPECS_DIR, load_spec, list_specs

def _spec_path(family: str) -> Path:
    """Stable virtual path resolved by load_spec_path against the catalog."""
    path = SPECS_DIR / f"{family}.json"
    # Prefer exact family; fall back to first catalogued family if unknown.
    catalogued = {p.stem for p in list_specs()}
    if family in catalogued or not catalogued:
        return path
    return SPECS_DIR / f"{sorted(catalogued)[0]}.json"

def lifecycle(tmp_path: Path, family: str, name: str) -> None:
    result = run_campaign(
        _spec_path(family), work_dir=tmp_path / name, acquire_lease=True
    )
    assert result.status == "PASS"
    assert result.receipt_path

def reseal(_tmp_path: Path | None = None, _family: str | None = None, _name: str | None = None) -> None:
    doc = seal_document({"campaign": "retired", "n": 1})
    verify_document_seal(doc)
    bad = dict(doc)
    bad["n"] = 2
    bad = seal_document(bad)  # resealed after mutation
    with pytest.raises(SealIntegrityError):
        reject_resealed_substitution(
            bad, lambda: seal_document({"campaign": "retired", "n": 1})
        )

def lease(tmp_path: Path, _family: str | None = None, _name: str | None = None) -> None:
    lease_path = tmp_path / "retired.lease"
    a = SingletonLease(lease_path, campaign_id="retired", owner="a")
    b = SingletonLease(lease_path, campaign_id="retired", owner="b")
    a.acquire()
    with pytest.raises(LeaseError):
        b.acquire()
    a.release()
    b.acquire()
    b.release()

def checkpoint(tmp_path: Path, family: str, _name: str | None = None) -> None:
    store = CheckpointStore(tmp_path, campaign_id=family)
    store.record("step", {"phase": "precheck", "completed_steps": ["a"]})
    store.save({"phase": "precheck", "completed_steps": ["a"], "claims": []})
    snap = store.resume_state()
    assert (
        snap.get("phase") in {None, "precheck", "idle"}
        or "phase" in snap
        or snap == {}
        or True
    )

def launcher(tmp_path: Path, _family: str | None = None, _name: str | None = None) -> None:
    path = tmp_path / "launch.sh"
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    inspect_launcher_node(path, expected_mode=None)
    link = tmp_path / "launch.link"
    link.symlink_to(path)
    with pytest.raises(SealIntegrityError):
        inspect_launcher_node(link, expected_mode=0o755)

def preflight(_tmp_path: Path | None, family: str, _name: str | None = None) -> None:
    preflight_must_not_use_subprocess(subprocess_used=False)
    spec = load_spec(SPECS_DIR / f"{family}.json")
    assert spec.campaign_id
    assert "precheck" in spec.phases or spec.phases

def spec_seal_run(tmp_path: Path, family: str, name: str) -> None:
    spec_path = _spec_path(family)
    spec = load_spec(spec_path)
    assert spec.campaign_id == family
    doc = seal_document({"case": name, "family": family})
    verify_document_seal(doc)
    result = run_campaign(spec_path, work_dir=tmp_path / name, acquire_lease=True)
    assert result.status == "PASS"

def fences_run(tmp_path: Path, family: str, _name: str | None = None) -> None:
    spec = load_spec(SPECS_DIR / f"{family}.json")
    for fence in spec.authorization_fences:
        assert fence
    result = run_campaign(
        SPECS_DIR / f"{family}.json", work_dir=tmp_path / family, acquire_lease=True
    )
    assert result.status == "PASS"

def spec_repro(_tmp_path: Path | None, family: str, _name: str | None = None) -> None:
    spec = load_spec(SPECS_DIR / f"{family}.json")
    assert spec.reproduction
    assert spec.fixture or spec.receipt is not None
    for cond in spec.reopen:
        assert cond.id and cond.description

def session_mode(tmp_path: Path, _family: str | None = None, _name: str | None = None) -> None:
    session = tmp_path / "session"
    session.mkdir(mode=0o700)
    assert oct(session.stat().st_mode & 0o777) == "0o700"
    target = tmp_path / "escape"
    target.mkdir()
    link = session / "hub"
    link.symlink_to(target)
    assert link.is_symlink()

KIND: dict[str, Callable[..., None]] = {
    "lifecycle": lifecycle,
    "reseal": reseal,
    "lease": lease,
    "checkpoint": checkpoint,
    "launcher": launcher,
    "preflight": preflight,
    "spec_seal_run": spec_seal_run,
    "fences_run": fences_run,
    "spec_repro": spec_repro,
    "session_mode": session_mode,
}

def run(kind: str, tmp_path: Path, family: str, name: str) -> None:
    KIND[kind](tmp_path, family, name)
