"""Sparse-checkout collection guard.

``pytest tools/future`` imports every test module before ``-k`` runs.
Two modules fail at import when receipts/ or hcli/ are not materialized.
Ignore them only while those paths are absent so the noetic/ebpw filter
can run; they collect again when the tree is widened.
"""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001
    name = getattr(collection_path, "name", None) or Path(str(collection_path)).name
    if name == "test_path_to_71.py":
        a = _REPO / "receipts/future/SEALED_DEFAULT_ABSOLUTE.json"
        b = _REPO / "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"
        if not a.is_file() and not b.is_file():
            return True
    if name == "test_status_causality_gates.py":
        if not (_REPO / "hcli/agentos/resident_gate.py").is_file():
            return True
    return None
