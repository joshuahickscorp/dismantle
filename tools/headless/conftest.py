"""Collection rules for tools/headless under a sparse checkout.

`python3 -m pytest tools/headless` collects every `*_test.py`. Several of
those import `tools/haider/hcli` at module level. That tree is often not
materialized here; an ImportError during collection would fail the session
before the closure harness ran. Skip those files only when haider is absent.
"""
from __future__ import annotations

from pathlib import Path

_HAIDER = Path(__file__).resolve().parents[1] / "haider"
_HAIDER_COLLECT = frozenset(
    {
        "rollback_integrity_test.py",
    }
)


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001
    path = Path(collection_path)
    if path.suffix != ".py":
        return None
    if _HAIDER.is_dir():
        return None
    name = path.name
    if name.startswith("hcli_") and name.endswith("_test.py"):
        return True
    if name in _HAIDER_COLLECT:
        return True
    return None
