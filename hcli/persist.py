"""Leaf persistence helpers.

``atomic_write_json`` used to live in ``dag_store``, which pulled
``max_policy`` into an import SCC (dag_store -> workunit -> resources
-> max_policy -> dag_store). This module has no hcli imports.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Union


def atomic_write_json(path: Union[str, Path], obj: Any) -> None:
    """Write JSON via a same-directory temp file and ``os.replace``."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True)
    tmp_name = f".{dest.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    tmp_path = dest.parent / tmp_name
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, dest)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
