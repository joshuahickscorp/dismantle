#!/usr/bin/env python3.12
"""Shim — implementation archived; lifecycle authority is tools.condense.engine.

Archived body: tools/condense/archive/glm52_rate_ladder.py
Campaign specs: tools/condense/engine/specs/

Archived source is exec'd into this module namespace so monkeypatches hit the
same globals the functions use. ``__file__`` points at the archived body.
"""
from __future__ import annotations

from pathlib import Path as _PathForShim

__shim_file__ = str(_PathForShim(__file__).resolve())
_ARCHIVE_PATH = _PathForShim(__shim_file__).parent / "archive" / "glm52_rate_ladder.py"
__file__ = str(_ARCHIVE_PATH.resolve())

_src = _ARCHIVE_PATH.read_text(encoding="utf-8")
exec(compile(_src, __file__, "exec"), globals())
del _src, _ARCHIVE_PATH, _PathForShim, __shim_file__
