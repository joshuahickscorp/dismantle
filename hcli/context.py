"""Worker context packets.

The old ``Context`` class stored the full root goal and is gone. The only
worker-context type is ``WorkerPacket`` in ``goal.py``, next to the compiler
that produces it. This module re-exports that type so callers have one
name to import and we do not keep a second packet shape.
"""
from __future__ import annotations

from .goal import WorkerPacket, compile_worker_context

__all__ = ["WorkerPacket", "compile_worker_context"]
