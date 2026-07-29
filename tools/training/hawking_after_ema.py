#!/usr/bin/env python3
"""lab_harness archive shim — body at tools/training/archive/hawking_after_ema.py."""
from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path

def _repo() -> Path:
    p = Path(__file__).resolve()
    for _ in range(10):
        if (p / "tools" / "foundry" / "lab_harness").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parents[min(4, len(Path(__file__).resolve().parents) - 1)]

_TARGET = _repo() / 'tools/training/archive/hawking_after_ema.py'

def _load():
    name = Path(__file__).stem
    spec = importlib.util.spec_from_file_location(name, _TARGET)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load archived module at {_TARGET}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.modules[__name__] = mod
    spec.loader.exec_module(mod)
    return mod

if __name__ == "__main__":
    sys.argv[0] = str(_TARGET)
    runpy.run_path(str(_TARGET), run_name="__main__")
else:
    _mod = _load()
    for k, v in vars(_mod).items():
        if k not in {"__name__", "__file__", "__package__", "__doc__"}:
            globals()[k] = v
