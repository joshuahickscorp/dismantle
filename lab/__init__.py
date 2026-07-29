"""Hawking laboratory — Core C experiment engine.

Process lifecycle for campaigns: experiment specs, run/resume, receipts, and
governance. Live science bodies remain under tools/condense/; this package is
the engine only and does not ship a second science implementation.
"""
from __future__ import annotations

__version__ = "1.0.0"
AUTHORITY = "lab"
SCHEMA = "hawking.lab.core.v1"

from lab.runtime import ExperimentRuntime, RunResult, run_experiment
from lab.receipts import Receipt, ReceiptAuthority, seal, verify
from lab.spec import ExperimentSpec, SpecError, load_spec, validate_spec

__all__ = [
    "AUTHORITY",
    "ExperimentRuntime",
    "ExperimentSpec",
    "Receipt",
    "ReceiptAuthority",
    "RunResult",
    "SCHEMA",
    "SpecError",
    "load_spec",
    "run_experiment",
    "seal",
    "validate_spec",
    "verify",
]
