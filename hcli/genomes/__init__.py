"""Genomes: learned persistent machine / model / runtime science.

``MachineGenome`` is a compatibility bag in ``hcli.machine``. Admission
numbers come from ``resolve_runtime_limits``; the producer is
``tools/headless/machine_probe.py``. This package is the ownership
surface, not a second genome authority.
"""
from hcli.machine import (
    GenomeFreshness,
    GenomeStale,
    MachineGenome,
    host_snapshot,
)

__all__ = [
    "GenomeFreshness",
    "GenomeStale",
    "MachineGenome",
    "host_snapshot",
]
