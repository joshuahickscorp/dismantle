"""Genomes: learned persistent machine / model / runtime science.

``MachineGenome`` is implemented in ``hcli.machine`` (runtime identity
lives next to the probe that produces it). This package is the ownership
surface.
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
