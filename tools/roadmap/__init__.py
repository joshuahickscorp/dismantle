"""Roadmap IR and adversarial capability auditor.

Turns H-ROADMAP.md (Appendix O gates + Appendix A genes) into a machine-readable
graph, then fills every status from repo evidence: git-tracked definitions,
non-test call sites of the implementing symbol (via
tools.future.capability_reachability), tests, receipts as citations only, and
hardware presence probes.

A definition is not a capability. A module import is not a call. A receipt is
not evidence that anything runs.
"""
from __future__ import annotations

SCHEMA = "hawking.roadmap.capability_graph.v1"
VERSION = 2
ALLOWED_STATUSES = (
    "BUILT",
    "SCAFFOLDED",
    "ABSENT",
    "BLOCKED_HARDWARE",
    "BLOCKED_EXTERNAL",
    "DORMANT",
    "UNREACHABLE",
)
EVIDENCE_TIER = "STATIC"
GRAPH_REL = "civilization/CAPABILITY_GRAPH.json"
