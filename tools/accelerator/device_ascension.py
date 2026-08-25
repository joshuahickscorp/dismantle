"""Device Ascension — the per-machine campaign. FRONT G (G049, steer S015 §23).

Eleven stages from fingerprint to ADP seal. The stages are real functions rather
than a narrative, and a stage that has not run reports NOT_RUN rather than being
skipped silently.

The machine-specific win this looks for is the one the steer points at in §125:
APPLE UNIFIED MEMORY IS A WEAPON -- what CUDA-era copy can disappear? A CUDA port
allocates device memory, copies host to device, computes, and copies back. On this
architecture there is no separate device memory, so both copies are avoidable. That
win is machine-specific in the strict sense P6 requires: it does not exist on a
discrete-GPU machine at all, because there the copies are not optional.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

STAGES = ("fingerprint", "benchmark", "roof_measurement", "workload_census",
          "kernel_selection", "autotune", "memory_plan", "dispatch_plan",
          "concurrency_sweep", "sustained_qualification", "adp_seal")

PROFILES = ("INTERACTIVE", "MAX_THROUGHPUT", "MIN_LATENCY", "MIN_MEMORY",
            "SUSTAINED", "BATTERY_AWARE", "HCLI_AUTONOMOUS", "ODYSSEY_RESEARCH")

KNOWLEDGE_LEVELS = ("EXACT_MACHINE", "SOC_FAMILY", "APPLE_GENERAL")


@dataclass
class Ascension:
    machine: dict[str, Any]
    stages: dict[str, Any] = field(default_factory=dict)

    def record(self, stage: str, payload: Any) -> None:
        if stage not in STAGES:
            raise ValueError(f"{stage!r} is not one of the eleven stages")
        self.stages[stage] = payload

    def not_run(self, stage: str, reason: str) -> None:
        self.record(stage, {"status": "NOT_RUN", "reason": reason})

    def completed(self) -> list[str]:
        return [s for s in STAGES
                if s in self.stages
                and not (isinstance(self.stages[s], dict)
                         and self.stages[s].get("status") == "NOT_RUN")]

    def may_seal_adp(self) -> tuple[bool, str]:
        """A production ADP needs sustained evidence, not a microbenchmark (§29)."""
        sust = self.stages.get("sustained_qualification")
        if not isinstance(sust, dict) or sust.get("status") == "NOT_RUN":
            return False, ("sustained_qualification has not run; §29 forbids sealing a "
                           "production ADP on microbenchmark evidence")
        if not sust.get("passed"):
            return False, "sustained_qualification did not pass"
        return True, "sustained evidence present"

    def seal(self, profile: str, config: dict[str, Any],
             knowledge_level: str) -> dict[str, Any]:
        if profile not in PROFILES:
            raise ValueError(f"{profile!r} is not a named profile; §24 forbids vague "
                             f"'optimized settings'")
        if knowledge_level not in KNOWLEDGE_LEVELS:
            raise ValueError(f"{knowledge_level!r} is not a knowledge level")
        ok, why = self.may_seal_adp()
        return {
            "adp": f"ADP-{self.machine.get('soc','UNKNOWN').replace(' ','')}-{profile}",
            "profile": profile,
            "config": config,
            "knowledge_level": knowledge_level,
            "production_sealed": ok,
            "status": "SEALED" if ok else "PROVISIONAL",
            "why": why,
            "stages_completed": self.completed(),
            "stages_not_run": [s for s in STAGES if s not in self.completed()],
            "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
