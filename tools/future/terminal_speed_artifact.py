#!/usr/bin/env python3
"""The terminal speed artifact, and the reasons it may not be written yet.

S022 §66/§68 (G066) asks for exactly one of two receipts:

    RESIDENT_71TPS_UNLOCK.json          -- 71 TPS reached, with repeatability,
                                           capability, zero fallbacks, source
                                           identity, clean build, stable memory
                                           and restart

    MAX_RESIDENT_PHYSICAL_ROOF.json     -- the binding limit PROVEN, naming the
                                           dominant remaining costs, the
                                           irreducible current information, the
                                           best representation and its physical
                                           evidence, the next hardware
                                           requirement, and the next model-body
                                           alternative

and it adds the sentence that makes this module necessary: "Probably impossible"
is not an acceptable output; a proof of the binding limit is. The acceptance
adds that EVERY NUMBER must be traceable to a landed measurement receipt.

Those two failure modes are opposite and both are easy. A premature roof receipt
declares a limit while three measurements are outstanding, which is the same
error as declaring victory. So this module does not write prose - it holds the
prerequisites as data, resolves every number it would quote from the receipt
that owns it, and REFUSES to emit while a prerequisite is open. When the
outstanding measurements land, the artifact assembles itself from receipts
rather than being written by hand.

    python3 tools/future/terminal_speed_artifact.py            # status
    python3 tools/future/terminal_speed_artifact.py --build    # emit or refuse
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

import causal_budget_71 as cb  # noqa: E402

RECORDED_BY = "tools/future/terminal_speed_artifact.py"
UNLOCK_NAME = "RESIDENT_71TPS_UNLOCK.json"
ROOF_NAME = "MAX_RESIDENT_PHYSICAL_ROOF.json"

TARGET_TPS = 71.0
TARGET_MS = 1000.0 / TARGET_TPS  # 14.085 ms


class TerminalArtifactRefused(RuntimeError):
    """The terminal artifact cannot be written yet, and the reasons are named."""


# Each prerequisite is a MEASUREMENT that must land before the binding limit can
# be called proven rather than believed. A prerequisite is met when its receipt
# exists AND carries the field that makes it a measurement rather than a plan.
PREREQUISITES: tuple[dict[str, Any], ...] = (
    {
        "id": "G038_per_region_attribution",
        "receipt": "receipts/future/TOKEN_REGION_TIMESTAMPS.json",
        "field": ["attributed_ms", "unattributed_ms"],
        "why": (
            "0.321 ms of GPU time inside the decode step belongs to no organ "
            "(causal_budget_71.causal_residual). A roof receipt that names the "
            "dominant remaining costs cannot leave 1.1% of the token unattributed "
            "and call the accounting complete."
        ),
    },
    {
        "id": "G044_granularity_falsifier",
        "receipt": "receipts/future/MLP_GRANULARITY_FALSIFIER.json",
        "field": ["fused_region_gb_s"],
        "why": (
            "One representative MLP layer, contiguous, few fused regions, "
            "identical arithmetic, bit-identical output. Rising toward the LM "
            "head's 497.4 implicates fragmentation; staying near 350 kills it. "
            "Either answer changes what the binding limit IS."
        ),
    },
    {
        "id": "G075_current_body_baseline",
        "receipt": "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json",
        "field": ["decode_wall_ms_per_token"],
        "why": (
            "deltanet_widen_f4 landed as a measured token-identical 1.0245 ms "
            "win, so every ladder rung is arithmetic over a body that no longer "
            "runs. A roof measured against a superseded baseline is not a roof."
        ),
    },
)


def _resolved(rel: str, field: list[str]) -> Any:
    path = REPO / rel
    if not path.exists():
        return None
    cur: Any = json.loads(path.read_text())
    for key in field:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def prerequisite_status() -> list[dict[str, Any]]:
    rows = []
    for pre in PREREQUISITES:
        got = _resolved(pre["receipt"], list(pre["field"]))
        rows.append(
            {
                "id": pre["id"],
                "receipt": pre["receipt"],
                "field": ".".join(pre["field"]),
                "met": got is not None,
                "value": got,
                "why": pre["why"],
            }
        )
    return rows


def reached_71() -> dict[str, Any]:
    """Is the target reached on the CURRENT body? Read, never assumed."""
    residual = cb.causal_residual()
    wall_ms = float(residual["wall_ms"])
    current = residual["baseline_is_stale"]["current_body_ms"]
    return {
        "target_tps": TARGET_TPS,
        "target_ms": round(TARGET_MS, 3),
        "measured_wall_ms": wall_ms,
        "measured_tps": round(1000.0 / wall_ms, 2),
        "current_body_ms": current,
        "reached": False if isinstance(current, str) else bool(float(current) <= TARGET_MS),
        "why_not_assumed": (
            "the baseline this reads is the last MEASURED complete token; while "
            "current_body_ms is UNKNOWN the target cannot be claimed reached, and "
            "it cannot be claimed unreachable either"
        ),
    }


def which_receipt() -> dict[str, Any]:
    """UNLOCK, ROOF, or NEITHER_YET - and never 'probably impossible'."""
    hit = reached_71()
    if hit["reached"]:
        return {"emit": UNLOCK_NAME, "why": "the target is reached on a measured body"}
    open_pre = [r for r in prerequisite_status() if not r["met"]]
    if open_pre:
        return {
            "emit": None,
            "why": (
                "the binding limit is not yet PROVEN: "
                f"{len(open_pre)} of {len(PREREQUISITES)} measurements are open"
            ),
            "open": [r["id"] for r in open_pre],
        }
    return {"emit": ROOF_NAME, "why": "every named measurement has landed; the limit can be proven"}


def build() -> dict[str, Any]:
    verdict = which_receipt()
    if verdict["emit"] is None:
        raise TerminalArtifactRefused(
            "refusing to write the terminal artifact: "
            + verdict["why"]
            + "; open="
            + ", ".join(verdict.get("open") or [])
            + ". A roof declared while its own measurements are outstanding is "
            "the same error as declaring victory, and 'probably impossible' is "
            "explicitly not an acceptable output."
        )
    return {
        "schema": "hawking.future.terminal_speed_artifact.v1",
        "version": 1,
        "recorded_by": RECORDED_BY,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "which": verdict["emit"],
        "target": reached_71(),
        "prerequisites": prerequisite_status(),
        "citations_resolved": cb.resolve_all(),
        "causal_residual": cb.causal_residual(),
    }


def status() -> dict[str, Any]:
    return {
        "which_receipt": which_receipt(),
        "target": reached_71(),
        "prerequisites": prerequisite_status(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    if not args.build:
        print(json.dumps(status(), indent=1, sort_keys=True))
        return 0
    doc = build()
    print(write_receipt(doc["which"], doc, RECORDED_BY))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
