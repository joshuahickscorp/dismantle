"""Theia bounty-engine execution path.

Half 1 (this module) is live: ingest a local Hawking receipt as a self-bounty,
run H.2, score with H.1. Half 2 (the model ladder) is BLOCKED_EXTERNAL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tools.future._common import REPO, write_receipt
from tools.theia.intake import IntakeResult, run_intake
from tools.theia.ladder import STAGES, evaluate_wake
from tools.theia.self_bounty import bounty_from_receipt, value_inputs_from_receipt

DEFINITION = (
    "Theia is Hawking's locally runnable generalist model trained to chase "
    "verified intellectual bounties: monetary or non-monetary problems where "
    "success can be grounded in artifacts, tests, proofs, measurements, "
    "reproductions or authorized program rules."
)
RECORDED_BY = "tools/theia/engine.py"
RECEIPT_NAME = "THEIA_BOUNTY_ENGINE.json"
DEFAULT_SELF_BOUNTY = REPO / "receipts" / "future" / "AUTONOMY_SCARS.json"

REUSE = {
    "reused": [
        "tools.future._common.write_receipt / REPO / RECEIPTS — receipt seal and write path",
        "tools.future.autonomy_scars.scars() — independent check that AUTONOMY_SCARS.json agrees with the module that produced it",
        "complete_ebpw missing-input doctrine — H.1 refuses a missing/zero factor rather than defaulting it to 1",
    ],
    "could_not_reuse": [
        "tools/odyssey/pareto_archive.py dominates()/composite_wus_per_hour_per_GB — axes are complete_ebpw, TPOT, TTFT, capability_passed, hcli_wus_per_hour for resident-body selection, not H.1 bounty factors",
        "tools/future/complete_ebpw.py cost() — bills representation parts in bytes/ms/bpw, not verified_reward/information_gain/risk",
        "tools/future/scar_reevaluator.py FAMILY_IMPL_COST_RANK — ordinal ranks over codec families, not H.1 cost terms",
        "tools/gravity_cost_vector.py / tools/cost_vector_t.py — B/M/F/L/R/T representation vectors, not bounty value",
        "tools/future/autonomy_scars.SCARS Python tuple as the artifact — the contract requires a receipts/ artifact; the JSON is the artifact, the module is only the independent verifier",
    ],
}

SECURITY_STATEMENT = {
    "network_egress": False,
    "credential_handling": False,
    "active_test": False,
    "payload_generation": False,
    "scanning": False,
    "ACTIVE_TEST": "modeled; transition refused; cannot be forced",
    "scope": (
        "immutable once pinned; loaded only from an operator-supplied "
        "authority file; never derived from bounty text; fail closed"
    ),
}


def run(receipt_path: Path) -> IntakeResult:
    path = Path(receipt_path)
    bounty, kind, doc = bounty_from_receipt(path)

    def inputs(_artifact: Path):
        return value_inputs_from_receipt(path, doc)

    return run_intake(
        bounty,
        value_inputs_factory=inputs,
        self_bounty_kind=kind,
        expected_schema=str(doc.get("schema")),
    )


def ladder_snapshot() -> list[dict[str, Any]]:
    out = []
    for stage in STAGES:
        ev = evaluate_wake(stage)
        out.append(
            {
                "name": stage.name,
                "size_hint": stage.size_hint,
                "purpose": stage.purpose,
                "status": stage.status,
                "blocker": stage.blocker,
                "wake_condition": stage.wake_condition,
                "wake_satisfied": ev.satisfied,
                "wake_missing": list(ev.missing),
            }
        )
    return out


def build_receipt_doc(result: IntakeResult) -> dict[str, Any]:
    return {
        "schema": "hawking.theia.bounty_engine.v1",
        "version": 1,
        "recorded_by": RECORDED_BY,
        "definition": DEFINITION,
        "evidence_class": "STATIC_ONLY",
        "claim_boundary": (
            "Bounty-engine sidecar. H.1 scores schedule work and do not "
            "declare a result true. No hardware measurement. No network "
            "egress, credential handling, scanning, payload generation, or "
            "ACTIVE_TEST. Model-ladder stages are BLOCKED_EXTERNAL."
        ),
        "halves": {
            "bounty_engine": {"status": "LIVE", "path": "tools/theia"},
            "model_ladder": {
                "status": "BLOCKED_EXTERNAL",
                "stages": [s.name for s in STAGES],
            },
        },
        "reuse": REUSE,
        "security": SECURITY_STATEMENT,
        "model_ladder": ladder_snapshot(),
        "self_bounty_run": result.to_json_dict(),
    }


def write_engine_receipt(result: IntakeResult) -> Path:
    return write_receipt(
        RECEIPT_NAME, build_receipt_doc(result), recorded_by=RECORDED_BY
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m tools.theia")
    parser.add_argument(
        "--self-bounty",
        default=str(DEFAULT_SELF_BOUNTY),
        help="local receipts/ artifact to ingest as a Hawking self-bounty",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="do not write receipts/future/THEIA_BOUNTY_ENGINE.json",
    )
    args = parser.parse_args(argv)
    result = run(Path(args.self_bounty))
    payload = result.to_json_dict()
    payload["definition"] = DEFINITION
    payload["model_ladder"] = {
        s["name"]: {
            "status": s["status"],
            "wake_condition_id": s["wake_condition"]["id"],
        }
        for s in ladder_snapshot()
    }
    if not args.no_write:
        out = write_engine_receipt(result)
        payload["engine_receipt"] = str(out)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
