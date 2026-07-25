#!/usr/bin/env python3.12
"""Math-Preserve PASS 3: profile-conditioned packing from PASS 2's frozen manifest.

Not yet wired to a real packing run -- PASS 2 has sealed zero sparse layers of
evidence as of this module's creation (PASS 1 is still mid-fetch), so there is no
real manifest to pack against, and writing the packing logic against a shape
nobody has seen yet is exactly the kind of speculative work this campaign avoids
elsewhere. This module exists so the pipeline's shape is complete end to end and
the entry point + refusal gate are proven now, before the expensive part.

What PASS 3 has to do once PASS 2 is frozen:
  1. Re-stream every source shard PASS 1 already verified once (re-fetch if
     evicted; the campaign's own rule is "do not redownload verified bytes" where
     avoidable, so prefer whatever is still resident before refetching).
  2. For each routed-expert tensor, look up its (layer, expert) in the frozen
     manifest's `coalition_expert_ids` / `remainder_expert_ids`: coalition members
     pack at a protected rate, remainder at the floor rate -- this is the one
     piece `glm52_pack.pack_shard` cannot do today, since its `production_rung`
     is a single rate applied uniformly to every routed-expert row in a shard
     (see glm52_pack.py:377-466, specifically the `chosen = next(...)` selection
     against one fixed `production_rung`). Extending it needs an optional
     `rate_override: dict[tuple[int, int], str]` keyed by (layer, expert),
     threaded into that same selection -- not a new packer.
  3. Every non-expert category packs exactly as General-R0 already does (same
     ladder, same protected-tensor handling) -- Math-Preserve's distinguishing
     claim is WHICH experts get bits, not a second opinion on everything else.
  4. Verify exact bytes, payload hashes, and complete coverage against the same
     official tensor count General-R0 was graded against (glm52_assemble.py).
  5. Evict raw source only after the corresponding compact payload is sealed --
     same discipline PASS 1 already applies, not a new invariant to get right.

    python3.12 tools/prometheus/math_pass3_pack.py status
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

MANIFEST = REPO / "PROMETHEUS_MATH_ALLOCATION_MANIFEST.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def manifest_ready() -> tuple[bool, dict | None, str]:
    if not MANIFEST.exists():
        return False, None, f"{MANIFEST} does not exist yet -- run PASS 2's `freeze`"
    manifest = json.loads(MANIFEST.read_text())
    if not manifest.get("complete"):
        missing = manifest.get("sparse_layers_missing_evidence", [])
        return False, manifest, (
            f"{MANIFEST} exists but complete=false ({len(missing)} sparse layers "
            "still missing capsule evidence) -- PASS 2 should have refused to "
            "write this; do not pack against it"
        )
    return True, manifest, ""


def status() -> dict:
    ready, manifest, reason = manifest_ready()
    return {
        "schema": "hawking.prometheus.math_pass3_status.v1",
        "at": _now(),
        "ready_to_pack": ready,
        "reason": reason or "manifest is complete; packing not yet implemented (see module docstring)",
        "manifest_layers": len(manifest["per_layer"]) if manifest else 0,
    }


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "status"
    if command == "status":
        print(json.dumps(status(), indent=2, sort_keys=True))
        return 0
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
