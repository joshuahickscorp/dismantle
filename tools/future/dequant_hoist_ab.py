#!/usr/bin/env python3
"""G092 MEASURED: the dequant hoist is 1.63x, and the bit-identity is an artifact.

The candidate is built and run. Same bytes, same loads, the affine applied once
per 8-weight chunk instead of once per weight, using a precomputed per-chunk sum
of x that is a property of x rather than of the output row.

    production      195.7 GB/s
    hoist           319.7 GB/s     1.6338x
    arm_a_stripped  502.1 GB/s     (all arithmetic removed)

The hoist recovers 40.5% of the production-to-ARM-A span. Predicted bracket from
the issue-rate ladder was 1.268x-1.426x, so the MEASURED ratio is ABOVE the
projection - but this run is contended (loadavg 8.20) and ARM A's own ratio is
2.59x here against 1.51-1.64x in quieter runs, so every ratio in this run is
inflated. The SPAN RECOVERY is the load-robust number: 40.5%, against k6's 42%
on the earlier ladder. The hoist performs like k6 (2.0 FMA/B), not like the
1.6667 FMA/B its instruction count suggests.

THE BIT-IDENTITY IS AN ARTIFACT OF THE TEST INPUT AND MUST NOT BE QUOTED.
All 39,936 outputs match production exactly. That is not evidence the transform
is FP-exact: the harness fills x with (i % 17) * 0.125 - 1.0, dyadic values with
four mantissa bits, and the codes are 2-bit, so both orderings land on exactly
representable intermediates. It is the input LEAST likely to expose a
reassociation difference. Real activations are untested.

The comparison itself is sound: planting a 1e-6 error in the kernel took
bit_exact from 39,936 to 0 with max_abs 6.4e-4, which is 640 chunks times 1e-6.

    python3 tools/future/dequant_hoist_ab.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402

RECORDED_BY = "tools/future/dequant_hoist_ab.py"
RECEIPT_NAME = "DEQUANT_HOIST_AB.json"
RAW_REL = "receipts/future/_G092_HOIST_raw.json"

# The synthetic input the harness uses. Named because it is the reason the
# bit-identity cannot be quoted.
HARNESS_INPUT = "(i % 17) * 0.125 - 1.0"


class AbRefused(RuntimeError):
    """A raw arm is missing, or an arm did not run."""


def _raw() -> dict[str, Any]:
    path = REPO / RAW_REL
    if not path.is_file():
        raise AbRefused(f"{RAW_REL} is not on disk; run the probe first")
    return json.loads(path.read_text())


def measured() -> dict[str, Any]:
    mlp = _raw()["mlp"]
    for arm in ("production", "hoist", "arm_a_stripped"):
        if arm not in mlp:
            raise AbRefused(f"arm {arm} is absent; this is not a matched comparison")
    prod = float(mlp["production"]["effective_gb_s"])
    hoist = float(mlp["hoist"]["effective_gb_s"])
    arm_a = float(mlp["arm_a_stripped"]["effective_gb_s"])
    return {
        "production_gb_s": prod,
        "hoist_gb_s": hoist,
        "arm_a_stripped_gb_s": arm_a,
        "hoist_over_production": round(hoist / prod, 4),
        "arm_a_over_production": round(arm_a / prod, 4),
        "span_recovered": round((hoist - prod) / (arm_a - prod), 4),
        "loadavg": _raw().get("concurrent_load", {}).get("loadavg"),
        "why_span_and_not_ratio": (
            "this run is contended and ARM A's own ratio is 2.59x here against "
            "1.51-1.64x in quieter runs, so every ratio is inflated together. The "
            "fraction of the production-to-ARM-A span recovered is the load-robust "
            "comparison, and it puts the hoist at 40.5% against k6's 42% on the "
            "earlier ladder."
        ),
    }


def bit_identity() -> dict[str, Any]:
    cmp_rows = _raw()["mlp"]["hoist"]["output_compare"]
    total = sum(int(c["n_compared"]) for c in cmp_rows)
    exact = sum(int(c["n_bit_exact"]) for c in cmp_rows)
    return {
        "n_compared": total,
        "n_bit_exact": exact,
        "all_exact": exact == total,
        "MUST_NOT_BE_QUOTED_AS_BIT_IDENTICAL": (
            f"the harness fills x with {HARNESS_INPUT} - dyadic values with four "
            "mantissa bits - and the codes are 2-bit, so both orderings land on "
            "exactly representable intermediates. This is the input LEAST likely "
            "to expose a reassociation difference. Real activations are UNTESTED "
            "and the transform reassociates, so the honest expectation is that it "
            "is NOT bit-identical in production."
        ),
        "the_comparison_is_sound": (
            "planting a 1e-6 error in the kernel took bit_exact from 39936 to 0 "
            "with max_abs 6.4e-4, which is 640 chunks times 1e-6. The instrument "
            "detects what it is supposed to detect."
        ),
        "what_would_settle_it": (
            "re-run the comparison with x drawn from a real captured activation "
            "rather than the dyadic fill. Until then the FP behaviour of this "
            "transform on production data is unknown."
        ),
    }


def prediction_vs_measurement() -> dict[str, Any]:
    return {
        "predicted_bracket_from_ladder": [1.2679, 1.4256],
        "predicted_from_fma_count": 1.6,
        "measured_ratio_this_run": measured()["hoist_over_production"],
        "measured_span_recovery": measured()["span_recovered"],
        "k6_span_recovery_on_the_earlier_ladder": 0.42,
        "reading": (
            "on the load-robust measure the hoist behaves like k6 at 2.0 FMA/B, "
            "not like the 1.6667 its instruction count suggests. The likely cause "
            "is the one thing the FMA count ignores: the hoist adds a LOAD - "
            "sumx8[col>>3] - that k6 does not have. Trading six FMA for one load "
            "is not free, and an arithmetic-only model could not have said so."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "schema": "hawking.future.dequant_hoist_ab.v1",
        "version": 1,
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "gpu_authority": True,
        "kernel": "alu_roofline_affine_q2_geo_tpr64_tg128_hoist",
        "measured": measured(),
        "bit_identity": bit_identity(),
        "prediction_vs_measurement": prediction_vs_measurement(),
        "claim_boundary": (
            "One MLP layer, gate+up+down, on sealed-3.14, three dispatches in one "
            "command buffer, MTLCommandBuffer GPUStartTime/GPUEndTime, 11 reps "
            "after 5 warmup. CONTAMINATED WINDOW: absolute GB/s are not "
            "promotable and the ratio is inflated with every other ratio in the "
            "run, so the span recovery is the comparison that survives. The "
            "output comparison uses the harness's DYADIC synthetic input and "
            "therefore says nothing about FP behaviour on real activations."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_measured_receipt(
            REPO / "receipts" / "future" / RECEIPT_NAME, doc, RECORDED_BY,
            provenance=measurement_provenance(lock_held=True, lane="g092-hoist"),
        ))
        return 0
    print(json.dumps({k: doc[k] for k in ("measured", "prediction_vs_measurement")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
