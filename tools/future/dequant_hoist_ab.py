#!/usr/bin/env python3
"""G092 REFUTED: the dequant hoist buys nothing, and the 1.63x was the input.

The candidate is built and it is arithmetically correct. It is not faster.

Same binary, same arms, two inputs:

                        synthetic dyadic    real activations
    production               195.7 GB/s          317.4 GB/s
    hoist                    319.7 GB/s          317.3 GB/s
    arm_a_stripped           502.1 GB/s          502.4 GB/s

ARM A is 502.1 and 502.4 - the machine state is comparable across the two runs.
The hoist is 319.7 and 317.3 - the hoist does not care what x is. PRODUCTION is
the arm that moves: 195.7 to 317.4, a factor of 1.6235 from the INPUT ALONE.

So the earlier headline, hoist/production = 1.6338x, was not the hoist running
fast. It was production running slow on the harness fill. On real activations
the ratio is 0.9995x. There is no win here.

WHY THE INPUT CHANGES PRODUCTION AND NOT THE HOIST IS NOT ESTABLISHED. The two
kernels differ in exactly one way that could plausibly interact with operand
values: production forms w = q*scale + bias and then w*x per weight, so both the
dequantised weight and the product depend on scale/bias magnitudes; the hoist
forms q*x with q in {0,1,2,3} and applies scale once per chunk. A denormal or
flush-to-zero path in the per-weight product would show exactly this signature.
That is a HYPOTHESIS, not a finding, and it is not required for the verdict.

The verdict does not depend on the mechanism: on real activations, matched
bytes, matched arms, same machine state, THE HOIST IS NOT FASTER.

THE INSTRUMENT CONSEQUENCE IS LARGER THAN THE CANDIDATE. Every ratio this
harness has produced against a fill_f32 production arm is measured against a
production arm that may be depressed by up to 1.62x. The ALU roofline's own
production arm was 329.6 GB/s, close to the real-activation 317.4 and far from
195.6, so the roofline is NOT obviously contaminated - but that is now a thing
to check per receipt, not to assume.

FP: the hoist is NOT bit-identical on real activations, and the error is at f32
epsilon. rel_fro 8.2e-08 to 9.8e-08, max_abs 1.19e-07 on gate/up and 4.77e-07 on
down. The transform reassociates, so this is the expected and acceptable answer;
the earlier bit-exact result on the dyadic fill was the artifact, as recorded.

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

# Two runs of the SAME binary. The only difference is x.
REAL_REL = "receipts/future/_G092_HOIST_REALX_raw.json"
SYNTH_REL = "receipts/future/_G092_HOIST_raw.json"

HARNESS_FILL = "(i % 17) * 0.125 - 1.0"
ARMS = ("production", "hoist", "arm_a_stripped")


class AbRefused(RuntimeError):
    """A raw arm is missing, or an arm did not run."""


def _raw(rel: str) -> dict[str, Any]:
    path = REPO / rel
    if not path.is_file():
        raise AbRefused(f"{rel} is not on disk; run the probe first")
    return json.loads(path.read_text())


def _arms(rel: str) -> dict[str, float]:
    mlp = _raw(rel)["mlp"]
    for arm in ARMS:
        if arm not in mlp:
            raise AbRefused(f"{rel}: arm {arm} is absent; this is not a matched comparison")
    return {a: float(mlp[a]["effective_gb_s"]) for a in ARMS}


def _one_run(rel: str, label: str) -> dict[str, Any]:
    gb = _arms(rel)
    prod, hoist, arm_a = gb["production"], gb["hoist"], gb["arm_a_stripped"]
    return {
        "input": label,
        "raw": rel,
        "production_gb_s": prod,
        "hoist_gb_s": hoist,
        "arm_a_stripped_gb_s": arm_a,
        "hoist_over_production": round(hoist / prod, 4),
        "arm_a_over_production": round(arm_a / prod, 4),
        "span_recovered": round((hoist - prod) / (arm_a - prod), 4),
        "loadavg": _raw(rel).get("concurrent_load", {}).get("loadavg"),
    }


def measured() -> dict[str, Any]:
    """The headline is the REAL-ACTIVATION run. The synthetic run is the control."""
    real = _one_run(REAL_REL, "real_captured_activation")
    synth = _one_run(SYNTH_REL, f"synthetic_dyadic_fill {HARNESS_FILL}")

    # The machine-state check. If ARM A moved between the runs, the two runs are
    # not comparable and the whole conclusion is void.
    arm_a_drift = abs(real["arm_a_stripped_gb_s"] - synth["arm_a_stripped_gb_s"])
    arm_a_rel = arm_a_drift / synth["arm_a_stripped_gb_s"]
    if arm_a_rel > 0.05:
        raise AbRefused(
            "ARM A moved "
            f"{synth['arm_a_stripped_gb_s']:.1f} -> {real['arm_a_stripped_gb_s']:.1f} GB/s "
            f"({arm_a_rel:.1%}) between the two runs; machine state is not held "
            "fixed, so the input cannot be credited with the production change"
        )

    return {
        "verdict": "NO_SPEEDUP_ON_REAL_ACTIVATIONS",
        "headline_hoist_over_production": real["hoist_over_production"],
        "real_activation_run": real,
        "synthetic_control_run": synth,
        "arm_a_holds_across_runs": {
            "synthetic_gb_s": synth["arm_a_stripped_gb_s"],
            "real_gb_s": real["arm_a_stripped_gb_s"],
            "relative_drift": round(arm_a_rel, 4),
            "why_it_matters": (
                "ARM A has all the arithmetic removed, so it reads the same bytes "
                "in both runs and is the machine-state control. It holds to "
                f"{arm_a_rel:.2%}, which is what licenses attributing production's "
                "change to the input rather than to contention."
            ),
        },
        "the_input_moved_production_not_the_hoist": {
            "production_synth_gb_s": synth["production_gb_s"],
            "production_real_gb_s": real["production_gb_s"],
            "production_real_over_synth": round(
                real["production_gb_s"] / synth["production_gb_s"], 4
            ),
            "hoist_synth_gb_s": synth["hoist_gb_s"],
            "hoist_real_gb_s": real["hoist_gb_s"],
            "hoist_real_over_synth": round(real["hoist_gb_s"] / synth["hoist_gb_s"], 4),
            "reading": (
                "the hoist is input-insensitive and production is input-sensitive. "
                "The apparent 1.63x was production being slow on the fill."
            ),
        },
        "mechanism_is_not_established": (
            "production forms w = q*scale + bias then w*x per weight; the hoist "
            "forms q*x with q in {0,1,2,3} and applies scale once per chunk. A "
            "denormal or flush-to-zero path in the per-weight product would "
            "produce this signature. That is a hypothesis. The verdict does not "
            "rest on it."
        ),
    }


def bit_identity() -> dict[str, Any]:
    """FP behaviour on REAL activations. The dyadic-fill bit-exactness was the artifact."""
    cmp_rows = _raw(REAL_REL)["mlp"]["hoist"]["output_compare"]
    total = sum(int(c["n_compared"]) for c in cmp_rows)
    exact = sum(int(c["n_bit_exact"]) for c in cmp_rows)
    return {
        "measured_on": "real_captured_activation",
        "n_compared": total,
        "n_bit_exact": exact,
        "all_exact": exact == total,
        "max_rel_fro": max(float(c["rel_fro"]) for c in cmp_rows),
        "max_abs_err": max(float(c["max_abs_err"]) for c in cmp_rows),
        "per_tensor": [
            {
                "tensor": c["tensor"],
                "bit_identical": bool(c["bit_identical"]),
                "rel_fro": float(c["rel_fro"]),
                "max_abs_err": float(c["max_abs_err"]),
            }
            for c in cmp_rows
        ],
        "reading": (
            "NOT bit-identical, and that is the correct answer for a transform "
            "that reassociates. The error sits at f32 epsilon: rel_fro under 1e-7 "
            "on every projection. Numerically this transform would have been safe "
            "to adopt. It is rejected on speed, not on accuracy."
        ),
        "the_earlier_bit_exact_result_was_the_artifact": (
            f"the harness fill is {HARNESS_FILL} - dyadic values with four mantissa "
            "bits - and the codes are 2-bit, so both orderings landed on exactly "
            "representable intermediates. That was recorded as an artifact at the "
            "time and the real-activation run confirms it."
        ),
    }


def instrument_consequence() -> dict[str, Any]:
    return {
        "scar": "FILL_F32_IS_NOT_A_NEUTRAL_INPUT_FOR_THE_PRODUCTION_DEQUANT_ARM",
        "statement": (
            "the synthetic dyadic fill measured the production kernel at 195.6 "
            "GB/s across two runs and real activations measured it at 317.4, a "
            "1.62x difference from the input alone, while ARM A and the hoist "
            "were unmoved. Any ratio taken against a fill_f32 production arm is "
            "therefore suspect until the arm is re-run on real x."
        ),
        "what_is_NOT_claimed": (
            "this does not retroactively void the ALU roofline. Its production "
            "arm measured 329.6 GB/s, which is near the real-activation 317.4 and "
            "far from 195.6, so that receipt does not carry the signature. The "
            "obligation is to CHECK each receipt, not to assume contamination."
        ),
        "reopen": (
            "if any future receipt reports a production dequant arm below ~250 "
            "GB/s on this machine, suspect the input before the kernel."
        ),
    }


def prediction_vs_measurement() -> dict[str, Any]:
    real = _one_run(REAL_REL, "real")
    return {
        "predicted_bracket_from_ladder": [1.2679, 1.4256],
        "measured_on_real_activations": real["hoist_over_production"],
        "inside_bracket": False,
        "direction": "BELOW",
        "reading": (
            "the issue-rate ladder predicted 1.27x-1.43x from removing one FMA "
            "per weight. The measurement says 1.00x. The ladder's premise - that "
            "the production kernel is issue-limited on that FMA - is refuted for "
            "this kernel on real input: removing the FMA changes nothing, so the "
            "FMA was not on the critical resource."
        ),
        "what_this_kills": (
            "the per-weight-affine-arithmetic branch of the ALU school for the "
            "MLP q2 matvec. Cheaper decode arithmetic of THIS shape is not the "
            "lever. It does not kill the wider decode-tax target, which is "
            "measured against ARM A's full 1.58x span, still unexplained."
        ),
    }


def build() -> dict[str, Any]:
    m = measured()
    return {
        "obligation": "G092",
        "candidate": "affine_q2_geo_tpr64_tg128_hoist",
        "kernel": "alu_roofline_affine_q2_geo_tpr64_tg128_hoist",
        "shader": "crates/hawking-core/examples/alu_roofline_organs.metal",
        "transform": (
            "apply the per-group affine once per 8-weight chunk instead of once "
            "per weight, using sumx8 - a per-chunk sum of x that is a property of "
            "x, not of the output row, so it amortises over all rows"
        ),
        "verdict": m["verdict"],
        "measured": m,
        "bit_identity": bit_identity(),
        "prediction_vs_measurement": prediction_vs_measurement(),
        "instrument_consequence": instrument_consequence(),
        "correction": (
            "an earlier version of this receipt reported 1.6338x as the measured "
            "speedup. That number was real as an observation and wrong as a "
            "claim about the hoist: it compared the hoist against a production "
            "arm depressed by the synthetic input. Corrected here on evidence, "
            "not withdrawn quietly."
        ),
        "ms_per_token_saved": 0.0,
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
    print(json.dumps(
        {k: doc[k] for k in ("verdict", "measured", "prediction_vs_measurement")},
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
