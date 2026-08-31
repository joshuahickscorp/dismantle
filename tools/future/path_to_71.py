#!/usr/bin/env python3
"""Not one promised path to 71. A set of compositions, each with its evidence.

Every component here carries its own evidence class, and the arithmetic keeps
them apart: QUALIFIED means measured on the real resident with parity;
PROSPECTIVE means the byte model is measured but capability is not; SPECULATIVE
means a school is running and no number exists yet.

The honest headline: everything currently QUALIFIED and PROSPECTIVE composed
together does not reach 50 TPS, let alone 71. 71 requires a representation
result that does not exist yet, and this module says so by arithmetic rather
than by hedging.

    python3 tools/future/path_to_71.py --record
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402

RECEIPT = REPO / "receipts" / "future" / "PATH_TO_71.json"

TOKEN_MS = 28.722
HOST_MS = 0.989
ACTIVE_GB = 9.878901136
MLP_GB = 5.347795776
MLP_MS = 15.541
DELTANET_MS = 8.227

# --- components, each with what is actually known about it -----------------
COMPONENTS: dict[str, dict[str, Any]] = {
    "ba_delta_fusion": {
        "ms_saved": 0.1384, "evidence": "QUALIFIED",
        "parity": "token-identical across 8 runs, 128 tokens, fallbacks 0",
        "source": "receipts/future/BA_DELTA_AB.json",
        "cost": "set one env flag",
    },
    "mlp_dispatch_size": {
        "ms_saved": 1.258, "evidence": "PROSPECTIVE",
        "why_prospective": "the 374.4 GB/s at 337.7 MB/dispatch is measured, but "
                           "no implementation batches the MLP that way yet",
        "parity": "arithmetic is unchanged; a real implementation must prove it",
        "source": "receipts/future/DISPATCH_SIZE_SWEEP.json",
        "cost": "an executor change",
    },
    "aux_group_size_1024": {
        "gb_saved": 1.002700800, "evidence": "PROSPECTIVE",
        "why_prospective": "byte model exact (aux(G) = 4*n_params/G + 58176); "
                           "capability UNMEASURED, no generate gate has been run",
        "source": "receipts/future/MLP_AUXILIARY_INFORMATION.json",
        "cost": "a refit plus a capability screen",
    },
    "aux_u8": {
        "gb_saved": 0.534773760, "evidence": "PROSPECTIVE",
        "why_prospective": "same; and it overlaps group_size, they are not additive",
        "source": "receipts/future/MLP_AUXILIARY_INFORMATION.json",
        "overlaps": ["aux_group_size_1024"],
        "cost": "a refit plus a capability screen",
    },
    "mlp_entropy_floor": {
        "gb_saved": 0.277697891, "evidence": "PROSPECTIVE",
        "why_prospective": "the 1.87018-of-2-bits floor is measured; an actual "
                           "entropy coder that a GPU can consume in-register does "
                           "not exist and may cost more ALU than it saves bytes",
        "source": "receipts/future/MLP_CODE_INFORMATION.json",
        "cost": "a codec plus a native consumer",
    },
    "mlp_low_rank_program": {
        "gb_saved": 0.0, "evidence": "REFUTED",
        "why": "activation-WEIGHTED functional rank, not raw SVD: 10% held-out "
               "output error needs rank 5120 (full) on three of four layers. The "
               "largest rank whose complete ledger is under the incumbent "
               "5,347,795,776 bytes is r=617, and there hold error is 35-84%. "
               "Reaching 10% costs 44.3 GB added.",
        "source": "receipts/future/MLP_FUNCTIONAL_RANK.json",
    },
    "mlp_nonlinear_program": {
        "gb_saved": 0.0, "evidence": "REFUTED",
        "why": "six families on the real corpus, all held-out by prompt: "
               "FACTORIZE_THE_FACTORS 0.977, PRODUCT_DICTIONARY 0.997 (0.928 "
               "even with an unfair Y-oracle), NONLINEAR_GENERATOR 0.918, "
               "CONDITIONAL_PROGRAM 0.917, GENERATED_BLOCK 0.943. The MEAN "
               "PREDICTOR on the same hold set is 0.970, so the factorized "
               "program is worse than predicting the mean. Common mechanism: "
               "every one is an r-bottleneck. Reopen is a full-width structured "
               "operator only.",
        "source": "receipts/future/MLP_NONLINEAR_PROGRAM.json",
    },
    "deltanet_generated_transition": {
        "gb_saved": 0.0, "evidence": "REFUTED",
        "why": "the candidate removed 2,139,096,960 bytes and added 4,548,560 - "
               "worth about 5.8 ms/token if the function had held. It does not: "
               "the shared map does not emit this layer's coefficients and the "
               "state diverges at STEP 1, not after a long horizon. "
               "MEASURED_NEGATIVE, economics IMMATERIAL.",
        "source": "receipts/future/DELTANET_GENERATED_TRANSITION.json",
    },
    "deltanet_q3": {
        "gb_saved": 0.503316480, "evidence": "REFUTED",
        "why": "entropies are 3.465-3.479 of 4 across q/k/v/z and any_supported "
               "is false; no sensitivity measurement licenses the reduction",
        "source": "receipts/future/DELTANET_QKVZ_PRECISION.json",
    },
}

SCHOOLS_RUNNING = (
    "mlp full-width structured operator (Monarch/butterfly/distilled control) - "
    "the only reopen left after six r-bottleneck families died",
    "mlp error budget: what relative output error the model actually tolerates",
    "auxiliary byte levers under a real capability screen",
    "is the MLP kernel ALU-bound or memory-bound",
    "deltanet multi-step authority (the instrument, after the one-step candidate died)",
    "sparse residual concentration curve",
)


def _apply(ms: float, gb_removed: float, comp_ms: float) -> float:
    """Bytes come off the MLP at the MLP's own measured rate."""
    mlp_after = MLP_MS * (MLP_GB - gb_removed) / MLP_GB
    return ms - (MLP_MS - mlp_after) - comp_ms


def compose(ids: list[str]) -> dict[str, Any]:
    gb = 0.0
    ms = 0.0
    seen: set[str] = set()
    used: list[str] = []
    skipped: list[str] = []
    for cid in ids:
        c = COMPONENTS[cid]
        if c["evidence"] == "REFUTED":
            skipped.append(f"{cid}: REFUTED")
            continue
        if any(o in seen for o in c.get("overlaps", [])):
            skipped.append(f"{cid}: overlaps {c['overlaps']}, not additive")
            continue
        seen.add(cid)
        used.append(cid)
        gb += c.get("gb_saved", 0.0)
        ms += c.get("ms_saved", 0.0)
    total = _apply(TOKEN_MS, gb, ms)
    worst = "QUALIFIED"
    for cid in used:
        if COMPONENTS[cid]["evidence"] == "PROSPECTIVE":
            worst = "PROSPECTIVE"
    return {
        "components": used,
        "skipped": skipped,
        "gb_removed": round(gb, 4),
        "ms_removed_directly": round(ms, 4),
        "token_ms": round(total, 3),
        "tps": round(1000.0 / total, 2),
        "weakest_evidence": worst,
    }


def paths() -> list[dict[str, Any]]:
    now = {"path": "PATH_00", "label": "measured now",
           "token_ms": TOKEN_MS, "tps": round(1000.0 / TOKEN_MS, 2),
           "weakest_evidence": "MEASURED", "components": [], "skipped": []}
    rows = [now]
    for pid, label, ids in (
        ("PATH_01", "everything QUALIFIED today", ["ba_delta_fusion"]),
        ("PATH_02", "qualified + the executor lever that survived",
         ["ba_delta_fusion", "mlp_dispatch_size"]),
        ("PATH_03", "that + the best auxiliary lever",
         ["ba_delta_fusion", "mlp_dispatch_size", "aux_group_size_1024", "aux_u8"]),
        ("PATH_04", "everything on record, refuted excluded",
         ["ba_delta_fusion", "mlp_dispatch_size", "aux_group_size_1024", "aux_u8",
          "mlp_entropy_floor", "deltanet_q3"]),
    ):
        r = compose(list(ids))
        r["path"] = pid
        r["label"] = label
        rows.append(r)
    return rows


def gap_to_71() -> dict[str, Any]:
    best = max(paths(), key=lambda r: r["tps"])
    need_ms = 1000.0 / 71.0
    # After PATH_04, how much MORE has to go, at current executor efficiency?
    remaining = best["token_ms"] - need_ms
    gpu_after = best["token_ms"] - HOST_MS
    gpu_needed = need_ms - HOST_MS
    return {
        "best_composed_path": best["path"],
        "best_composed_tps": best["tps"],
        "best_composed_token_ms": best["token_ms"],
        "target_token_ms": round(need_ms, 3),
        "still_to_remove_ms": round(remaining, 3),
        "still_to_remove_share_of_gpu": round(1 - gpu_needed / gpu_after, 4),
        "verdict": (
            "Everything on record — every qualified win, every prospective byte "
            f"lever, refuted ones excluded — reaches {best['tps']} TPS. Reaching "
            f"71 needs another {remaining:.2f} ms, which is "
            f"{(1 - gpu_needed / gpu_after) * 100:.0f}% of the GPU time that "
            "would remain. No component on record can supply that. It requires a "
            "representation result that does not exist yet."
        ),
        "what_would_supply_it": (
            "A functional replacement for the MLP or DeltaNet that eliminates "
            "information rather than coding it better. Both organs are at their "
            "entropy floor, so the only remaining source of that magnitude is "
            "weights that need not exist independently."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "schema": "hawking.future.path_to_71.v1",
        "version": 1,
        "recorded_by": "tools/future/path_to_71.py",
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "gpu_authority": False,
        "baseline": {"token_ms": TOKEN_MS, "host_ms": HOST_MS,
                     "tps": round(1000.0 / TOKEN_MS, 2), "active_gb": ACTIVE_GB},
        "components": COMPONENTS,
        "paths": paths(),
        "gap_to_71": gap_to_71(),
        "schools_running": list(SCHOOLS_RUNNING),
        "claim_boundary": (
            "Arithmetic over recorded components. QUALIFIED means measured on the "
            "real resident with parity. PROSPECTIVE means the byte model is "
            "measured and capability is NOT — no generate gate has been run on "
            "any auxiliary lever. Overlapping levers are not summed. A composed "
            "path is a prediction, not a result, and every one above PATH_01 "
            "contains at least one component nobody has implemented."
        ),
    }


def record() -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n")
    return RECEIPT


if __name__ == "__main__":
    d = build()
    if "--record" in sys.argv:
        print(f"wrote {record()}")
    for r in d["paths"]:
        print(f"  {r['path']}  {r['tps']:6.2f} TPS  {r['token_ms']:7.3f} ms  "
              f"[{r['weakest_evidence']:11s}] {r['label']}")
    g = d["gap_to_71"]
    print(f"\n  gap: {g['still_to_remove_ms']} ms more, "
          f"{g['still_to_remove_share_of_gpu'] * 100:.0f}% of remaining GPU time")
