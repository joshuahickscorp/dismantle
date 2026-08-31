#!/usr/bin/env python3
"""The live 71-TPS budget, ranked by gain per unit of experiment cost.

The important number this produces is not 71. It is 47.97: what the resident
would reach if every organ merely matched the bandwidth the LM head ALREADY
achieves on this box, with no byte reduction at all. That is the granularity
hypothesis's entire payoff, and it is a demonstrated regime rather than a
theoretical one.

The second important number is 66.54: the ceiling if every organ hit the clean
single-GEMV roof of 703.5 GB/s with today's bytes. It is not 71.21, because the
earlier 71.21 divided bytes by the roof and forgot the 0.99 ms host gap. So 71
TPS is NOT reachable at the clean roof on today's bytes. It needs the roof AND
about 7% fewer bytes, or the roof AND the host gap gone.

    python3 tools/future/causal_budget_71.py --record
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402

RECEIPT = REPO / "receipts" / "future" / "RESIDENT_71TPS_CAUSAL_BUDGET.json"

CLEAN_GEMV_GB_S = 703.5          # single clean GEMV, measured
DEMONSTRATED_GB_S = 497.4        # the LM head, measured on this box TODAY
HOST_GAP_MS = 0.989              # measured, 3 runs, stable to 5 decimals
ACTIVE_BYTES = 9_878_901_136

ORGANS: tuple[dict[str, Any], ...] = (
    {"organ": "mlp", "gb": 5.347795776, "ms": 15.541, "gb_s": 344.1, "dispatches": 192},
    {"organ": "deltanet", "gb": 2.961659904, "ms": 8.227, "gb_s": 360.0, "dispatches": 337},
    {"organ": "gqa", "gb": 0.891292160, "ms": 2.607, "gb_s": 341.9, "dispatches": 96},
    {"organ": "lm_head", "gb": 0.675430440, "ms": 1.358, "gb_s": 497.4, "dispatches": 2},
)

# Byte levers with a measured byte model. Capability is UNMEASURED for all.
BYTE_LEVERS: tuple[dict[str, Any], ...] = (
    {"id": "quantize_aux_u8", "gb_saved": 0.534773760, "status": "OPEN",
     "source": "receipts/future/MLP_AUXILIARY_INFORMATION.json"},
    {"id": "group_size_256", "gb_saved": 0.802160640, "status": "OPEN",
     "source": "receipts/future/MLP_AUXILIARY_INFORMATION.json"},
    {"id": "group_size_1024", "gb_saved": 1.002700800, "status": "OPEN",
     "source": "receipts/future/MLP_AUXILIARY_INFORMATION.json"},
)


def token_ms(gb_s: float | None = None, gb_saved: float = 0.0,
             host_ms: float = HOST_GAP_MS) -> float:
    """Token time if every organ ran at gb_s. None means keep measured rates."""
    total = 0.0
    saved_left = gb_saved
    for o in ORGANS:
        gb = o["gb"]
        take = 0.0
        # Every byte lever in this receipt is an MLP auxiliary-array change.
        if saved_left > 0 and o["organ"] == "mlp":
            take = min(saved_left, gb)
            saved_left -= take
        if gb_s:
            total += (gb - take) / gb_s * 1000.0
        else:
            # Keep the organ's OWN measured rate. Removing bytes at that rate is
            # the honest counterfactual: it does not assume the organ also gets
            # faster per byte, which is a separate hypothesis.
            total += o["ms"] * (gb - take) / gb
    return total + host_ms


def tps(ms: float) -> float:
    return 1000.0 / ms


def ladder() -> list[dict[str, Any]]:
    now = token_ms()
    rows = [
        {"rung": "measured now", "ms": round(now, 3), "tps": round(tps(now), 2),
         "requires": "nothing", "class": "MEASURED"},
        {"rung": "every organ at the LM head's demonstrated 497.4 GB/s",
         "ms": round(token_ms(DEMONSTRATED_GB_S), 3),
         "tps": round(tps(token_ms(DEMONSTRATED_GB_S)), 2),
         "requires": "executor work only; zero byte reduction",
         "class": "DEMONSTRATED_REGIME",
         "note": "this is the granularity hypothesis's entire payoff, and the "
                 "rate is already achieved on this box by one organ"},
    ]
    for lever in BYTE_LEVERS:
        ms = token_ms(DEMONSTRATED_GB_S, lever["gb_saved"])
        rows.append({
            "rung": f"demonstrated regime + {lever['id']}",
            "ms": round(ms, 3), "tps": round(tps(ms), 2),
            "requires": f"the above, plus {lever['gb_saved']:.3f} GB removed",
            "class": "DEMONSTRATED_PLUS_OPEN_BYTE_LEVER",
            "capability": "UNMEASURED",
        })
    roof = token_ms(CLEAN_GEMV_GB_S)
    rows.append({
        "rung": "every organ at the clean GEMV roof 703.5 GB/s",
        "ms": round(roof, 3), "tps": round(tps(roof), 2),
        "requires": "beating the LM head as well, on every organ",
        "class": "ROOF_ON_TODAYS_BYTES",
    })
    need_ms = 1000.0 / 71.0
    byte_ms_at_roof = need_ms - HOST_GAP_MS
    gb_for_71 = byte_ms_at_roof / 1000.0 * CLEAN_GEMV_GB_S
    rows.append({
        "rung": "71 TPS",
        "ms": round(need_ms, 3), "tps": 71.0,
        "requires": (
            f"the clean roof AND bytes down to {gb_for_71:.4f} GB "
            f"({(1 - gb_for_71 / (ACTIVE_BYTES / 1e9)) * 100:.1f}% fewer), "
            "or the clean roof AND the 0.99 ms host gap eliminated"
        ),
        "class": "NOT_REACHABLE_AT_THE_ROOF_ON_TODAYS_BYTES",
    })
    return rows


def experiments() -> list[dict[str, Any]]:
    """Ranked by TPS gain per unit of experiment cost."""
    now = token_ms()
    rows = []
    for o in ORGANS:
        if o["organ"] == "lm_head":
            continue
        demo_ms = o["gb"] / DEMONSTRATED_GB_S * 1000.0
        saved = o["ms"] - demo_ms
        rows.append({
            "id": f"reach_demonstrated_bandwidth_{o['organ']}",
            "organ": o["organ"],
            "current_ms": o["ms"],
            "current_gb_s": o["gb_s"],
            "target_ms_at_demonstrated": round(demo_ms, 3),
            "ms_saved": round(saved, 3),
            "tps_gain": round(tps(now - saved) - tps(now), 2),
            "falsifier": "one representative layer, contiguous, one/few fused "
                         "regions, identical arithmetic",
            "cost": "ONE_EXPERIMENT",
            "status": "RUNNING" if o["organ"] == "mlp" else "QUEUED",
        })
    for lever in BYTE_LEVERS:
        ms = token_ms(None, lever["gb_saved"])
        rows.append({
            "id": lever["id"],
            "gb_saved": lever["gb_saved"],
            "ms_saved": round(now - ms, 3),
            "tps_gain": round(tps(ms) - tps(now), 2),
            "falsifier": "held-out reconstruction plus organ error on a real layer",
            "cost": "ONE_FIT_PLUS_A_CAPABILITY_SCREEN",
            "capability": "UNMEASURED",
            "status": "OPEN",
        })
    rows.append({
        "id": "eliminate_all_host_gap",
        "ms_saved": HOST_GAP_MS,
        "tps_gain": round(tps(now - HOST_GAP_MS) - tps(now), 2),
        "falsifier": "already measured; this is the CEILING of the whole host class",
        "cost": "NOT_WORTH_RUNNING",
        "status": "CLOSED",
        "why": "receipts/future/WALL_GPU_RECONCILIATION.json bounds every host "
               "term at 0.99 ms combined",
    })
    rows.sort(key=lambda r: -r["tps_gain"])
    return rows


def build() -> dict[str, Any]:
    now = token_ms()
    return {
        "schema": "hawking.future.causal_budget_71.v1",
        "version": 1,
        "recorded_by": "tools/future/causal_budget_71.py",
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "gpu_authority": False,
        "measured_now": {
            "token_ms": round(now, 3), "tps": round(tps(now), 2),
            "active_bytes": ACTIVE_BYTES,
            "host_gap_ms": HOST_GAP_MS,
            "organs": list(ORGANS),
        },
        "the_two_numbers_that_matter": {
            "demonstrated_regime_tps": round(tps(token_ms(DEMONSTRATED_GB_S)), 2),
            "why": "what the resident reaches if every organ merely matches the "
                   "bandwidth the LM head ALREADY achieves here, with zero byte "
                   "reduction. A demonstrated regime, not a theoretical one.",
            "roof_on_todays_bytes_tps": round(tps(token_ms(CLEAN_GEMV_GB_S)), 2),
            "and_why_it_is_not_71": (
                "the earlier 71.21 divided bytes by the clean roof and omitted "
                "the 0.99 ms host gap. With it, the roof on today's bytes is "
                "66.54 TPS. 71 needs the roof AND about 7% fewer bytes, or the "
                "roof AND no host work at all."
            ),
        },
        "ladder": ladder(),
        "experiments_ranked_by_gain": experiments(),
        "claim_boundary": (
            "Arithmetic over measured organ times, measured byte shares and a "
            "measured host gap. Every rung above 'measured now' is a TARGET, not "
            "an achievement: no organ other than the LM head has been shown to "
            "reach 497 GB/s, and no byte lever has passed a capability screen. "
            "The ms figures are DIAGNOSTIC_RELATIVE; the byte shares are exact."
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
    for r in d["ladder"]:
        print(f"  {r['tps']:6.2f} TPS  {r['ms']:7.3f} ms   {r['rung']}")
    print()
    for e in d["experiments_ranked_by_gain"][:6]:
        print(f"  +{e['tps_gain']:5.2f} TPS  {e['cost']:32s} {e['id']}")
