#!/usr/bin/env python3
"""G095: the 60-TPS gap, per-organ win table, and the escalation clock.

S026 makes 60 TPS the intermediate attack target and 71 the convergence target.
This module computes both gaps from disk authority rather than from a typed
number, prices every dominant organ in COMPLETE-TOKEN TPS at 10/20/30% and at
perfect removal, ranks the live experiment set by MAX MS REMOVABLE, and runs the
escalation clock S026 §48 requires.

THE CLOCK DOES NOT FORCE ABANDONMENT. It states what each phase licenses, so
that continuing to work the incumbent past +18h is a decision someone made
rather than a default nobody noticed.

    python3 tools/future/gap_ledger_60.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402
import causal_budget_71 as cb  # noqa: E402

RECORDED_BY = "tools/future/gap_ledger_60.py"
RECEIPT_NAME = "GAP_LEDGER_60.json"
BUDGET_REL = "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"
CLOCK_REL = "receipts/future/TPS_ESCALATION_CLOCK_START.json"

CHECKPOINTS = (40.0, 50.0, 60.0, 71.0)

# S025's materiality threshold. Below this a perfect win is not worth an hour.
MATERIAL_MS = 1.0

# S026 §48. Each phase names what it LICENSES, not what it forbids.
PHASES = (
    (0.0, "DISCOVERY", "incumbent-kernel work is the privileged route"),
    (6.0, "SURVIVOR_OR_COLLAPSE",
     "require a credible multi-ms survivor OR a major search-space collapse; "
     "if neither, the absence is itself the result and must be recorded"),
    (12.0, "WIDEN",
     "activate the alternate-resident and decoding schools strongly, in "
     "parallel with incumbent work rather than instead of it"),
    (18.0, "DEPRIVILEGE",
     "if the incumbent is still under 50 TPS with no credible path, stop "
     "treating incumbent-only optimization as privileged"),
)


class GapRefused(RuntimeError):
    """Disk authority is missing or self-inconsistent."""


def _budget() -> dict[str, Any]:
    p = REPO / BUDGET_REL
    if not p.is_file():
        raise GapRefused(f"{BUDGET_REL} is not on disk; there is no live token budget")
    d = json.loads(p.read_text())
    for k in ("decode_wall_ms_per_token", "decode_wall_tps", "organs"):
        if k not in d:
            raise GapRefused(f"{BUDGET_REL} is missing {k}")
    return d


def live() -> dict[str, Any]:
    d = _budget()
    ms = float(d["decode_wall_ms_per_token"])
    tps = float(d["decode_wall_tps"])
    # The receipt carries both; if they disagree the receipt is not authority.
    if abs(1000.0 / ms - tps) > 0.05:
        raise GapRefused(
            f"{BUDGET_REL} is self-inconsistent: {ms} ms/token implies "
            f"{1000.0/ms:.3f} TPS but the receipt says {tps}"
        )
    return {
        "ms_per_token": ms,
        "tps": tps,
        "gpu_ms_per_token": float(d["decode_gpu_ms_per_token"]),
        "source": BUDGET_REL,
        "git_head": d["organs"].get("git_head"),
    }


def checkpoints() -> list[dict[str, Any]]:
    cur = live()["ms_per_token"]
    out = []
    for t in CHECKPOINTS:
        need = 1000.0 / t
        out.append({
            "tps": t,
            "ms_per_token_required": round(need, 4),
            "ms_to_remove_from_live": round(cur - need, 4),
            "fraction_of_token_to_remove": round((cur - need) / cur, 4),
            "reached": cur <= need,
        })
    return out


def organ_win_table() -> list[dict[str, Any]]:
    """S026 §12: every organ priced in COMPLETE-TOKEN TPS, not in organ percent."""
    d = _budget()
    cur = float(d["decode_wall_ms_per_token"])
    rows = []
    for r in d["organs"]["rows"]:
        ms = float(r["gpu_ms"])
        entry = {
            "organ": r["organ"],
            "current_ms": ms,
            "share_of_token": round(ms / cur, 4),
            "dispatches": r["dispatches"],
        }
        for pct in (10, 20, 30, 100):
            saved = ms * pct / 100.0
            entry[f"tps_at_{pct}pct_win"] = round(1000.0 / (cur - saved), 3)
            entry[f"ms_at_{pct}pct_win"] = round(saved, 4)
        entry["perfect_removal_max_ms"] = round(ms, 4)
        entry["material"] = ms >= MATERIAL_MS
        entry["why"] = (
            "perfect elimination is below the materiality threshold; do not "
            f"spend an hour here" if ms < MATERIAL_MS else
            f"a 20% win here is {round(ms*0.2,3)} ms of the {round(cur,3)} ms token"
        )
        rows.append(entry)
    rows.sort(key=lambda r: -r["current_ms"])
    return rows


def three_dominant() -> dict[str, Any]:
    """S026 §11: mlp_gate_up, mlp_down and DeltaNet dominate. Price them jointly."""
    d = _budget()
    cur = float(d["decode_wall_ms_per_token"])
    want = {"mlp_gate_up", "mlp_down", "deltanet"}
    rows = [r for r in d["organs"]["rows"] if r["organ"] in want]
    if len(rows) != 3:
        raise GapRefused(
            f"expected the three dominant organs {sorted(want)}, found "
            f"{sorted(r['organ'] for r in rows)}"
        )
    total = sum(float(r["gpu_ms"]) for r in rows)
    out = {
        "organs": sorted(want),
        "combined_ms": round(total, 4),
        "share_of_token": round(total / cur, 4),
    }
    for pct in (10, 20, 30):
        saved = total * pct / 100.0
        out[f"tps_at_{pct}pct_across_all_three"] = round(1000.0 / (cur - saved), 3)
        out[f"ms_at_{pct}pct_across_all_three"] = round(saved, 4)
    out["reading"] = (
        f"these three are {out['share_of_token']:.1%} of the token. A 20% win "
        f"across all three is {out['ms_at_20pct_across_all_three']} ms, which "
        f"takes the LIVE resident to {out['tps_at_20pct_across_all_three']} TPS - "
        "not to 60 on its own. S026's arithmetic reaching 60 assumed the ~49.8 "
        "PROSPECTIVE composed path as the base, and that path is not qualified."
    )
    return out


def ranked_experiments() -> list[dict[str, Any]]:
    """S026 §H: rank every live experiment by MAX MS REMOVABLE."""
    out = []
    for e in cb.experiments():
        ms = float(e.get("ms_saved", 0.0))
        out.append({
            "id": e["id"],
            "max_ms_removable": ms,
            "material": ms >= MATERIAL_MS,
            "status": e.get("status") or e.get("lever_status") or "OPEN",
            "organ": e.get("organ"),
        })
    out.sort(key=lambda r: -r["max_ms_removable"])
    return out


def _clock_start() -> dict[str, Any]:
    """Stamped once. A clock that restarts every build measures nothing."""
    p = REPO / CLOCK_REL
    if p.is_file():
        return json.loads(p.read_text())
    doc = {
        "started_unix": time.time(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "started_at_tps": live()["tps"],
        "authority": "S026 §48",
        "note": (
            "stamped once and never rewritten. Restarting this clock on every "
            "build would make every phase permanently DISCOVERY."
        ),
    }
    p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def escalation_clock() -> dict[str, Any]:
    start = _clock_start()
    elapsed_h = (time.time() - float(start["started_unix"])) / 3600.0
    phase = PHASES[0]
    for p in PHASES:
        if elapsed_h >= p[0]:
            phase = p
    nxt = next((p for p in PHASES if p[0] > elapsed_h), None)
    cur_tps = live()["tps"]
    return {
        "started_utc": start["started_utc"],
        "started_at_tps": start["started_at_tps"],
        "elapsed_hours": round(elapsed_h, 3),
        "phase": phase[1],
        "phase_licenses": phase[2],
        "hours_to_next_phase": round(nxt[0] - elapsed_h, 3) if nxt else None,
        "next_phase": nxt[1] if nxt else None,
        "tps_moved_since_start": round(cur_tps - float(start["started_at_tps"]), 3),
        "deprivilege_condition_would_fire": cur_tps < 50.0,
        "not_an_abandonment": (
            "the clock forbids complacency, not the incumbent. Passing a phase "
            "boundary licenses widening the search; it never requires dropping "
            "work that is producing evidence."
        ),
    }


# arm_a / production, MEASURED at warmup 60 on real captured activations. These
# are the rate each matvec reaches with ALL of its decode arithmetic removed and
# every byte still loaded, so they bound the entire arithmetic school.
ARM_A_RATIO = {
    "q2": (1.5128, ("mlp_gate_up", "mlp_down"),
           "receipts/future/OP_CLASS_ABLATION.json"),
    "q4": (1.5599, ("deltanet", "q4_remainder", "gqa_attention"),
           "receipts/future/Q4_BITCAST_AB.json"),
}


def arithmetic_ceiling() -> dict[str, Any]:
    """What the WHOLE decode-arithmetic school is worth if it perfectly succeeds.

    Not a projection from a candidate. arm_a removes every arithmetic operation
    from the inner loop while loading every byte production loads, so its rate
    is the measured floor on time for these kernels at these bytes.
    """
    d = _budget()
    cur = float(d["decode_wall_ms_per_token"])
    rows = {r["organ"]: float(r["gpu_ms"]) for r in d["organs"]["rows"]}
    parts = []
    total = 0.0
    for codec, (ratio, organs, source) in ARM_A_RATIO.items():
        missing = [o for o in organs if o not in rows]
        if missing:
            raise GapRefused(f"{codec}: budget has no rows for {missing}")
        ms = sum(rows[o] for o in organs)
        saved = ms - ms / ratio
        total += saved
        parts.append({
            "codec": codec,
            "organs": list(organs),
            "organ_ms": round(ms, 4),
            "arm_a_over_production": ratio,
            "ms_saved_at_perfect_removal": round(saved, 4),
            "source": source,
        })
    after = cur - total
    return {
        "parts": parts,
        "total_ms_at_perfect_removal": round(total, 4),
        "token_ms_after": round(after, 4),
        "tps_after": round(1000.0 / after, 3),
        "reaches_60": after <= 1000.0 / 60.0,
        "still_short_of_60_by_ms": round(after - 1000.0 / 60.0, 4),
        "verdict": (
            "THE DECODE-ARITHMETIC SCHOOL CANNOT REACH 60 EVEN IF IT PERFECTLY "
            f"SUCCEEDS. Removing every arithmetic operation from every q2 and q4 "
            f"matvec reaches {round(1000.0/after, 1)} TPS. 60 needs "
            f"{round(after - 1000.0/60.0, 2)} ms from a DIFFERENT class - fewer "
            "bytes, less host time, or work removed rather than made cheaper."
        ),
        "what_this_does_not_say": (
            "it does not say the arithmetic school is not worth working. It is "
            "the largest single block on the board and the bitcast candidates "
            "have already taken a measured 3.854 ms of it. It says the school "
            "must not be the ONLY thing running if 60 is the target."
        ),
        "assumption": (
            "arm_a's ratio measured on one representative projection per codec "
            "is applied to every organ that runs that codec. Organs differ in "
            "shape, so this is an estimate at the organ level even though each "
            "ratio is measured. It would take a per-organ arm_a to tighten."
        ),
    }


def build() -> dict[str, Any]:
    lv = live()
    cps = checkpoints()
    sixty = next(c for c in cps if c["tps"] == 60.0)
    seventyone = next(c for c in cps if c["tps"] == 71.0)
    ranked = ranked_experiments()
    material = [r for r in ranked if r["material"] and r["status"] not in ("CLOSED",)]
    return {
        "obligation": "G095",
        "authority": "S026 §0, §1, §11, §12, §48",
        "live": lv,
        "checkpoints": cps,
        "gap_to_60_ms": sixty["ms_to_remove_from_live"],
        "gap_to_71_ms": seventyone["ms_to_remove_from_live"],
        "organ_win_table": organ_win_table(),
        "three_dominant_kernels": three_dominant(),
        "ranked_experiments": ranked,
        "material_open_experiments": material,
        "sum_of_material_open_ms": round(
            sum(r["max_ms_removable"] for r in material), 4),
        "does_the_open_set_reach_60": (
            sum(r["max_ms_removable"] for r in material) >= sixty["ms_to_remove_from_live"]
        ),
        "arithmetic_ceiling": arithmetic_ceiling(),
        "escalation_clock": escalation_clock(),
        "honest_reading": (
            "the material open experiments are bandwidth-recovery bounds - what "
            "each organ would cost if it ran at a rate already demonstrated "
            "elsewhere on this machine. They are CEILINGS, not candidates: no "
            "mechanism that reaches them is on record. Summing them and "
            "comparing to the gap says whether the gap is closable IN "
            "PRINCIPLE by the open set, not whether it is closable today."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        # This ledger MEASURES NOTHING. Every hardware number in it is read out
        # of RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json, which carries its own
        # provenance. Stamping this build time as a measurement time would be a
        # fabricated provenance, so it is marked a retrofit of that receipt.
        print(write_measured_receipt(
            REPO / "receipts" / "future" / RECEIPT_NAME, doc, RECORDED_BY,
            provenance=measurement_provenance(
                lock_held=False, lane="derived", retrofit=True),
        ))
        return 0
    print(json.dumps({k: doc[k] for k in (
        "live", "gap_to_60_ms", "gap_to_71_ms", "three_dominant_kernels",
        "material_open_experiments", "sum_of_material_open_ms",
        "does_the_open_set_reach_60", "escalation_clock")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
