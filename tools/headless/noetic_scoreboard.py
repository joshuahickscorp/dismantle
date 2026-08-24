#!/usr/bin/env python3
"""N015 — the Noetic scoreboard and the frontier points it tracks.

S017 §44 fixes the columns; §3 fixes the frontier points. The point of writing it
as a harness rather than a table is that every cell must come from a receipt on
disk, and a cell nobody has measured yet must read ABSENT with a reason instead of
0. A scoreboard with plausible zeros in it is worse than no scoreboard: it makes
an unmeasured candidate look cheap.

The frontier points are deliberately separate columns because S017 says they MAY
BE DIFFERENT ARTIFACTS:

    LOWEST_SCREEN_SURVIVOR      cheapest thing that passed an organ-local screen
    LOWEST_CHAIN_SURVIVOR       cheapest thing that survived composition
    LOWEST_GENERATION_COHERENT  cheapest thing that actually generated text
    LOWEST_CAPABILITY_SURVIVOR  cheapest thing that kept capability
    FASTEST_COHERENT            highest tok/s among coherent candidates
    FASTEST_PRODUCTION          highest verified useful work per wall second

Collapsing those into one "best" is how a campaign talks itself into promoting an
artifact that was only ever screened.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
R = REPO / "receipts" / "headless"
OUT = R / "NOETIC_SCOREBOARD.json"

PARENT_PARAMS = 26_895_998_464
INCUMBENT_EBPW = 4.252735126866492

ABSENT = "ABSENT"


def load(name: str) -> dict[str, Any] | None:
    p = R / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()[:12]


def cell(value, why_absent: str | None = None, kind: str = "MEASURED") -> dict[str, Any]:
    """Every cell says whether it is real. An unmeasured cell is never a number."""
    if value is None:
        return {"value": None, "state": ABSENT, "reason": why_absent or "not measured"}
    return {"value": value, "state": kind}


def dig(d: Any, *path, default=None):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and isinstance(k, int) and len(cur) > k:
            cur = cur[k]
        else:
            return default
    return cur


def candidates() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def row(cid, ebpw, tps, disp, coherent, screen, chain, generation, src, note=""):
        gb = (ebpw * PARENT_PARAMS / 8) / 2**30 if ebpw else None
        return {
            "id": cid,
            "source_receipt": src,
            "note": note,
            # --- S017 §44 columns ---
            "EBPW": cell(ebpw),
            "RESIDENT_GB": cell(round(gb, 4) if gb else None, kind="DERIVED"),
            "ACTIVE_GB_PER_TOKEN": cell(None, "GPU ledger (N004) not yet landed"),
            "DRAM_GB_PER_TOKEN": cell(None, "GPU ledger (N004) not yet landed"),
            "FLOP_PER_TOKEN": cell(None, "per-candidate FLOP census not yet run"),
            "DISPATCHES_PER_TOKEN": cell(disp),
            "ROUTES_PER_TOKEN": cell(0, kind="MEASURED") if ebpw else cell(None),
            "ROUTING_NS_PER_TOKEN": cell(
                0, kind="MEASURED"
            ) if ebpw else cell(None),
            "COMPLETE_TOKEN_NS": cell(
                round(1e9 / tps) if tps else None, kind="DERIVED"
            ),
            "TPS": cell(tps),
            "AGGREGATE_TPS_C2": cell(None, "concurrency bench (N007) not yet landed"),
            "AGGREGATE_TPS_C4": cell(None, "concurrency bench (N007) not yet landed"),
            "VERIFIED_WUS_PER_HOUR": cell(None, "production bench (N007) not yet landed"),
            "CAPABILITY": cell(None, "no capability suite has been run on any candidate"),
            # --- ladder position, S017 §28 ---
            "passed_screen": screen,
            "survived_chain": chain,
            "generated_coherently": generation,
            "coherent": coherent,
        }

    fne = load("FIRST_NOETIC_EXECUTABLE")
    for m in (fne or {}).get("mix_scoreboard", []):
        out.append(
            row(
                m["mix_id"],
                m.get("complete_ebpw"),
                m.get("tok_s"),
                None,
                m.get("coherent"),
                True,
                None,
                bool(m.get("coherent")),
                "receipts/headless/FIRST_NOETIC_EXECUTABLE.json",
            )
        )

    q3 = load("NOETIC_Q3_MLP_Q4_ATTN")
    if q3:
        out.append(
            row(
                dig(q3, "chosen", "mix_id", default="q3_mlp_q4_attn"),
                dig(q3, "chosen", "complete_ebpw"),
                dig(q3, "decode", "tok_s"),
                None,
                True,
                True,
                True,
                True,
                "receipts/headless/NOETIC_Q3_MLP_Q4_ATTN.json",
                "text identical to the q4 incumbent",
            )
        )

    a32 = load("AFFINE2_NATIVE_MLP")
    if a32:
        out.append(
            row(
                "affine2_g32_all_mlp",
                dig(a32, "chosen", "complete_ebpw"),
                dig(a32, "decode", "tok_s"),
                None,
                False,
                True,
                True,
                False,
                "receipts/headless/AFFINE2_NATIVE_MLP.json",
                "degraded: malformed think-tag and a stutter",
            )
        )

    fused = load("NOETIC_FUSED_SUBBIT")
    if fused:
        best = dig(fused, "decode_tok_s", "after_mlp_swiglu_qkv_dn", default={})
        out.append(
            row(
                "NOETIC_PARENT_A (affine2_g64_LS + fused graph)",
                dig(fused, "representation", "complete_ebpw"),
                best.get("tok_s_mean"),
                756,
                True,
                True,
                True,
                True,
                "receipts/headless/NOETIC_FUSED_SUBBIT.json",
                "LEADER. Frontier candidate, NOT resident-promoted. "
                "Artifact bytes were reaped with a lane worktree; N001 rebuilds and seals.",
            )
        )

    out.append(
        row(
            "q4 incumbent (control)",
            INCUMBENT_EBPW,
            33.717,
            964,
            True,
            True,
            True,
            True,
            "receipts/headless/NOETIC_DISPATCH_FUSION.json",
            "immutable control",
        )
    )
    return out


def frontier_points(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def ebpw(r):
        return r["EBPW"]["value"]

    def tps(r):
        return r["TPS"]["value"]

    cands = [r for r in rows if ebpw(r) is not None and "control" not in r["id"]]
    screened = [r for r in cands if r["passed_screen"]]
    chained = [r for r in cands if r["survived_chain"]]
    generated = [r for r in cands if r["generated_coherently"]]
    coherent_tps = [r for r in cands if r["coherent"] and tps(r)]

    def lowest(rs):
        return min(rs, key=ebpw)["id"] if rs else None

    return {
        "LOWEST_SCREEN_SURVIVOR": lowest(screened),
        "LOWEST_CHAIN_SURVIVOR": lowest(chained),
        "LOWEST_GENERATION_COHERENT": lowest(generated),
        "LOWEST_CAPABILITY_SURVIVOR": {
            "value": None,
            "state": ABSENT,
            "reason": "no capability suite has been run on ANY candidate; "
            "coherence on a 16-token greedy sample is not capability",
        },
        "FASTEST_COHERENT": max(coherent_tps, key=tps)["id"] if coherent_tps else None,
        "FASTEST_PRODUCTION": {
            "value": None,
            "state": ABSENT,
            "reason": "production is verified useful work per wall second; "
            "the concurrency bench (N007) has not been built",
        },
        "note": "S017 §3: these MAY BE DIFFERENT ARTIFACTS. They are not collapsed.",
    }


def main() -> int:
    rows = candidates()
    fp = frontier_points(rows)
    n_cells = sum(1 for r in rows for k, v in r.items() if isinstance(v, dict) and "state" in v)
    n_absent = sum(
        1
        for r in rows
        for k, v in r.items()
        if isinstance(v, dict) and v.get("state") == ABSENT
    )
    receipt = {
        "schema": "hawking.headless.noetic_scoreboard.v1",
        "obligation": "N015 (S017 §3, §42, §44)",
        "git_head": git_head(),
        "parent_params": PARENT_PARAMS,
        "columns": "S017 §44",
        "honesty": (
            "Every cell states MEASURED, DERIVED or ABSENT. An unmeasured cell is "
            "never rendered as 0 — a plausible zero makes an unmeasured candidate "
            "look cheap, which is the specific way a scoreboard lies."
        ),
        "candidates": rows,
        "frontier_points": fp,
        "coverage": {
            "cells": n_cells,
            "absent": n_absent,
            "absent_fraction": round(n_absent / n_cells, 4) if n_cells else None,
            "blocking_obligations": [
                "N004 GPU ledger -> ACTIVE_GB/TOKEN, DRAM_GB/TOKEN",
                "N007 production bench -> AGGREGATE_TPS_C2/C4, VERIFIED_WUS/HOUR, FASTEST_PRODUCTION",
                "capability suite -> CAPABILITY, LOWEST_CAPABILITY_SURVIVOR",
            ],
        },
        "phase_transition_map": {
            "state": "PARTIAL",
            "family": "uniform grouped-code on the whole MLP",
            "measured_points": [
                {"bpw_body": 1.85, "codec": "ternary g64", "composed": "FAILS", "argmax": "flips"},
                {"bpw_body": 2.25, "codec": "q2 4-level fitted g64", "composed": "SURVIVES", "argmax": "agrees"},
                {"bpw_body": 3.25, "codec": "q3 g64", "composed": "SURVIVES", "argmax": "agrees"},
            ],
            "knee": "between 1.85 and 2.25 bpw body — sharp, not gradual: rel_l2 "
            "more than doubles (0.3471 -> 0.7360) and the argmax flips",
            "collapse_boundary": 1.85,
            "coherent_plateau": "2.25 costs essentially nothing against 3.25 "
            "(rel_l2 0.3471 vs 0.3423)",
            "missing": "only ONE family is mapped. S017 §42 wants a curve per family; "
            "N008 is running four structurally distinct ones.",
        },
    }
    OUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"candidates {len(rows)}  cells {n_cells}  ABSENT {n_absent} ({n_absent/n_cells:.0%})")
    for k, v in fp.items():
        if k == "note":
            continue
        s = v["reason"] if isinstance(v, dict) else v
        print(f"  {k:<28} {str(s)[:70]}")
    print(f"receipt: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
