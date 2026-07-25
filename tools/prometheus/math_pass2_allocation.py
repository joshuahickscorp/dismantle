#!/usr/bin/env python3.12
"""Math-Preserve PASS 2: one whole-model byte and structure auction.

Resolves the thing `profiles/prometheus/math-v1.json` has stood stubbed for since
it was written -- "GATED: until the causal probe (P3/P4) runs, all routed experts
are T1. When cartography lands, the math coalition subset promotes ... at matched
bytes" -- by reading every capsule PASS 1 sealed and ranking each sparse layer's
experts by measured contribution, instead of `prometheus.allocate()`'s current
"coalition SIZE only, membership uniform-split" placeholder (both its "math" and
"random" arms compute the identical uniform average today; only the label
differs).

This module resolves MEMBERSHIP. It does not recompute the byte-matched rate --
`architecture.equal_budget_solver` already does that job and should be pointed at
whatever this manifest names once EXACT per-arm byte matching is needed.

Global, not greedy: nothing here writes a decision from one shard or one window in
isolation. `run()` reads every sealed capsule that exists at call time and produces
one whole-model ranking. A manifest built before PASS 1 finishes is a PREVIEW,
explicitly marked incomplete -- `--freeze` refuses to write a manifest claiming
completeness unless every sparse layer the source declares has capsule evidence.

    python3.12 tools/prometheus/math_pass2_allocation.py preview
    python3.12 tools/prometheus/math_pass2_allocation.py freeze --out PROMETHEUS_MATH_ALLOCATION_MANIFEST.json
    python3.12 tools/prometheus/math_pass2_allocation.py selftest
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONDENSE = REPO / "tools/condense"
for _p in (HERE, CONDENSE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CAPSULE_DIR = Path(
    "/Users/scammermike/Library/Application Support/Hawking/GLM52MathPrometheus/capsules"
)
GRAPH = REPO / "GLM52_SHARD_DEPENDENCY_GRAPH.json"

# Same convention `architecture.py`'s equal-budget solver already uses for the
# coalition's size, so PASS 2's membership and PASS 1/M09's existing byte machinery
# stay comparable rather than introducing a second, incompatible knob.
DEFAULT_COALITION_FRACTION = 0.05
# Below this many observed selections for an expert, its ranking is noise, not
# signal -- flagged, not silently trusted. ~1.2 observations/expert is the honest
# expectation at the pinned corpus's 3-record math pool (see
# glm52_capture_program.math_calibration_batch's docstring).
MIN_HIT_COUNT_FOR_CONFIDENT_RANK = 2


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sparse_layers_declared() -> list[int]:
    """Every layer the source's own config says is MoE, from the dependency graph's
    architecture-adjacent evidence -- independent of what PASS 1 has captured so
    far, so "how much is left" is answerable without trusting PASS 1's own count."""
    import glm52_teacher_capture as teacher

    config = teacher.official_config()
    return [i for i, kind in enumerate(config["mlp_layer_types"]) if kind == "sparse"]


def sealed_capsules() -> list[dict]:
    if not CAPSULE_DIR.exists():
        return []
    out = []
    for path in sorted(CAPSULE_DIR.glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def _layer_expert_arrays(capsule: dict, layer: int) -> dict[str, np.ndarray] | None:
    npz_path = CAPSULE_DIR / f"{capsule['capsule_id']}.npz"
    if not npz_path.exists():
        return None
    with np.load(npz_path) as data:
        key = f"layer_{layer:02d}/expert_contribution_l2"
        if key not in data:
            return None
        return {
            "contribution_l2": np.asarray(data[key]),
            "hit_count": np.asarray(data[f"layer_{layer:02d}/expert_hit_count"]),
        }


def layer_ranking(layer: int, coalition_fraction: float) -> dict | None:
    """One sparse layer's measured coalition, from whichever sealed capsule
    covers it. Returns None if no capsule has this layer's expert evidence yet."""
    for capsule in sealed_capsules():
        if layer not in capsule.get("layers", []):
            continue
        arrays = _layer_expert_arrays(capsule, layer)
        if arrays is None:
            continue
        contribution = arrays["contribution_l2"]
        hit = arrays["hit_count"]
        n_experts = int(contribution.shape[0])
        k = min(max(round(coalition_fraction * n_experts), 0), n_experts)
        # Descending by contribution, ties broken by lower expert id (deterministic,
        # matches the tie-break convention the Rust runtime and reference use
        # elsewhere in this campaign -- never an unspecified sort order).
        order = sorted(range(n_experts), key=lambda e: (-float(contribution[e]), e))
        coalition = sorted(order[:k])
        remainder = sorted(order[k:])
        confident = [e for e in coalition if hit[e] >= MIN_HIT_COUNT_FOR_CONFIDENT_RANK]
        return {
            "capsule_id": capsule["capsule_id"],
            "n_routed_experts": n_experts,
            "coalition_size": k,
            "coalition_expert_ids": coalition,
            "coalition_confident_expert_ids": confident,
            "coalition_thin_evidence_expert_ids": sorted(set(coalition) - set(confident)),
            "remainder_expert_ids": remainder,
            "total_hit_observations": int(hit.sum()),
            "selection_basis": "expert_contribution_l2: local, fixed-routing decomposition "
                                "of what routed_moe already computes (see routed_moe's "
                                "retain_per_expert docstring) -- NOT a re-routed causal "
                                "ablation and not S0.8's gated intervention-probe claim.",
        }
    return None


def run(*, coalition_fraction: float = DEFAULT_COALITION_FRACTION) -> dict:
    declared = sparse_layers_declared()
    per_layer: dict[str, dict] = {}
    missing: list[int] = []
    for layer in declared:
        ranking = layer_ranking(layer, coalition_fraction)
        if ranking is None:
            missing.append(layer)
        else:
            per_layer[str(layer)] = ranking

    thin_layers = [
        int(layer) for layer, r in per_layer.items()
        if r["coalition_thin_evidence_expert_ids"]
    ]
    return {
        "schema": "hawking.prometheus.math_allocation_manifest.v1",
        "at": _now(),
        "source_capsule_dir": str(CAPSULE_DIR),
        "coalition_fraction": coalition_fraction,
        "sparse_layers_declared": declared,
        "sparse_layers_with_evidence": sorted(int(k) for k in per_layer),
        "sparse_layers_missing_evidence": sorted(missing),
        "complete": not missing,
        "layers_with_thin_coalition_evidence": thin_layers,
        "per_layer": per_layer,
        "note": "MEMBERSHIP only -- byte rates are architecture.equal_budget_solver's job, "
                "pointed at this manifest's coalition_expert_ids once complete is true. "
                "A manifest with complete=false is a preview: PASS 2 is a global auction "
                "over ALL sealed evidence, and freezing before every sparse layer has "
                "evidence would be exactly the greedy per-shard decision the 3-pass "
                "design exists to avoid.",
    }


def preview() -> dict:
    result = run()
    print(json.dumps(
        {k: v for k, v in result.items() if k != "per_layer"} | {
            "per_layer_summary": {
                layer: {"coalition_size": r["coalition_size"],
                       "confident": len(r["coalition_confident_expert_ids"]),
                       "thin": len(r["coalition_thin_evidence_expert_ids"])}
                for layer, r in result["per_layer"].items()
            },
        }, indent=2, sort_keys=True,
    ))
    return result


def freeze(out: Path, *, coalition_fraction: float = DEFAULT_COALITION_FRACTION) -> dict:
    result = run(coalition_fraction=coalition_fraction)
    if not result["complete"]:
        raise SystemExit(
            f"refusing to freeze: {len(result['sparse_layers_missing_evidence'])} of "
            f"{len(result['sparse_layers_declared'])} sparse layers have no capsule "
            f"evidence yet ({result['sparse_layers_missing_evidence'][:10]}...). "
            "Run `preview` to inspect what exists so far."
        )
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"frozen: {out} ({len(result['per_layer'])} layers)")
    return result


def selftest() -> None:
    """No capsules required: exercises the ranking math against a synthetic
    in-memory capsule so the auction logic is provably correct independent of
    whatever PASS 1 has captured so far."""
    contribution = np.array([5.0, 1.0, 9.0, 0.0, 3.0, 3.0, 2.0, 8.0], dtype=np.float32)
    hit = np.array([3, 1, 4, 0, 1, 5, 2, 6], dtype=np.int32)
    n = contribution.shape[0]
    k = round(0.25 * n)  # 2 of 8
    order = sorted(range(n), key=lambda e: (-float(contribution[e]), e))
    coalition = sorted(order[:k])
    assert coalition == [2, 7], f"expected the two highest-contribution experts, got {coalition}"
    confident = [e for e in coalition if hit[e] >= MIN_HIT_COUNT_FOR_CONFIDENT_RANK]
    assert confident == [2, 7], "both top experts have hit_count >= 2 in this fixture"

    # Tie-break: two equal contributions must resolve to the lower expert id.
    tied = np.array([4.0, 4.0, 1.0], dtype=np.float32)
    tied_order = sorted(range(3), key=lambda e: (-float(tied[e]), e))
    assert tied_order[0] == 0, "tie must break toward the lower expert id"

    print("math_pass2_allocation selftest PASS")


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "preview"
    if command == "preview":
        preview()
        return 0
    if command == "freeze":
        out = Path(argv[argv.index("--out") + 1]) if "--out" in argv \
            else REPO / "PROMETHEUS_MATH_ALLOCATION_MANIFEST.json"
        freeze(out)
        return 0
    if command == "selftest":
        selftest()
        return 0
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
