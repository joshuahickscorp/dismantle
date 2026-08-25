#!/usr/bin/env python3
"""G023 unblock step 2: actually run KernelPlanner for model #2.

Registering kernels is not planning. The KernelPlanner stage has one job: for every
organ in the specimen's organ graph, name a kernel competent to execute the planned
representation, or record that none exists. §71 forbids evaluating a representation with
an incompetent kernel, and that rule cannot be honoured without this mapping.

The output is deliberately allowed to be mostly gaps. A planner that reports full
coverage by inventing kernels is worse than one that reports where the holes are.
"""
import json, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RH = REPO / "receipts/headless"
SPECIMEN = "Qwen/Qwen3-30B-A3B"

# which kernel organ_identity can serve which organ-graph organ
SERVES = {
    "moe_expert": {"moe_expert", "moe_expert_gate_up"},
    "gqa_attention": {"gqa_attention"},
    "moe_router": set(),
    "embed": set(),
    "lm_head": set(),
    "rmsnorm": set(),
}


def main():
    census = json.load(open(RH / "ARCHITECTURE_RECOGNIZER.json"))
    spec = None
    for key in ("specimens", "heldout_specimens"):
        for s in census.get(key, []) or []:
            if (s.get("result") or {}).get("repo") == SPECIMEN:
                spec = s
    if spec is None:
        raise SystemExit(f"{SPECIMEN} not in the architecture census")
    organs = [o["organ"] for o in spec["result"]["organs"]
              if o["status"] in ("KNOWN", "DECLARED_UNMEASURED")]

    lib = json.load(open(RH / "KERNEL_LIBRARY.json"))
    by_organ = {}
    for k in lib["kernels"]:
        by_organ.setdefault(k["organ_identity"], []).append(k)

    plan, covered, gaps = [], 0, []
    for organ in sorted(organs):
        cands = []
        for ko in SERVES.get(organ, set()):
            for k in by_organ.get(ko, []):
                # the library holds two field shapes: the {kind, value} envelope and
                # a bare value from older entries. Assuming the envelope crashed here.
                def kind_of(v):
                    if isinstance(v, dict):
                        return v.get("kind", "PRESENT_NO_KIND")
                    return "ABSENT" if v in (None, "", [], {}) else "PRESENT_BARE"
                cands.append({
                    "kernel_identity": k["kernel_identity"],
                    "representation_identity": k["representation_identity"],
                    "parity": kind_of(k.get("parity")),
                    "measurements": kind_of(k.get("measurements")),
                })
        row = {
            "organ": organ,
            "n_competent_kernels": len(cands),
            "kernels": cands,
            "status": "COVERED" if cands else "NO_KERNEL",
            "may_be_evaluated": bool(cands),
        }
        if cands:
            covered += 1
            # §71: competence is not the same as qualification
            unqualified = [c for c in cands if c["parity"] == "ABSENT"]
            row["n_unqualified"] = len(unqualified)
            row["qualification"] = (
                f"COMPETENT_BUT_UNQUALIFIED: {len(unqualified)} of {len(cands)} "
                f"candidates have parity ABSENT, so a representation may be PLANNED "
                f"against them but must not be SCORED against them until parity is run"
                if unqualified else
                f"QUALIFIED: all {len(cands)} candidates carry a parity result")
        else:
            gaps.append(organ)
        plan.append(row)

    out = {
        "schema": "hawking.odyssey.kernel_planner_model2.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_by": "tools/odyssey/kernel_planner_model2.py",
        "obligation": "G023 — KernelPlanner stage, run for model #2",
        "hand_authored": False,
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "specimen": {"repo": SPECIMEN,
                     "revision": spec["result"].get("revision"),
                     "organ_graph_source": "receipts/headless/ARCHITECTURE_RECOGNIZER.json"},
        "organ_plan": plan,
        "n_organs": len(organs),
        "n_covered": covered,
        "n_gaps": len(gaps),
        "gaps": gaps,
        "section_71_rule": {
            "rule": "representations are never evaluated with incompetent kernels",
            "how_it_is_honoured": "an organ with no competent kernel is marked NO_KERNEL "
                                  "and may_be_evaluated=false, so no representation can "
                                  "be scored on it. Organs WITH kernels are marked "
                                  "COMPETENT_BUT_UNQUALIFIED because every registered "
                                  "expert kernel has parity and measurements ABSENT.",
            "previously": "this rule could not be checked at all for model #2, because "
                          "the library contained no kernel for any of its organs",
        },
        "stage_status": ("RAN_WITH_GAPS" if gaps else "RAN_COMPLETE"),
        "honest_summary": f"{covered} of {len(organs)} organs have a competent kernel. "
                          f"The remaining {len(gaps)} ({', '.join(gaps)}) have none, so "
                          f"model #2 still cannot be compiled end to end. What changed is "
                          f"that the gap is now enumerated per organ instead of being "
                          f"hidden behind a stage recorded as AUTOMATIC.",
    }
    out["pass"] = covered > 0
    p = RH / "KERNEL_PLANNER_MODEL2.json"
    p.write_text(json.dumps(out, indent=1))

    for r in plan:
        print(f"  {r['organ']:22s} {r['status']:10s} kernels={r['n_competent_kernels']:2d}"
              f"  evaluable={r['may_be_evaluated']}")
    print(f"\n{out['honest_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
