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
    "moe_router": {"moe_router"},
    "embed": {"embed"},
    "lm_head": {"lm_head"},
    "rmsnorm": {"rmsnorm"},
}

# §71 is a claim about (organ, REPRESENTATION), not about organ names. The first version
# of this planner matched on organ alone and reported moe_expert COVERED by 18 kernels --
# but the planner seeds q2_affine for that organ and every one of those kernels executes
# binary_group, hgravs01_factored or uniform_q4_group. Matching by name would have let a
# representation be scored against a kernel that cannot execute it, which is exactly what
# §71 forbids.
# Read from the two vocabularies that actually exist on disk, not invented:
#   kernel representation_identity: q4_control, binary, binary_sparse_residual,
#     q2_affine, shared_basis, ternary, binary_group, hgravs01_factored,
#     uniform_q4_group, unknown
#   planner seeded family: conventional_low_bit, q2_affine, binary_sparse_residual,
#     ternary, leftover_f32
REPRESENTATION_EXECUTES = {
    "conventional_low_bit": {"q4_control", "uniform_q4_group", "uniform_qn_group"},
    "q2_affine": {"q2_affine"},
    # NOT {binary, binary_group}: a plain binary kernel has no residual path and cannot
    # execute a binary-plus-sparse-residual representation. Conflating them is the same
    # looseness that made moe_expert look covered by organ name alone.
    "binary_sparse_residual": {"binary_sparse_residual"},
    "binary_group": {"binary", "binary_group"},
    "ternary": {"ternary"},
    # an f32 passthrough organ is not executed by a quantized GEMV kernel at all
    "leftover_f32": set(),
}
NEEDS_NO_GEMV = {"leftover_f32"}

# leftover_f32 is a real answer for a passthrough organ like a norm, and a NON-ANSWER for
# an organ that must execute a matvec: selecting it for moe_expert would report the
# model's largest organ (18,432 tensors) as COVERED by declining to compress it at all,
# which is a 4x expansion dressed as a plan.
GEMV_ORGANS = {"embed", "lm_head", "moe_expert", "moe_router", "gqa_attention"}

REPRESENTATION_INDEPENDENT = {"topk_select", "norm_only"}


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

    rehearsal = json.load(open(RH / "QWEN_TRANSFER_REHEARSAL.json"))
    # every seeded family in score order. Taking only the top-scored one made the planner
    # report a gap whenever the FIRST choice lacked a kernel, when the planner's actual
    # job is to select the best family it can competently execute.
    seeded = {}
    for r in rehearsal["plan"]["organ_plan"]:
        seeds = sorted(r.get("seeded_representations") or [],
                       key=lambda s: -s.get("score", 0))
        seeded[r["organ"]] = [{"family": s["family"], "score": s.get("score")}
                              for s in seeds]

    def kind_of(v):
        if isinstance(v, dict):
            return v.get("kind", "PRESENT_NO_KIND")
        return "ABSENT" if v in (None, "", [], {}) else "PRESENT_BARE"

    plan, covered, gaps = [], 0, []
    for organ in sorted(organs):
        organ_kernels = []
        for ko in SERVES.get(organ, set()):
            organ_kernels.extend(by_organ.get(ko, []))

        considered, selected = [], None
        for cand_family in seeded.get(organ, []):
            fam = cand_family["family"]
            can = REPRESENTATION_EXECUTES.get(fam, set())
            # A representation-INDEPENDENT kernel (a top-k select over logits) supports
            # an organ but cannot execute it: it touches no weights. Counting it alone
            # marked moe_router COVERED for q2_affine when the only q2_affine-capable
            # thing present was a selector and every router MATVEC is binary_group.
            weight_bearing = [k for k in organ_kernels
                              if k["representation_identity"] in can]
            supporting = [k for k in organ_kernels
                          if k["representation_identity"] in REPRESENTATION_INDEPENDENT]
            competent = (weight_bearing + supporting) if weight_bearing else []
            if organ not in GEMV_ORGANS:
                competent = weight_bearing + supporting
            row = {"family": fam, "score": cand_family["score"],
                   "n_competent_kernels": len(competent),
                   "n_weight_bearing": len(weight_bearing),
                   "n_supporting_representation_independent": len(supporting),
                   "needs_no_gemv_kernel": fam in NEEDS_NO_GEMV,
                   "kernels": [k["kernel_identity"] for k in competent][:8]}
            considered.append(row)
            passthrough_ok = fam in NEEDS_NO_GEMV and organ not in GEMV_ORGANS
            row["passthrough_rejected_for_gemv_organ"] = (
                fam in NEEDS_NO_GEMV and organ in GEMV_ORGANS)
            if selected is None and (competent or passthrough_ok):
                selected = {**row, "why": ("f32 passthrough on a non-GEMV organ: no "
                                           "kernel is required"
                                           if fam in NEEDS_NO_GEMV else
                                           f"highest-scoring seeded family with a "
                                           f"competent kernel")}

        downgraded = bool(selected and considered and
                          selected["family"] != considered[0]["family"])
        r = {
            "organ": organ,
            "seeded_families_in_score_order": considered,
            "selected_representation": selected["family"] if selected else None,
            "selection_reason": selected["why"] if selected else None,
            "n_kernels_for_organ": len(organ_kernels),
            "n_competent_kernels": selected["n_competent_kernels"] if selected else 0,
            "downgraded_from_top_seed": downgraded,
            "status": ("COVERED" if selected else
                       "REPRESENTATION_MISMATCH" if organ_kernels else "NO_KERNEL"),
            "may_be_evaluated": bool(selected),
        }
        if downgraded:
            r["downgrade_note"] = (
                f"the top seeded family {considered[0]['family']!r} has no competent "
                f"kernel; the planner selected {selected['family']!r} instead. §71: a "
                f"representation with no competent kernel may not be evaluated.")
        if not selected and organ_kernels:
            r["mismatch"] = (
                f"{len(organ_kernels)} kernel(s) exist for {organ!r} but none executes "
                f"any seeded family; they execute "
                f"{sorted({k['representation_identity'] for k in organ_kernels})}")
        if selected and selected["n_competent_kernels"]:
            ks = [k for k in organ_kernels
                  if k["kernel_identity"] in set(selected["kernels"])]
            unq = [k for k in ks if kind_of(k.get("parity")) == "ABSENT"]
            r["qualification"] = (
                f"COMPETENT_BUT_UNQUALIFIED: {len(unq)} of {len(ks)} have parity ABSENT"
                if unq else f"QUALIFIED: all {len(ks)} carry a parity result")
        if selected:
            covered += 1
        else:
            gaps.append(organ)
        plan.append(r)

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
        top = (r["seeded_families_in_score_order"] or [{}])[0].get("family")
        mark = "  <-- DOWNGRADED" if r.get("downgraded_from_top_seed") else ""
        print(f"  {r['organ']:15s} top={str(top):22s} "
              f"selected={str(r['selected_representation']):22s} "
              f"{r['status']:23s} competent={r['n_competent_kernels']:2d}{mark}")
    print(f"\n{out['honest_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
