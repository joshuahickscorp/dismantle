"""Build ROADMAP_STATE.json from disk truth.

The ERA MAP is judgement and is written here in the open. Everything else --
obligation status, receipt counts, test counts, commit -- is DERIVED from disk,
because a ledger that lets a human retype a status is a ledger that drifts.
"""
import json, pathlib, re, subprocess, sys

HAWKING = pathlib.Path(__file__).resolve().parent.parent
GOAL = pathlib.Path.home() / ".claude/ultragoal/hawking-odyssey-maxx-ascension/GOAL.md"

# Judgement, stated in the open. An obligation lands where its EVIDENCE lands,
# not where its title sounds like it belongs.
ERA_MAP = {
    "I-A_AGENTOS_HCLI":   ["G013", "G014", "G015", "G030", "G031"],
    "I-B_DOCTOR":         ["G016", "G017", "G018", "G019", "G020", "G021", "G035"],
    "I-C_GRAVITY_NOETIC": ["G001", "G002", "G003", "G004", "G005", "G006", "G022",
                           "G023", "G032", "G033", "G034", "G036", "G037", "G038",
                           "G040", "G042"],
    "I-D_ACCELERATOR":    ["G043", "G044", "G045", "G046", "G047", "G049", "G055", "G058"],
    "I-E_ODYSSEY_I":      ["G007", "G008", "G009", "G010", "G011", "G012", "G024",
                           "G025", "G026", "G027", "G028", "G029", "G039", "G041",
                           "G048", "G056"],
    "II-E_GREEN_MACHINE": ["G057"],
    "IV-A_FUSION":        ["G053", "G054"],
    "IV-B_HMF_HGVAS":     ["G050"],
    "IV-D_EGPU":          ["G051", "G052"],
}
ERA_I = [k for k in ERA_MAP if k.startswith("I-")]

# S015 §II: a civilization is not complete because files exist. These NINE are the
# categories completion is weighted by; a percentage is (satisfied / 9) and nothing
# else, so it can never be computed from a file count.
EVIDENCE_CATEGORIES = ["artifact", "runtime", "adversarial_verification",
                       "negative_control", "failure_recovery", "durable_receipt",
                       "integration", "measured_useful_work", "named_boundaries"]

# Assessed against receipts on disk. Each False is a REAL gap, not a formality.
# GATES are not obligations. I-D has receipts in all nine categories and EVERY gate
# open -- the first build of this ledger reported it 100% and that was the exact
# inflation S015 §II forbids. Evidence coverage is a CEILING, never the score.
OPEN_GATES = {
  "I-A_AGENTOS_HCLI": ["S016 civilization-grade scheduler: Era/Civilization/Program/Gate "
                       "as first-class scheduler concepts -- NOT_STARTED"],
  "I-B_DOCTOR": ["gate: on an UNSEEN model, reduce a huge hypothesis space to a small "
                 "high-information experimental set and explain why -- not run"],
  "I-C_GRAVITY_NOETIC": ["G023 Noetic compiler pipeline",
                         "real-weight execution gate",
                         "EBPW namespace separation not yet permanent in code"],
  "I-D_ACCELERATOR": ["C2M T3: several real open CUDA codebases -- NOT CLAIMED",
                      "P2 CUDA differential -- blocked on NVIDIA hardware",
                      "AIR multi-backend (G058) -- Metal only"],
  "I-E_ODYSSEY_I": ["model #2 is not a Noetic executable",
                    "no real weights have executed (blocked on FAST_LOCAL_STORAGE)",
                    "G056 lake fast-forward in flight"],
}

EVIDENCE = {
  "I-A_AGENTOS_HCLI": dict(artifact=1, runtime=1, adversarial_verification=1,
      negative_control=1, failure_recovery=1, durable_receipt=1, integration=1,
      measured_useful_work=1, named_boundaries=1,
      note="the OLD gate is met. S016's civilization-grade scheduler (Era/Civilization/"
           "Program/Gate as first-class scheduler concepts) is NOT_STARTED and is NOT "
           "counted here -- it is a NEW gate, tracked as a blocker."),
  "I-B_DOCTOR": dict(artifact=1, runtime=1, adversarial_verification=0,
      negative_control=1, failure_recovery=0, durable_receipt=1, integration=1,
      measured_useful_work=0, named_boundaries=1,
      note="39-technique library and applicability matrix exist. The gate -- on an "
           "UNSEEN model reduce a huge hypothesis space and explain why -- has not "
           "been run end to end on an unseen specimen."),
  "I-C_GRAVITY_NOETIC": dict(artifact=1, runtime=1, adversarial_verification=1,
      negative_control=1, failure_recovery=0, durable_receipt=1, integration=1,
      measured_useful_work=1, named_boundaries=1,
      note="G023 NOETIC_COMPILER PIPELINE open. Real-weight execution gate open. The "
           "frozen EBPW accounting bug is found and named but the namespaces "
           "(DESIGN_EXPECTED / ARTIFACT_PHYSICAL / RUNTIME_MEASURED) are not yet "
           "permanently separated in code."),
  "I-D_ACCELERATOR": dict(artifact=1, runtime=1, adversarial_verification=1,
      negative_control=1, failure_recovery=1, durable_receipt=1, integration=1,
      measured_useful_work=1, named_boundaries=1,
      note="Every category has receipts. The GATE is still open: C2M T3 is NOT "
           "CLAIMED (no real open CUDA project runs) and P2 has no CUDA differential "
           "because no NVIDIA hardware exists. Category coverage is not gate closure."),
  "I-E_ODYSSEY_I": dict(artifact=1, runtime=1, adversarial_verification=1,
      negative_control=1, failure_recovery=0, durable_receipt=1, integration=1,
      measured_useful_work=0, named_boundaries=1,
      note="4 specimens censused, compounding MEASURED (100%/40%/0% with the Falcon "
           "zero making it meaningful). Model #2 is NOT a Noetic executable and NO "
           "REAL WEIGHTS have executed -- so 'measured useful work' is FALSE for the "
           "school's own product."),
}


def obligations():
    """Parse GOAL.md. Status comes from the file, never from this script."""
    text = GOAL.read_text()
    out = {}
    for m in re.finditer(r"^- \[([ x])\] (G\d+) — (.{0,70})", text, re.M):
        out[m.group(2)] = {"checked": m.group(1) == "x", "title": m.group(3).strip()}
    for m in re.finditer(r"^- \[[ x]\] (G\d+) .*?\| status: ([A-Z_]+)", text, re.M):
        if m.group(1) in out:
            out[m.group(1)]["status"] = m.group(2)
    return out


def build():
    obs = obligations()
    mapped = {g for v in ERA_MAP.values() for g in v}
    unmapped = sorted(set(obs) - mapped)
    orphan = sorted(mapped - set(obs))          # named in the map, absent from GOAL.md

    civ = {}
    for name, ids in ERA_MAP.items():
        present = [g for g in ids if g in obs]
        open_ids = [g for g in present if not obs[g]["checked"]]
        ev = EVIDENCE.get(name)
        sat = sum(ev[c] for c in EVIDENCE_CATEGORIES) if ev else None
        gates = OPEN_GATES.get(name, [])
        ob_pct = round(100 * (len(present) - len(open_ids)) / len(present), 1) if present else None
        ev_pct = round(100 * sat / len(EVIDENCE_CATEGORIES), 1) if ev else None
        civ[name] = {
            "obligations": present,
            "verified": len(present) - len(open_ids),
            "open": open_ids,
            "evidence_satisfied": sat,
            "evidence_of": len(EVIDENCE_CATEGORIES) if ev else None,
            "evidence_pct": ev_pct,
            "obligation_pct": ob_pct,
            "completion_pct": min(ev_pct, ob_pct) if (ev_pct is not None and ob_pct is not None) else None,
            "completion_basis": ("min(evidence categories satisfied / 9, obligations "
                                 "verified / total). The MINIMUM on purpose: I-D has all "
                                 "nine categories and zero verified obligations, and "
                                 "reporting 100% there is the inflation S015 §II forbids. "
                                 "Never a file count."),
            "open_gates": gates,
            "status": ("CIVILIZATION_COMPLETE" if (ev_pct == 100 and ob_pct == 100 and not gates)
                       else "INTEGRATED" if (ev_pct == 100 and not gates)
                       else "ADVERSARIALLY_VERIFIED" if (ev and ev["adversarial_verification"] and ev["negative_control"])
                       else "PHYSICALLY_RUNNING" if (ev and ev["runtime"])
                       else "BUILDING" if ev else "EXPLORING"),
            "evidence": {c: bool(ev[c]) for c in EVIDENCE_CATEGORIES} if ev else None,
            "note": ev.get("note") if ev else "Era-IV/II advance work; not scored under Era-I sovereignty.",
        }

    counts = {}
    for g, v in obs.items():
        counts[v.get("status", "UNKNOWN")] = counts.get(v.get("status", "UNKNOWN"), 0) + 1

    rec = HAWKING / "receipts/headless"
    tests = subprocess.run(
        ["/usr/local/bin/python3", "-m", "pytest", str(HAWKING / "tools/accelerator"), "-q"],
        capture_output=True, text=True, cwd=HAWKING).stdout
    tm = re.search(r"(\d+) passed", tests)

    return {
        "roadmap_version": "HAWKING_CIVILIZATION_ASCENSION_V1",
        "frozen_plan": "HAWKING_SUPER_ROADMAP_FREEZE_V1_2026-08-25.md",
        "generated_from": "disk truth: GOAL.md + receipts + git + a real pytest run",
        "active_era": "I",
        "era_sovereignty": ("ERA I is sovereign. Later-era work is permitted ONLY when it "
                            "is already running, consumes an idle resource, produces "
                            "infrastructure Era I needs, or resolves an uncertainty that "
                            "changes Era-I design. It NEVER earns civilization completion."),
        "active_civilizations": ERA_I,
        "civilization_status": civ,
        "obligation_status_counts": counts,
        "obligations_total": len(obs),
        "unmapped_obligations": unmapped,
        "orphan_map_entries": orphan,
        "receipt_count": len(list(rec.glob("*.json"))),
        "accelerator_receipt_count": len(list(rec.glob("ACCELERATOR_*.json"))),
        "last_verified_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=HAWKING).stdout.strip(),
        "last_verified_test_count": int(tm.group(1)) if tm else None,
        "test_count_is_from_a_run_not_arithmetic": True,
        "named_gates": {
            "NVIDIA_CUDA_HARDWARE": "P2 differential. No local NVIDIA execution exists.",
            "FAST_LOCAL_STORAGE": ("real weights for G048. Falcon-H1-7B is 15.17 GB and the "
                                   "contended USB bus measured under ~0.5 MB/s against "
                                   "96-131 MiB/s quiet."),
            "SUDO_POWERMETRICS": "thermal_envelope; sudo not available to this process.",
            "SUDO_PURGE_OR_96GiB_WORKING_SET": "a repeatable cold read.",
            "XCRUN_METAL": "AOT metallib and generated-code inspection. Toolchain absent.",
        },
        "resource_ownership": {
            "USB_BUS_corpdrive": "ModelLake fill (4 hf download workers) -- OPERATOR PRIORITISED",
            "GPU": "free for zero-I/O accelerator science",
            "TIER1_SSD": "~/noetic/stage holds 202 MB: 64 Qwen3 expert tensors + Falcon config",
        },
    }


if __name__ == "__main__":
    s = build()
    pathlib.Path(__file__).with_name("ROADMAP_STATE.json").write_text(json.dumps(s, indent=1))
    print(f"era {s['active_era']} | {s['obligations_total']} obligations "
          f"{s['obligation_status_counts']} | unmapped={s['unmapped_obligations']} "
          f"orphan={s['orphan_map_entries']} | tests={s['last_verified_test_count']}")
    for k in ERA_I:
        c = s["civilization_status"][k]
        print(f"  {k:20s} {c['completion_pct']:5.1f}%  {c['verified']}/{len(c['obligations'])} verified"
              f"  open={c['open']}")
