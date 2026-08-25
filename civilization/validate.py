"""Adversarial validator for ROADMAP_STATE.json.

Its job is NOT to confirm the ledger looks tidy. It is to REFUSE a ledger that
inflates. Every rule here exists because the failure it names is one this program
has actually committed: a percentage computed from file counts, a status claimed
while its gate is open, a test count written from arithmetic instead of a run.
"""
import json, pathlib, re, sys

CATEGORIES = 9


def validate(state, goal_text=None):
    """Return list of violations. Empty list = the ledger is defensible."""
    v = []
    g = lambda k: state.get(k)

    if g("active_era") != "I":
        v.append(f"active_era is {g('active_era')}: Era I is sovereign until its gates close")

    for name, c in state.get("civilization_status", {}).items():
        ev, ob, comp = c.get("evidence_pct"), c.get("obligation_pct"), c.get("completion_pct")
        if ev is None:
            continue
        # THE INFLATION RULE. Reporting evidence coverage as completion is exactly
        # what the first build of this ledger did for I-D: 9/9 categories, 0/8
        # obligations, printed 100%.
        if comp != min(ev, ob):
            v.append(f"{name}: completion {comp} != min(evidence {ev}, obligations {ob}) -- inflation")
        if c.get("status") == "CIVILIZATION_COMPLETE" and c.get("open_gates"):
            v.append(f"{name}: CIVILIZATION_COMPLETE with {len(c['open_gates'])} open gates")
        # round to the SAME precision the ledger stores. Comparing a rounded value
        # against an unrounded one by exact equality reports arithmetic as a finding --
        # a defect this program sealed once already on a noisy pagein counter.
        if c.get("evidence") and round(100 * sum(c["evidence"].values()) / CATEGORIES, 1) != ev:
            v.append(f"{name}: evidence_pct does not match its own category table")
        if len(c.get("open", [])) and c.get("obligation_pct") == 100:
            v.append(f"{name}: obligation_pct 100 with {len(c['open'])} open obligations")

    if state.get("unmapped_obligations"):
        v.append(f"unmapped obligations: {state['unmapped_obligations']} -- every "
                 "obligation must land in the era topology or the map is a fiction")
    if state.get("orphan_map_entries"):
        v.append(f"orphan map entries: {state['orphan_map_entries']} -- named in the map, "
                 "absent from GOAL.md")

    if not state.get("test_count_is_from_a_run_not_arithmetic"):
        v.append("test count not marked as coming from a run (S015 XI: the run is evidence)")
    if not isinstance(state.get("last_verified_test_count"), int):
        v.append("last_verified_test_count is not an integer from a real pytest run")

    # Status counts must match the authority on disk, not this file.
    if goal_text is not None:
        real = {}
        for m in re.finditer(r"status: ([A-Z_]+)", goal_text):
            real[m.group(1)] = real.get(m.group(1), 0) + 1
        if real != state.get("obligation_status_counts"):
            v.append(f"status counts {state.get('obligation_status_counts')} disagree with "
                     f"GOAL.md {real} -- disk state is authority")

    if not state.get("named_gates"):
        v.append("no named gates: S015 §129 requires a blocker to name its exact missing input")
    return v


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    st = json.loads((here / "ROADMAP_STATE.json").read_text())
    goal = (pathlib.Path.home() / ".claude/ultragoal/hawking-odyssey-maxx-ascension/GOAL.md").read_text()
    bad = validate(st, goal)
    print("\n".join(f"VIOLATION: {b}" for b in bad) or "ledger defensible: 0 violations")
    sys.exit(1 if bad else 0)
