"""The generation pass must REFUSE on real evidence, not perform a refusal.

`refused_on_evidence: 0` was not a broken filter. The loop asked the negative
index whether a python module name was a dead hypothesis family, which is a
category error the index can only ever answer no to. And its other work sources
-- the frontier, and the Codex candidate queue -- are pruned before the sidecar
ever sees them, so consuming those can honestly report zero refusals forever.

The danger in fixing that is the obvious one: proposing only ideas already known
to be dead, so the counter goes up and nothing was decided. These tests are
weighted against exactly that -- the proposal space must be the fixed taxonomy,
most of it must SURVIVE, and every rejection must cite a scar file that exists.
"""
import json

import pytest

from tools.future import autonomy_run as ar
from tools.future import negative_index as ni
from tools.future._common import REPO


def test_proposal_space_is_the_fixed_taxonomy_not_the_set_of_dead_ideas():
    """NEGATIVE CONTROL against circularity.

    If the generator drew its families from the scars, every proposal would die
    and the pass would decide nothing. The taxonomy is a fixed vocabulary, so
    most proposals must survive.
    """
    assert set(ar.FAMILY_TAXONOMY) == set(ni.FAMILY_SLUGS)
    scars = ni.ingest()
    families_with_scars = {
        s.hypothesis_family for s in scars if s.hypothesis_family != ni.UNRECORDED
    }
    unscarred = set(ar.FAMILY_TAXONOMY) - families_with_scars
    assert unscarred, "every proposable family carries a scar; the space is circular"


def test_live_parents_are_canonical_slugs_or_no_refusal_can_ever_fire():
    """A scar is model-targeted: a wrong slug silently refuses nothing, forever."""
    for parent in ar.LIVE_PARENTS:
        assert ni.canon_model(parent) == parent, f"{parent} is not a canonical slug"


def test_the_grid_kills_some_and_spares_most():
    scars = ni.ingest()
    dead = alive = 0
    for parent in ar.LIVE_PARENTS:
        for organ in sorted(set(ar.fs.SCHOOL_ORGAN_SLUG.values())):
            for fam in ar.FAMILY_TAXONOMY:
                if ni.refuse_if_dead(
                    {"model": parent, "organ": organ, "hypothesis_family": fam}, scars
                ):
                    dead += 1
                else:
                    alive += 1
    assert dead > 0, "the filter never fires; a refusal would be theatre"
    assert alive > dead, "more than half the space is dead; the proposals are pre-selected"


def test_every_rejection_cites_a_scar_source_that_exists_on_disk():
    """A citation that does not resolve is an assertion wearing a receipt's clothes."""
    scars = ni.ingest()
    checked = 0
    for parent in ar.LIVE_PARENTS:
        for organ in ("routed_experts", "router", "attention", "mlp"):
            for fam in ar.FAMILY_TAXONOMY:
                dead = ni.refuse_if_dead(
                    {"model": parent, "organ": organ, "hypothesis_family": fam}, scars
                )
                if not dead:
                    continue
                src = str(dead.get("source_path") or "")
                assert src, "refusal with no source_path cannot be cited"
                assert (REPO / src).exists(), f"cited scar source is absent: {src}"
                assert dead.get("reopen_condition"), "a scar with no reopen condition is a wall"
                checked += 1
                if checked >= 25:
                    return
    assert checked, "no refusal was produced to check"


def test_a_short_run_emits_idea_rejected_the_judge_can_read(tmp_path):
    """End to end: the driver must emit the event shape the trial judge scores."""
    tl = tmp_path / "tl.json"
    ar.run(trial="15m", duration_s=45, timeline=tl)
    doc = json.loads(tl.read_text())
    rejected = [e for e in doc["events"] if e["kind"] == "idea_rejected"]
    assert rejected, "no idea_rejected event; the reject condition cannot be met"
    for event in rejected[:20]:
        assert event["payload"].get("idea"), "rejection did not name the idea"
        cites = event.get("cites") or []
        assert cites and all(c for c in cites), "rejection cited nothing"
    assert doc["summary"]["hypotheses_still_live"] > doc["summary"]["refused_on_evidence"]


def test_run_never_emits_an_idle_event(tmp_path):
    tl = tmp_path / "tl2.json"
    ar.run(trial="15m", duration_s=30, timeline=tl)
    kinds = {e["kind"] for e in json.loads(tl.read_text())["events"]}
    assert not (kinds & {"idle", "awaiting_instructions", "all_tasks_complete"})


def test_the_metal_blocker_is_measured_not_quoted():
    """A repeated blocker line was half false and it scoped the whole campaign.

    "no Metal-capable GPU and no Metal compiler" was carried from a blocker list
    into three places in this driver. The GPU is an M3 Ultra and it is present;
    what is absent is the offline shader compiler. A missing GPU would block
    every physical measurement, while a missing offline compiler blocks
    precompilation -- different scopes, different work unblocked.
    """
    from tools.future import hardware_doctor as hwd

    state = hwd.metal_state()
    why = ar._metal_why()
    assert state["is_a_measurement"] is False, "this is a capability probe, not a timing"
    assert state["runtime_source_compilation"] in {"AVAILABLE", "UNKNOWN"}, (
        "AVAILABLE only after the probe exercised it; UNKNOWN means not run, "
        "never a guess that it fails"
    )
    if state["gpu_present"]:
        assert "no Metal-capable GPU" not in why
        assert state["chip"] in why
    assert ("compiler is absent" in why) == (not state["offline_metal_compiler"])


def test_the_driver_speaks_the_lane_vocabulary_the_frontier_actually_uses():
    """Invented lane names made the frontier's own work silently unreachable.

    The driver declared CPU_ANALYSIS / CPU_VERIFY / CPU_REPRESENTATION / DISK_IO.
    No frontier item requires any of those, so `required_lanes <= available` was
    false for all 31 NEXT_WORK items and next_work() and refill() returned an
    empty list on every call. The loop still had work -- it queued capabilities
    directly -- so nothing looked broken, and the frontier's own work never ran.
    """
    from tools.future import frontiers as frontiers_mod

    assert set(ar.AVAILABLE_LANES) <= set(frontiers_mod.THIS_HOST_LANES)
    assert set(ar.BLOCKED_LANES) == set(frontiers_mod.HARDWARE_LANES)
    assert not (set(ar.AVAILABLE_LANES) & set(ar.BLOCKED_LANES))
    assert frontiers_mod.next_work(ar.AVAILABLE_LANES), (
        "the frontier yields no work for these lanes; the vocabulary is wrong again"
    )


def test_refill_is_exercised_without_waiting_for_starvation(tmp_path):
    """A loop that only asks for work at zero never refills in a full window.

    The 1h trial queued seven multi-hundred-GB verifications and so never once
    reached the end of its queue.
    """
    tl = tmp_path / "tl.json"
    ar.run(trial="15m", duration_s=180, timeline=tl)
    doc = json.loads(tl.read_text())
    kinds = [e["kind"] for e in doc["events"]]
    assert "result_ingested" in kinds, "the judge scores result_ingested, not receipt_ingested"
    refills = [e for e in doc["events"] if e["kind"] == "work_refilled"]
    ingests = [e for e in doc["events"] if e["kind"] == "result_ingested"]
    assert refills, "no refill happened inside the window"
    assert min(e["t_s"] for e in ingests) < max(e["t_s"] for e in refills), (
        "a refill must follow an ingested result to count as refilling after work"
    )
    for event in refills:
        assert event["payload"]["unit_ids"], "a refill that added nothing is not a refill"


def test_an_invalidated_run_is_recorded_and_never_reported_as_a_result():
    """Improving the test and claiming the original interval is the failure here."""
    doc = json.loads(ar.build().read_text())
    rows = doc["invalidated_runs"]
    assert rows, "the killed 1h run must be on the record"
    for row in rows:
        assert row["verdict"] == "INVALIDATED_BY_SUBSTRATE_MUTATION"
        assert row["why"] and row["kept"], "say what was lost and what survives"
