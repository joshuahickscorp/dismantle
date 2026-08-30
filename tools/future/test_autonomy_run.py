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
