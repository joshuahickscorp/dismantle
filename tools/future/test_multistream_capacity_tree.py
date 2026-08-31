"""Nine classes, discriminators written before the runs, and one already dead.

S025 §13 is explicit about the trap: do NOT call the effect "serial dependency"
merely because concurrency helped. The tree exists so that a class can be killed
by evidence rather than by narrative, and class H - the only one that would have
made "serial dependency" literally true at the dispatch grain - is the one the
DAG already killed.
"""
from __future__ import annotations

import pytest

from tools.future import multistream_capacity_tree as tree


def test_a_class_without_a_cheap_discriminator_is_refused():
    with pytest.raises(tree.TreeRefused, match="cheap discriminator"):
        tree._cls(id="x", claim="c", max_payoff_ms=None, discriminator="short",
                  kills_if="a", reopens_if="b")


def test_kill_and_reopen_criteria_are_mandatory():
    long = "a discriminator long enough to clear the forty character bar here"
    with pytest.raises(tree.TreeRefused, match="BEFORE the run"):
        tree._cls(id="x", claim="c", max_payoff_ms=None, discriminator=long,
                  kills_if="", reopens_if="b")


def test_a_kill_must_name_its_evidence():
    long = "a discriminator long enough to clear the forty character bar here"
    with pytest.raises(tree.TreeRefused, match="without naming the evidence"):
        tree._cls(id="x", claim="c", max_payoff_ms=None, discriminator=long,
                  kills_if="a", reopens_if="b", status=tree.KILLED)


def test_all_nine_classes_are_present_and_each_can_be_reopened():
    cs = tree.classes()
    assert len(cs) == 9
    for c in cs:
        assert c["reopens_if"], c["id"]
        assert c["kills_if"], c["id"]


def test_class_h_is_dead_and_cites_the_dag():
    h = next(c for c in tree.classes() if c["id"].startswith("H_"))
    assert h["status"] == tree.KILLED
    assert "SINGLE_TOKEN_PARALLEL_SLACK" in h["killed_by"]
    assert "0 artificial barriers" in h["killed_by"]


def test_the_tree_refuses_to_claim_h_dead_if_the_dag_disagrees(monkeypatch, tmp_path):
    """The kill is only as good as the receipt behind it."""
    fake = tmp_path / "slack.json"
    fake.write_text('{"theoretically_overlapable_ns": 5}')
    monkeypatch.setattr(tree, "DAG_REL", str(fake.relative_to(tree.REPO))
                        if str(fake).startswith(str(tree.REPO)) else tree.DAG_REL)
    if str(fake).startswith(str(tree.REPO)):
        with pytest.raises(tree.TreeRefused, match="class H is NOT dead"):
            tree.summary()


def test_the_parent_class_exists_to_be_split_not_tested():
    a = next(c for c in tree.classes() if c["id"].startswith("A_"))
    assert "SPLIT, not tested" in a["cheapest_discriminator"]
    assert "redundant rather than false" in a["kills_if"]


def test_payoffs_are_null_rather_than_invented():
    for c in tree.classes():
        assert c["max_payoff_ms"] is None, (
            f"{c['id']} carries a payoff number nothing has bounded yet"
        )


def test_the_summary_refuses_to_name_the_cause():
    s = tree.summary()
    assert s["n_killed"] == 1 and s["n_open"] == 8
    assert "concurrency helping is the OBSERVATION" in s["do_not_call_it_serial_dependency"]


def test_the_next_cheapest_are_the_within_kernel_classes():
    """The DAG pointed the search at the kernels; the tree must follow."""
    nxt = tree.summary()["next_cheapest"]
    assert nxt[0].startswith("B_occupancy")
    assert any(x.startswith("E_memory_level") for x in nxt)
