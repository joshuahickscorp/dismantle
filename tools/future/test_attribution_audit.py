"""G021 tests: an audit that undercounts exposure is worse than none."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attribution_audit as aa  # noqa: E402


def test_prose_is_not_counted_as_a_footer():
    """'all 800 records regenerated with different hashes' is prose. A naive
    grep counts it; the line-anchored pattern must not."""
    assert not aa.FOOTER.search("records regenerated with different hashes")
    assert not aa.FOOTER.search("Profile JSON regenerated with the new ID")
    assert aa.FOOTER.search("Generated with [Claude Code](https://x)")
    assert aa.FOOTER.search("\U0001F916 Generated with something")


def test_the_trailer_pattern_is_line_anchored_and_case_insensitive():
    assert aa.TRAILER.search("Co-authored-by: x")
    assert aa.TRAILER.search("Co-Authored-By: x")
    assert aa.TRAILER.search("body\nCo-authored-by: x")
    assert not aa.TRAILER.search("mentions co-authored-by in prose")


def test_the_canonical_lines_are_clean():
    c = aa.canonical()
    assert c["clean"] is True
    for ref in aa.CANONICAL:
        assert c[ref]["tool_identities"] == 0
        assert c[ref]["trailers"] == 0
        assert c[ref]["generated_with_footers"] == 0


def test_main_is_actually_pushed_not_assumed():
    assert aa.canonical()["main_is_pushed"] is True


def test_no_claude_or_anthropic_attribution_exists_anywhere():
    n = aa.no_claude_attribution_anywhere()
    assert n["verdict"] == "NONE"
    assert n["n_identities"] == 0
    assert n["generated_with_footers_anywhere"] == 0


def test_a_published_branch_with_no_local_head_is_still_counted():
    """The audit's own first run reported 1 published-dirty branch when the
    answer is 2: it skipped every refs/remotes/ ref, and one published branch
    had no local head to be counted through."""
    r = aa.remaining()
    refs = {x["ref"] for x in r["published_dirty"]}
    assert r["n_published_dirty"] >= 2
    assert any(x.startswith("refs/remotes/origin/") for x in refs), (
        "a published branch with no local head must appear in published_dirty"
    )


def test_published_rows_carry_merge_status_and_distance():
    for row in aa.remaining()["published_dirty"]:
        assert "merged_into_main" in row
        assert row["commits_ahead_of_main"] >= 0


def test_unmerged_published_branches_are_not_proposed_for_deletion():
    """'Nothing may be lost' is this obligation's own words."""
    r = aa.remaining()
    unmerged = [x for x in r["published_dirty"] if not x["merged_into_main"]]
    assert unmerged, "both remaining branches are unmerged"
    w = aa.what_this_does_not_do()
    assert w["does_not_rewrite"] is True and w["does_not_push"] is True
    assert "Nothing may be lost" in w["why"]


def test_the_rewrite_preconditions_name_the_970_commit_loss():
    pre = aa.what_this_does_not_do()["preconditions_for_any_rewrite"]
    joined = " ".join(pre)
    assert "prune-empty=never" in joined
    assert "970" in joined
    assert "bundle" in joined


def test_a_git_failure_refuses_rather_than_reporting_clean(monkeypatch):
    """canonical() is cached, so the uncached primitive is what to poke."""
    monkeypatch.setattr(aa, "_git", lambda *a: (_ for _ in ()).throw(
        aa.AuditRefused("boom")))
    aa._scan.cache_clear()
    with pytest.raises(aa.AuditRefused):
        aa._scan("main")


def test_the_verdict_names_the_exposure_count():
    d = aa.build()
    n = d["remaining"]["n_published_dirty"]
    assert str(n) in d["verdict"]
    assert "CANONICAL_CLEAN" in d["verdict"]
