"""The terminal artifact must refuse itself until its own measurements land.

G066 asks for exactly one of two receipts and adds the sentence that makes this
module necessary: "Probably impossible" is not an acceptable output; a proof of
the binding limit is. The two failure modes are opposite and both are easy - a
premature UNLOCK declares victory, a premature ROOF declares a limit while three
measurements are outstanding. Those are the same error.
"""
from __future__ import annotations

import pytest

from tools.future import terminal_speed_artifact as tsa


def test_it_refuses_while_any_prerequisite_is_open():
    open_pre = [r for r in tsa.prerequisite_status() if not r["met"]]
    if not open_pre:
        pytest.skip("all prerequisites landed; the refusal path is no longer reachable")
    with pytest.raises(tsa.TerminalArtifactRefused) as exc:
        tsa.build()
    for row in open_pre:
        assert row["id"] in str(exc.value), "a refusal must name what is missing"


def test_every_prerequisite_names_a_receipt_a_field_and_a_reason():
    rows = tsa.prerequisite_status()
    assert rows, "a terminal artifact with no prerequisites cannot refuse anything"
    for row in rows:
        assert row["receipt"].startswith("receipts/future/")
        assert row["field"]
        assert len(row["why"]) > 80, f"{row['id']} has no stated reason"


def test_a_missing_receipt_is_unmet_not_assumed_met():
    assert tsa._resolved("receipts/future/NO_SUCH_RECEIPT.json", ["x"]) is None


def test_a_present_receipt_missing_the_field_is_still_unmet():
    """Existence is not measurement. The field is what makes it one."""
    assert tsa._resolved(
        "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json", ["not_a_field"]
    ) is None
    assert tsa._resolved(
        "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json", ["measured_now", "tps"]
    ) is not None


def test_reached_is_never_claimed_while_the_baseline_is_unknown():
    hit = tsa.reached_71()
    assert hit["target_tps"] == 71.0
    assert hit["target_ms"] == pytest.approx(14.085, abs=1e-3)
    if isinstance(hit["current_body_ms"], str):
        assert hit["reached"] is False


def test_the_verdict_is_never_probably_impossible():
    v = tsa.which_receipt()
    assert v["emit"] in (None, tsa.UNLOCK_NAME, tsa.ROOF_NAME)
    assert "probably" not in v["why"].lower()
    assert "impossible" not in v["why"].lower()
