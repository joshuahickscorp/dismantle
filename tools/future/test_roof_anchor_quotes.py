"""The ceiling audit must CHECK its quotes, not record them.

roof_anchor exists to find numbers promoted across hops without their roof. Its
own CEILING_AUDIT stored quoted_value into a dict and never compared it against
the receipt it named - so when the budget's demonstrated rung moved 47.97 ->
47.25, the audit went on quoting 47.97 with every test green. An audit that
records a number instead of checking it is the defect it exists to find.
"""
from __future__ import annotations

import pytest

from tools.future import roof_anchor as ra


def test_every_row_is_checked_or_says_why_not():
    for row in ra.CEILING_AUDIT:
        if row.get("quote_checked"):
            continue
        assert row.get("why_not_checked"), f"{row['id']} silently skipped its quote"


def test_most_rows_actually_resolve():
    checked = [r for r in ra.CEILING_AUDIT if r.get("quote_checked")]
    assert len(checked) >= len(ra.CEILING_AUDIT) - 1


def test_a_wrong_quote_raises():
    with pytest.raises(ra.QuoteDrift):
        ra._audit_row(
            id="deliberate", receipt="receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
            field="ladder[rung=71 TPS].tps", quoted_value=1.0,
            rests_on_roof_id=None, roof_named_in_record=False, defect=None, reading="x",
        )


def test_unresolvable_field_is_recorded_not_passed():
    row = ra._audit_row(
        id="nope", receipt="receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json",
        field="no.such.path", quoted_value=1.0,
        rests_on_roof_id=None, roof_named_in_record=False, defect=None, reading="x",
    )
    assert row["quote_checked"] is False and row["why_not_checked"]


def test_selector_grammar():
    f = ra._resolve_field
    b = "receipts/future/RESIDENT_71TPS_CAUSAL_BUDGET.json"
    assert f(b, "ladder[rung=71 TPS].tps") == 71.0
    assert f(b, "measured_now.active_bytes") == 9_878_901_136
    assert f(b, "ladder[rung=no such rung].tps") is ra._UNRESOLVABLE
    assert f("receipts/future/NO_SUCH.json", "x") is ra._UNRESOLVABLE


def test_stale_inheritance_across_hops_is_detected():
    """A quote can match its own receipt and still be stale at the origin."""
    stale = [r for r in ra.CEILING_AUDIT if r.get("inherited_quote_is_stale")]
    assert {r["id"] for r in stale} == {
        "capability_map_inherits_roof_on_todays_bytes",
        "improvement_metabolism_inherits_roof_on_todays_bytes",
    }
    for r in stale:
        assert r["quoted_value"] == 66.54 and r["origin_value"] == 65.15
