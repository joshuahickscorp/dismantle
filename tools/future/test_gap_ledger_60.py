"""The 60-TPS gap is derived, and the clock cannot be restarted into DISCOVERY.

The two ways this module could lie are a typed gap and a clock that resets on
every build. Both are pinned here.
"""
from __future__ import annotations

import json

import pytest

from tools.future import gap_ledger_60 as g


def test_the_gap_is_derived_from_the_live_budget_not_typed():
    lv = g.live()
    d = json.loads((g.REPO / g.BUDGET_REL).read_text())
    assert lv["ms_per_token"] == d["decode_wall_ms_per_token"]
    assert g.build()["gap_to_60_ms"] == pytest.approx(
        lv["ms_per_token"] - 1000.0 / 60.0, abs=1e-4)


def test_a_self_inconsistent_budget_is_refused(monkeypatch):
    d = json.loads((g.REPO / g.BUDGET_REL).read_text())
    monkeypatch.setattr(g, "_budget", lambda: {**d, "decode_wall_tps": 99.0})
    with pytest.raises(g.GapRefused, match="self-inconsistent"):
        g.live()


def test_a_missing_budget_refuses_rather_than_defaulting(monkeypatch):
    monkeypatch.setattr(g, "BUDGET_REL", "receipts/future/NO_SUCH.json")
    with pytest.raises(g.GapRefused, match="no live token budget"):
        g.live()


def test_organs_are_priced_in_complete_token_tps_not_organ_percent():
    rows = g.organ_win_table()
    cur = g.live()["ms_per_token"]
    top = rows[0]
    assert top["organ"] == "mlp_gate_up", "the table must be ranked by cost"
    assert top["tps_at_20pct_win"] == pytest.approx(
        1000.0 / (cur - top["current_ms"] * 0.2), abs=1e-3)
    # A 20% win on the largest organ is still well short of 60.
    assert top["tps_at_20pct_win"] < 60.0


def test_immaterial_organs_are_named_as_not_worth_an_hour():
    rows = g.organ_win_table()
    small = [r for r in rows if not r["material"]]
    assert small, "embedding and sampling are below the threshold"
    assert all("do not" in r["why"] for r in small)


def test_the_three_dominant_kernels_do_not_reach_60_from_the_live_base():
    d = g.three_dominant()
    assert d["share_of_token"] > 0.75
    assert d["tps_at_20pct_across_all_three"] < 60.0
    assert "PROSPECTIVE composed path" in d["reading"], \
        "S026's 60-from-20% arithmetic used a base that is not qualified"


def test_the_open_set_does_not_reach_60_even_at_its_ceiling():
    """The decisive fact this ledger exists to surface."""
    b = g.build()
    assert b["does_the_open_set_reach_60"] is False
    assert b["sum_of_material_open_ms"] < b["gap_to_60_ms"]
    assert "CEILINGS, not candidates" in b["honest_reading"]


def test_experiments_are_ranked_by_max_ms_removable():
    r = g.ranked_experiments()
    assert r == sorted(r, key=lambda x: -x["max_ms_removable"])
    assert r[0]["id"] == "reach_demonstrated_bandwidth_mlp"


def test_the_clock_start_is_stamped_once_and_never_rewritten():
    first = g._clock_start()
    second = g._clock_start()
    assert first["started_unix"] == second["started_unix"]
    assert (g.REPO / g.CLOCK_REL).is_file()


def test_the_clock_phase_advances_with_elapsed_time(monkeypatch):
    start = g._clock_start()
    for hours, want in ((0.1, "DISCOVERY"), (7, "SURVIVOR_OR_COLLAPSE"),
                        (13, "WIDEN"), (19, "DEPRIVILEGE")):
        monkeypatch.setattr(
            g.time, "time", lambda h=hours: float(start["started_unix"]) + h * 3600)
        assert g.escalation_clock()["phase"] == want


def test_the_clock_is_explicitly_not_an_abandonment():
    c = g.escalation_clock()
    assert "forbids complacency, not the incumbent" in c["not_an_abandonment"]
    assert c["phase_licenses"]
