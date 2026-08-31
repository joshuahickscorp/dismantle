"""The objective must not be able to lie about what a milestone requires."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import token_ns_objective as obj


def test_roof_is_a_function_of_bytes_not_a_constant():
    full = obj.roof_tps(obj.ACTIVE_WEIGHT_BYTES_PER_TOKEN)
    half = obj.roof_tps(obj.ACTIVE_WEIGHT_BYTES_PER_TOKEN / 2)
    assert abs(half - 2 * full) < 1e-6, "halving bytes must double the roof"
    assert 70.0 < full < 72.0, f"current-byte roof drifted: {full}"


def test_target_above_the_mlp_ceiling_is_flagged_not_fudged():
    """Deleting 100% of MLP leaves 46% of bytes. Past that, MLP alone cannot."""
    ceiling = obj.mlp_alone_ceiling_tps()
    assert obj.required_mlp_fraction(ceiling + 5.0) is None, (
        "a target above the ceiling must report None, not a negative fraction"
    )
    assert obj.required_mlp_fraction(ceiling - 5.0) is not None
    for r in obj.ladder():
        if not r["reachable_by_mlp_bytes_alone"]:
            assert r["required_remaining_mlp_fraction"] is None
            assert r["lever"] == "UNREACHABLE_BY_MLP_ALONE"
        elif r["required_remaining_mlp_fraction"] >= 1.0:
            assert r["lever"] == "EXECUTOR_RECOVERY_ONLY"
        else:
            assert r["lever"] == "BYTES_MUST_FALL"


def test_the_ladder_splits_at_todays_roof():
    """Below the roof is an executor problem. Above it, bytes have to fall."""
    rows = obj.ladder()
    roof = obj.roof_tps(obj.ACTIVE_WEIGHT_BYTES_PER_TOKEN)
    for r in rows:
        expected = "EXECUTOR_RECOVERY_ONLY" if r["target_tps"] <= roof else "BYTES_MUST_FALL"
        if r["reachable_by_mlp_bytes_alone"]:
            assert r["lever"] == expected, (r["name"], r["lever"], roof)
    assert {r["lever"] for r in rows} >= {"EXECUTOR_RECOVERY_ONLY", "BYTES_MUST_FALL"}, (
        "the ladder must span both levers or it is not describing the campaign"
    )


def test_moonshot_sits_just_under_the_mlp_ceiling():
    """The honest fact: 150 TPS is nearly the MLP-alone wall, not comfortably under."""
    ceiling = obj.mlp_alone_ceiling_tps()
    assert 150.0 < ceiling < 160.0, f"ceiling drifted: {ceiling}"
    moon = next(r for r in obj.ladder() if r["name"] == "MOONSHOT")
    assert moon["reachable_by_mlp_bytes_alone"]
    assert moon["required_remaining_mlp_fraction"] < 0.05, (
        "the moonshot must be recorded as near-total MLP elimination"
    )


def test_ladder_is_monotone_in_difficulty():
    rows = obj.ladder()
    fracs = [r["required_total_byte_fraction"] for r in rows]
    assert fracs == sorted(fracs, reverse=True), "harder target must need fewer bytes"


def test_71_is_recorded_as_a_checkpoint_not_the_destination():
    doc = obj.build()
    assert "any single fixed TPS number, 71 included" in doc["not_the_objective"]
    m2 = next(r for r in doc["ladder"] if r["target_tps"] == 71.0)
    assert "checkpoint" in m2.get("note", "")
    assert doc["no_terminal_target"]


def test_anchor_inconsistency_stays_open_and_is_not_averaged_away():
    rec = obj.build()["anchor_reconciliation"]
    assert rec["status"] == "OPEN"
    assert abs(rec["disagreement_pct"]) > 3.0
    # The receipt must keep BOTH numbers, not a reconciled invention.
    assert rec["implied_bandwidth_gb_s"] != rec["recorded_bandwidth_gb_s"]


def test_record_round_trips():
    p = obj.record()
    d = json.loads(p.read_text())
    assert d["gpu_authority"] is False
    assert d["evidence_class"] == "STATIC_ONLY"
    assert d["primary_objective"].startswith("MINIMIZE")
