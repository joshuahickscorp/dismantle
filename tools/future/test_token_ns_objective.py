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
    # 65.58 from the measured 10.7278 GB/token at the clean-GEMV 703.5 GB/s.
    assert 64.0 < full < 67.0, f"current-byte roof drifted: {full}"


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


def test_moonshot_is_above_the_mlp_alone_ceiling():
    """With the measured byte count the 150 moonshot sits ABOVE the MLP wall.

    Deleting every MLP byte leaves the other 46% of traffic, which caps MLP-only
    progress at ~142.6 TPS. The moonshot therefore cannot be an MLP story at
    all — non-MLP bytes, state, routing, attention or dispatch have to fall too.
    The earlier 154.8 ceiling came from the 8.6%-low byte anchor.
    """
    ceiling = obj.mlp_alone_ceiling_tps()
    assert 140.0 < ceiling < 146.0, f"ceiling drifted: {ceiling}"
    moon = next(r for r in obj.ladder() if r["name"] == "MOONSHOT")
    assert not moon["reachable_by_mlp_bytes_alone"]
    assert moon["lever"] == "UNREACHABLE_BY_MLP_ALONE"
    assert moon["required_remaining_mlp_fraction"] is None


def test_71_is_no_longer_free_executor_recovery():
    """The measured byte count moved 71 above the clean-addressing roof.

    At 9.88 GB/token the roof was 71.2 and 71 TPS looked like pure executor
    recovery. At the measured 10.73 GB/token the roof is 65.6, so 71 now
    requires bytes to fall as well. This is the single most consequential
    correction the release-profile probe produced.
    """
    rows = {r["name"]: r for r in obj.ladder()}
    assert rows["M1"]["lever"] == "EXECUTOR_RECOVERY_ONLY", "50 TPS is still under the roof"
    assert rows["M2"]["lever"] == "BYTES_MUST_FALL", "71 TPS is now above the roof"


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


def test_anchor_reconciliation_closed_with_evidence_not_by_averaging():
    """It was OPEN at 4%. A measurement closed it; the old anchors are kept."""
    rec = obj.build()["anchor_reconciliation"]
    assert rec["status"] == "CLOSED"
    assert rec["closed_by"].endswith("RESIDENT_TOKEN_BUDGET.json")
    # The anchors must now actually multiply out.
    assert abs(rec["disagreement_pct"]) < 0.5, rec["disagreement_pct"]
    # And the superseded values must be preserved, not deleted.
    old = rec["superseded"]
    assert old["active_weight_bytes_per_token"] == 9_878_901_136
    assert old["decode_tps"] == 35.5
    assert old["production_decode_gb_s"] == 337.3


def test_record_round_trips():
    p = obj.record()
    d = json.loads(p.read_text())
    assert d["gpu_authority"] is False
    assert d["evidence_class"] == "STATIC_ONLY"
    assert d["primary_objective"].startswith("MINIMIZE")
