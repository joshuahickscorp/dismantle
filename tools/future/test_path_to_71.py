"""The path set must not flatter itself, and refuted components must stay out."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import path_to_71 as p


def test_refuted_components_are_excluded_and_named():
    r = p.compose(["deltanet_q3"])
    assert r["components"] == []
    assert any("REFUTED" in s for s in r["skipped"])


def test_overlapping_levers_are_not_summed():
    """aux_u8 and group_size_1024 attack the same 1.07 GB."""
    both = p.compose(["aux_group_size_1024", "aux_u8"])
    assert "aux_u8" not in both["components"]
    assert any("overlaps" in s for s in both["skipped"])
    # compose() rounds to 4 decimals for the receipt; compare at that precision.
    assert both["gb_removed"] == round(p.COMPONENTS["aux_group_size_1024"]["gb_saved"], 4)


def test_only_path_01_is_qualified():
    rows = {r["path"]: r for r in p.paths()}
    assert rows["PATH_01"]["weakest_evidence"] == "QUALIFIED"
    for pid in ("PATH_02", "PATH_03", "PATH_04"):
        assert rows[pid]["weakest_evidence"] == "PROSPECTIVE"


def test_everything_on_record_does_not_reach_50():
    """The headline that keeps the campaign honest."""
    best = max(p.paths(), key=lambda r: r["tps"])
    assert best["tps"] < 50.0, best["tps"]
    assert best["tps"] > 40.0, best["tps"]


def test_the_gap_is_stated_as_a_share_of_remaining_gpu():
    g = p.gap_to_71()
    assert g["still_to_remove_ms"] > 8.0
    assert 0.35 < g["still_to_remove_share_of_gpu"] < 0.50
    assert "does not exist yet" in g["verdict"]


def test_no_qualified_component_lacks_parity_evidence():
    for cid, c in p.COMPONENTS.items():
        if c["evidence"] == "QUALIFIED":
            assert c.get("parity"), cid
            assert c.get("source"), cid
