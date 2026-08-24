"""The scoreboard must not make an unmeasured candidate look cheap.

A table of numbers is the easiest place in this campaign to launder a guess into
a fact. Two rules carry that weight: an unmeasured cell renders ABSENT with a
reason rather than 0, and the frontier points stay separate columns because they
are demonstrably different artifacts.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
P = REPO / "receipts" / "headless" / "NOETIC_SCOREBOARD.json"


def _d():
    return json.loads(P.read_text())


def test_no_unmeasured_cell_renders_as_a_number():
    for r in _d()["candidates"]:
        for k, v in r.items():
            if isinstance(v, dict) and "state" in v:
                if v["state"] == "ABSENT":
                    assert v["value"] is None, f"{r['id']}.{k} is ABSENT but carries a value"
                    assert v.get("reason"), f"{r['id']}.{k} is ABSENT without a reason"


def test_frontier_points_are_not_collapsed():
    fp = _d()["frontier_points"]
    # The cheapest thing that passed a screen is NOT the cheapest thing that
    # generates: mix_c reaches 2.3440 EBPW and emits sixteen newlines.
    assert fp["LOWEST_SCREEN_SURVIVOR"] != fp["LOWEST_GENERATION_COHERENT"], (
        "screen survival and generation coherence currently resolve to different "
        "artifacts; collapsing them is how a screened candidate gets promoted"
    )


def test_capability_is_honestly_absent():
    fp = _d()["frontier_points"]["LOWEST_CAPABILITY_SURVIVOR"]
    assert fp["state"] == "ABSENT"
    assert "capability" in fp["reason"].lower()


def test_phase_map_records_the_measured_knee():
    pm = _d()["phase_transition_map"]
    pts = {p["bpw_body"]: p["composed"] for p in pm["measured_points"]}
    assert pts[1.85] == "FAILS" and pts[2.25] == "SURVIVES"
    assert pm["collapse_boundary"] == 1.85
    assert pm["state"] == "PARTIAL", "only one family is mapped; do not claim a full map"


def test_leader_is_not_called_resident():
    body = json.dumps(_d()).lower()
    assert "resident_model" not in body
    leader = [c for c in _d()["candidates"] if "PARENT_A" in c["id"]][0]
    assert "not resident-promoted" in leader["note"].lower()


def test_the_implied_floor_is_derived_from_measured_organ_floors():
    """The campaign's central number, and it must stay tied to measurement.

    If ORGAN_FRONTIERS ever moves a floor, this must move with it rather than
    remaining a remembered constant.
    """
    f = _d()["implied_whole_model_floor"]
    assert f["state"] == "DERIVED"
    assert abs(f["implied_whole_model_ebpw"] - 2.9398) < 0.05
    # The leader is already close to the floor: per-organ quantization is nearly spent.
    assert f["leader_today"] - f["implied_whole_model_ebpw"] < 0.25
    # And the non-MLP organs alone exceed the ~1 EBPW research pressure.
    assert f["floor_even_if_mlp_were_free_ebpw"] > f["research_pressure"], (
        "if this ever inverts, ~1 EBPW became reachable by per-organ quantization"
    )


if __name__ == "__main__":
    n = 0
    for _name, f in sorted(globals().items()):
        if _name.startswith("test_"):
            f(); n += 1; print(f"ok  {_name}")
    print(f"{n} passed")
