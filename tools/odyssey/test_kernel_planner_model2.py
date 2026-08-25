"""G023 KernelPlanner pins."""
import json
from pathlib import Path

import pytest

RH = Path(__file__).resolve().parents[2] / "receipts/headless"
R = RH / "KERNEL_PLANNER_MODEL2.json"
KL = RH / "KERNEL_LIBRARY.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="planner not run")


def rec():
    return json.load(open(R))


def test_the_planner_names_model_2_as_its_specimen():
    """The stage it replaces never mentioned model #2 at all."""
    assert rec()["specimen"]["repo"] == "Qwen/Qwen3-30B-A3B"


def test_the_organ_graph_comes_from_the_recognizer_not_a_hand_list():
    d = rec()
    assert "ARCHITECTURE_RECOGNIZER" in d["specimen"]["organ_graph_source"]
    census = json.load(open(RH / "ARCHITECTURE_RECOGNIZER.json"))
    organs = set()
    for k in ("specimens", "heldout_specimens"):
        for s in census.get(k, []) or []:
            if (s.get("result") or {}).get("repo") == d["specimen"]["repo"]:
                organs = {o["organ"] for o in s["result"]["organs"]}
    assert {r["organ"] for r in d["organ_plan"]} <= organs


def test_moe_kernels_are_registered_and_pass_the_library_validator():
    lib = json.load(open(KL))
    reg = lib["moe_registration"]
    assert reg["n_added"] == 18
    assert reg["declared_in_shader"] == reg["referenced_by_runtime"] == reg["in_both"]
    assert lib["n_rejected"] == 0
    assert {"moe_expert", "moe_expert_gate_up"} <= set(reg["organs_now_covered"])


def test_registered_kernels_carry_no_invented_measurements():
    """No parity or throughput number exists for these; claiming one would be forged."""
    lib = json.load(open(KL))
    added = set(lib["moe_registration"]["added"])
    for k in lib["kernels"]:
        if k["kernel_identity"] in added:
            for f in ("measurements", "supported_capability_regime", "parity"):
                assert k[f]["kind"] == "ABSENT", (k["kernel_identity"], f)
                assert k[f]["absent_reason"], (k["kernel_identity"], f)


def test_registered_kernels_point_at_a_shader_that_declares_them():
    root = Path(__file__).resolve().parents[2]
    lib = json.load(open(KL))
    added = [k for k in lib["kernels"]
             if k["kernel_identity"] in set(lib["moe_registration"]["added"])]
    assert added
    src = (root / added[0]["shader"]).read_text()
    for k in added:
        assert f"kernel void {k['kernel_identity']}" in src, k["kernel_identity"]


def test_section_71_is_checked_per_organ_AND_representation():
    """Matching on organ name alone reported moe_expert COVERED by 18 kernels while the
    planner had seeded q2_affine, which none of them executes."""
    d = rec()
    for r in d["organ_plan"]:
        assert r["may_be_evaluated"] is (r["selected_representation"] is not None)
        for fam in r["seeded_families_in_score_order"]:
            assert "family" in fam and "n_competent_kernels" in fam
    assert d["section_71_rule"]["rule"].startswith("representations are never evaluated")


def test_the_planner_considers_every_seeded_family_not_only_the_top_one():
    for r in rec()["organ_plan"]:
        assert len(r["seeded_families_in_score_order"]) >= 2, r["organ"]


def test_a_top_seed_without_a_competent_kernel_is_downgraded_not_accepted():
    d = rec()
    moe = next(r for r in d["organ_plan"] if r["organ"] == "moe_expert")
    assert moe["seeded_families_in_score_order"][0]["family"] == "q2_affine"
    assert moe["seeded_families_in_score_order"][0]["n_competent_kernels"] == 0
    assert moe["downgraded_from_top_seed"] is True
    assert moe["selected_representation"] != "q2_affine"
    assert moe["n_competent_kernels"] > 0


def test_f32_passthrough_is_refused_for_a_gemv_organ():
    """Selecting leftover_f32 for moe_expert would report the largest organ COVERED by
    declining to compress it -- a 4x expansion dressed as a plan."""
    moe = next(r for r in rec()["organ_plan"] if r["organ"] == "moe_expert")
    f32 = [f for f in moe["seeded_families_in_score_order"]
           if f["family"] == "leftover_f32"]
    assert f32 and f32[0]["passthrough_rejected_for_gemv_organ"] is True
    assert moe["selected_representation"] != "leftover_f32"


def test_a_norm_organ_may_select_f32_passthrough():
    n = next(r for r in rec()["organ_plan"] if r["organ"] == "rmsnorm")
    assert n["selected_representation"] == "leftover_f32"
    assert n["may_be_evaluated"] is True


def test_covered_organs_with_kernels_are_unqualified_until_parity_runs():
    """Competence is not qualification."""
    for r in rec()["organ_plan"]:
        if r["status"] == "COVERED" and r["n_competent_kernels"]:
            assert "UNQUALIFIED" in r["qualification"]


def test_the_planner_reports_gaps_rather_than_claiming_coverage():
    d = rec()
    assert d["n_gaps"] > 0
    assert d["stage_status"] == "RAN_WITH_GAPS"
    assert set(d["gaps"]) == {"embed", "lm_head", "moe_router"}


def test_the_pipeline_receipt_reflects_the_new_stage_status():
    p = json.load(open(RH / "NOETIC_COMPILER_PIPELINE.json"))
    kp = next(s for s in p["stages"] if s["stage"] == "KernelPlanner")
    assert kp["status"] == "RAN_WITH_GAPS"
    assert kp["output"]["n_gaps"] == 3
    assert p["n_stages_automatic_on_model_2"] == 5
