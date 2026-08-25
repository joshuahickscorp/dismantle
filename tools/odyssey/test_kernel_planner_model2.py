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


def test_section_71_is_now_checkable_and_gaps_block_evaluation():
    d = rec()
    for r in d["organ_plan"]:
        assert r["may_be_evaluated"] is (r["n_competent_kernels"] > 0)
    assert d["section_71_rule"]["rule"].startswith("representations are never evaluated")


def test_covered_organs_are_marked_unqualified_until_parity_runs():
    """Competence is not qualification."""
    for r in rec()["organ_plan"]:
        if r["status"] == "COVERED":
            assert "UNQUALIFIED" in r["qualification"]


def test_the_planner_reports_gaps_rather_than_claiming_coverage():
    d = rec()
    assert d["n_gaps"] > 0
    assert d["stage_status"] == "RAN_WITH_GAPS"
    assert set(d["gaps"]) == {"embed", "lm_head", "moe_router", "rmsnorm"}


def test_the_pipeline_receipt_reflects_the_new_stage_status():
    p = json.load(open(RH / "NOETIC_COMPILER_PIPELINE.json"))
    kp = next(s for s in p["stages"] if s["stage"] == "KernelPlanner")
    assert kp["status"] == "RAN_WITH_GAPS"
    assert kp["output"]["n_gaps"] == 4
    assert p["n_stages_automatic_on_model_2"] == 5
