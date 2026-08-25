"""G023 stage-audit pins. A pipeline that grades its own automation is the failure
mode §102 names."""
import json, re
from pathlib import Path

import pytest

RH = Path(__file__).resolve().parents[2] / "receipts/headless"
R = RH / "NOETIC_COMPILER_STAGE_AUDIT.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="stage audit not built")


def rec():
    return json.load(open(R))


def test_the_audit_disagrees_with_the_recorded_automation_count():
    d = rec()
    assert d["audited_automatic_on_model_2"] < d["recorded_automatic"]
    assert d["overstated_stages"] == ["KernelPlanner"]


def test_the_overstated_stage_really_has_no_model_2_subject():
    d = rec()
    ks = next(s for s in d["stages"] if s["stage"] == "KernelPlanner")
    assert ks["names_model_2"] is False
    assert ks["declares_specimen"] is None


def test_no_moe_organ_kernel_is_catalogued_in_the_library():
    d = rec()["overstatement"]
    assert d["moe_organ_kernels_in_the_library"] == "NONE"


def test_the_stages_that_did_run_declare_model_2_as_their_specimen():
    for s in rec()["stages"]:
        if s["audited_status"] == "AUTOMATIC_ON_MODEL_2":
            assert s["names_model_2"] is True, s["stage"]


def test_no_stage_produced_bytes():
    d = rec()
    assert all(s["produced_bytes"] is False for s in d["stages"])
    assert "PLANNING pipeline" in d["no_stage_produced_bytes"]["finding"]


def test_the_understatement_is_backed_by_files_that_exist():
    d = rec()["understatement"]["what_actually_exists"]
    root = Path(__file__).resolve().parents[2]
    assert (root / d["reader"]).is_file()
    assert d["reader_lines"] > 5000
    assert d["n_moe_expert_kernels"] >= 5
    assert d["moe_shader"]


def test_the_moe_kernels_are_named_not_merely_counted():
    d = rec()["understatement"]["what_actually_exists"]
    assert all(k.startswith("qwen30_expert_table_") for k in d["moe_expert_kernels"])


def test_the_revised_gap_is_smaller_than_both_previous_statements():
    d = rec()["understatement"]
    assert len(d["revised_gap"]) >= 3
    assert "18_867" in " ".join(d["revised_gap"])
    assert "smaller" in d["size"]


def test_the_correction_was_written_back_into_the_pipeline_receipt():
    p = json.load(open(RH / "NOETIC_COMPILER_PIPELINE.json"))
    ks = next(s for s in p["stages"] if s["stage"] == "KernelPlanner")
    assert ks["status"] == "NOT_RUN_FOR_MODEL_2"
    assert p["n_stages_automatic_on_model_2"] == 5


def test_the_obligation_remains_blocked_with_named_unmet_clauses():
    d = rec()
    assert d["still_blocked"] is True
    assert len(d["unmet_acceptance"]) == 3
