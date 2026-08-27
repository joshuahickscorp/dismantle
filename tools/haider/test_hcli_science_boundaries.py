from __future__ import annotations

from hcli.agentos.benchmark_boundary import (
    DIAGNOSTIC_CONTAMINATED,
    QUALIFIED_PROTECTED,
    classify_window,
)
from hcli.agentos.flash_executable import run_flash_executable_scaffold
from hcli.agentos.fpga_preboard import simulate_partition
from hcli.agentos.protected_benchmark_watcher import _classify_blockers
from hcli.agentos_cli import build_parser


def _quiet_sample() -> dict:
    return {"quiet": True, "contenders": [], "method": "test"}


def test_benchmark_boundary_cannot_be_overridden_by_caller_qualification():
    result = classify_window(
        {"quiet": False, "contenders": [{"comm": "WindowServer"}]},
        _quiet_sample(),
        {"state": "QUIESCED"},
        qualification=True,
    )
    assert result["benchmark_class"] == DIAGNOSTIC_CONTAMINATED
    assert result["qualification"] is False
    assert result["NOT_FOR_PROMOTION"] is True


def test_quiet_boundary_is_explicitly_protected_but_still_not_promotion_by_default():
    result = classify_window(_quiet_sample(), _quiet_sample(), {"state": "QUIESCED"})
    assert result["benchmark_class"] == QUALIFIED_PROTECTED
    assert result["qualification"] is True
    assert result["NOT_FOR_PROMOTION"] is True


def test_flash_executable_scaffold_writes_honest_budgets(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = "/Users/scammermike/Downloads/hawking/receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
    result = run_flash_executable_scaffold(repo_root=repo, science_receipt=source)
    assert result["status"] == "PASSED"
    manifest = result["manifest"]
    assert manifest["status"] == "SCAFFOLD_ONLY"
    assert manifest["promotion_allowed"] is False
    assert manifest["native_loader"]["status"] == "NOT_IMPLEMENTED"
    assert manifest["complete_token_timing"]["accepted_tps"] is None
    assert result["ebpw_budget"]["measured"]["complete_system_ebpw"] is None
    assert result["token_ns_budget"]["system_ledger"]["complete_generation_wall_ns"] is None


def test_fpga_partition_simulation_is_model_specific_and_never_hardware():
    qwen = simulate_partition("qwen27")
    flash = simulate_partition("flash-next")
    assert len(qwen["scenarios"]) == 3
    assert len(flash["scenarios"]) >= 6
    assert all(row["label"] == "[S]" for row in qwen["scenarios"] + flash["scenarios"])
    assert all(row["physical_execution"] is False for row in qwen["scenarios"] + flash["scenarios"])
    assert {row["decision"] for row in flash["scenarios"]} <= {"MIXED_CANDIDATE", "REJECT_MIXED_IF_NOT_BEAT"}


def test_watcher_never_classifies_modellake_as_pausable():
    classes = _classify_blockers([
        {"job_id": "lake", "label": "modellake-flash-next-acquire", "argv": ["python3", "tools/odyssey/modellake.py", "acquire"], "pid": 7},
        {"job_id": "bench", "label": "hcli-diagnostic", "argv": ["python3", "-m", "hcli", "agentos", "accelerator-regression"], "pid": 8},
    ])
    assert classes["blockers"][0]["kind"] == "MODELLAKE_UNTOUCHABLE"
    assert classes["pausable_hcli_jobs"][0]["job_id"] == "bench"


def test_cli_exposes_general_science_surfaces():
    parser = build_parser()
    assert parser.parse_args(["qwen27-runtime-archaeology"]).command == "qwen27-runtime-archaeology"
    assert parser.parse_args(["qwen27-mlp-ab"]).command == "qwen27-mlp-ab"
    assert parser.parse_args(["flash-executable"]).command == "flash-executable"
    assert parser.parse_args(["protected-bench-watch"]).command == "protected-bench-watch"
