from __future__ import annotations

import json
import struct

from hcli.agentos.benchmark_boundary import (
    DIAGNOSTIC_CONTAMINATED,
    QUALIFIED_PROTECTED,
    classify_window,
)
from hcli.agentos.flash_executable import run_flash_executable_scaffold
from hcli.agentos import flash_tensor_probe
from hcli.flash_next import PINNED_REVISION
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


def test_flash_tensor_probe_reads_bounded_slice_and_keeps_claim_boundary(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    specimen = lake / "specimens" / flash_tensor_probe.LAKE_SLUG
    manifests = lake / "manifests"
    specimen.mkdir(parents=True)
    manifests.mkdir(parents=True)
    shard_name = "model-00002-of-00131.safetensors"
    tensor_name = flash_tensor_probe.DEFAULT_TENSOR
    values = [0x3F80, 0x4000] * 32
    payload = b"".join(struct.pack("<H", value) for value in values)
    header = json.dumps({tensor_name: {"dtype": "BF16", "shape": [64], "data_offsets": [0, len(payload)]}}, separators=(",", ":")).encode()
    (specimen / shard_name).write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (specimen / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor_name: shard_name}}), encoding="utf-8"
    )
    (manifests / f"{flash_tensor_probe.LAKE_SLUG}.json").write_text(
        json.dumps({"repo": "Qwen/Qwen3.8-Flash-Next", "revision": PINNED_REVISION, "resolved_sha": PINNED_REVISION, "n_files": 2}), encoding="utf-8"
    )
    monkeypatch.setattr(flash_tensor_probe, "LAKE_ROOT", lake)
    receipt = tmp_path / "probe.json"

    result = flash_tensor_probe.run_flash_tensor_probe(
        root=specimen,
        sample_bytes=len(payload),
        emit=receipt,
    )

    assert result["status"] == "PASSED"
    assert result["source_label"] == "[V]"
    assert result["candidate_label"] == "[D]"
    assert result["model_loaded"] is False
    assert result["body_mutated"] is False
    assert result["source_tensor"]["slice_bytes"] == len(payload)
    assert result["dense_vs_packed_low_bit"]["candidate"]["effective_bits_per_value"] == 4.25
    assert result["dense_vs_packed_low_bit"]["comparison"]["capability_parity"] == "NOT_TESTED"


def test_flash_executable_scaffold_ingests_probe_as_bounded_evidence(tmp_path):
    repo = tmp_path / "repo"
    receipts = repo / "receipts" / "headless"
    receipts.mkdir(parents=True)
    source = "/Users/scammermike/Downloads/hawking/receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
    (receipts / "FLASH_FIRST_TENSOR_PROBE.json").write_text(json.dumps({
        "status": "PASSED",
        "source_label": "[V]",
        "candidate_label": "[D]",
        "tensor_name": "probe.tensor",
        "organ": {"id": "routed_experts", "label": "[D]"},
        "dense_vs_packed_low_bit": {
            "candidate": {"scheme": "test", "effective_bits_per_value": 4.25},
            "comparison": {"candidate_is_smaller": True},
        },
        "body_mutated": False,
        "model_loaded": False,
    }), encoding="utf-8")
    result = run_flash_executable_scaffold(repo_root=repo, science_receipt=source)
    assert result["status"] == "PASSED"
    assert result["manifest"]["source_tensor_probe"]["status"] == "PASSED"
    assert result["manifest"]["chosen_representation"]["status"] == "BOUNDED_SLICE_OBSERVED_NOT_WHOLE_MODEL"
    assert result["manifest"]["complete_token_timing"]["accepted_tps"] is None


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
    assert parser.parse_args(["flash-tensor-probe"]).command == "flash-tensor-probe"
    assert parser.parse_args(["protected-bench-watch"]).command == "protected-bench-watch"
