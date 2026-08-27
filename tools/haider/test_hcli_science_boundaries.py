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
from hcli.agentos import flash_representation_experiment
from hcli.agentos import flash_transform_parity
from hcli.agentos import flash_loader_roundtrip
from hcli.flash_next import PINNED_REVISION
from hcli.agentos.fpga_preboard import simulate_partition
from hcli.agentos.protected_benchmark_watcher import _classify_blockers
from hcli.agentos_cli import build_parser
from hcli.nomenclature import (
    CANONICAL_PIPELINE,
    COMPATIBILITY_ALIASES,
    NOMENCLATURE_VERSION,
)
from hcli.physical_graph import compile_physical_graph


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


def test_flash_executable_scaffold_ingests_full_transform_as_tensor_ebpw_only(tmp_path):
    repo = tmp_path / "repo"
    receipts = repo / "receipts" / "headless"
    receipts.mkdir(parents=True)
    source = "/Users/scammermike/Downloads/hawking/receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
    (receipts / "FLASH_FULL_TENSOR_TRANSFORM_PARITY.json").write_text(json.dumps({
        "status": "PASSED",
        "source_label": "[V]",
        "candidate_label": "[D]",
        "tensor_name": "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "source_tensor": {"shape": [2, 4, 64], "payload_bytes": 1024, "layout": "row-major"},
        "candidates": {
            "independent_q4_g64": {
                "candidate_bytes": 300,
                "effective_bits_per_value": 4.6875,
                "weight_reconstruction": {"cosine": 0.99},
                "reference_vector": {"cosine": 0.98},
            },
            "shared_bf16_basis_nf4_residual": {
                "candidate_bytes": 320,
                "effective_bits_per_value": 5.0,
                "weight_reconstruction": {"cosine": 0.995},
                "reference_vector": {"cosine": 0.99},
            },
        },
        "comparison": {
            "dense_bytes": 1024,
            "full_payload_read": True,
            "capability_parity": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        },
        "transform_parity": {"status": "PASSED", "pack_unpack_parity": True},
        "next_experiment": {"id": "flash-routed-expert-bounded-loader-roundtrip"},
        "body_mutated": False,
        "model_loaded": False,
        "whole_model_capability": "NOT_TESTED",
        "whole_model_runtime": "NOT_TESTED",
    }), encoding="utf-8")

    result = run_flash_executable_scaffold(repo_root=repo, science_receipt=source)

    assert result["status"] == "PASSED"
    manifest = result["manifest"]
    assert manifest["source_transform_parity"]["status"] == "PASSED"
    assert manifest["chosen_representation"]["status"] == "FULL_TENSOR_TRANSFORM_OBSERVED_NOT_WHOLE_MODEL"
    assert result["ebpw_budget"]["bounded_tensor_observation"]["is_complete_system"] is False
    assert result["ebpw_budget"]["bounded_tensor_observation"]["candidates"]["independent_q4_g64"]["effective_bits_per_value"] == 4.6875
    assert manifest["promotion_allowed"] is False


def test_flash_representation_experiment_uses_source_layout_and_direct_candidate_dots(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    specimen = lake / "specimens" / flash_representation_experiment.LAKE_SLUG
    manifests = lake / "manifests"
    specimen.mkdir(parents=True)
    manifests.mkdir(parents=True)
    tensor_name = flash_representation_experiment.DEFAULT_TENSOR
    shard_name = "model-00002-of-00131.safetensors"
    # Two experts x two complete rows x 64 columns: enough to exercise the
    # source [expert, row, column] layout and one full G64 group per row.
    values = []
    for expert in range(2):
        for row in range(2):
            values.extend([0x3F80 + expert * 0x20 + row * 0x10 + (column % 4) for column in range(64)])
    payload = b"".join(struct.pack("<H", value) for value in values)
    header = json.dumps({tensor_name: {"dtype": "BF16", "shape": [2, 2, 64], "data_offsets": [0, len(payload)]}}, separators=(",", ":")).encode()
    (specimen / shard_name).write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (specimen / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor_name: shard_name}}), encoding="utf-8"
    )
    (manifests / f"{flash_representation_experiment.LAKE_SLUG}.json").write_text(
        json.dumps({
            "repo": "Qwen/Qwen3.8-Flash-Next",
            "revision": PINNED_REVISION,
            "resolved_sha": PINNED_REVISION,
            "path": str(specimen),
            "n_files": 2,
        }), encoding="utf-8"
    )
    monkeypatch.setattr(flash_representation_experiment, "LAKE_ROOT", lake)

    result = flash_representation_experiment.run_flash_representation_experiment(
        root=specimen,
        expert_indices=[0, 1],
        row_count=2,
        emit=tmp_path / "representation.json",
    )

    assert result["status"] == "PASSED"
    assert result["source_tensor"]["layout"].startswith("row-major [expert, row, column]")
    assert result["source_tensor"]["values_read"] == 256
    assert result["candidates"]["independent_q4_g64"]["direct_representation_dot"] is True
    assert result["candidates"]["shared_bf16_basis_nf4_residual"]["direct_representation_dot"] is True
    assert result["comparison"]["same_source_rows"] is True
    assert result["comparison"]["capability_parity"] == "NOT_TESTED"
    assert result["body_mutated"] is False


def test_flash_transform_parity_streams_complete_tensor_without_runtime_claim(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    specimen = lake / "specimens" / flash_transform_parity.LAKE_SLUG
    manifests = lake / "manifests"
    specimen.mkdir(parents=True)
    manifests.mkdir(parents=True)
    tensor_name = flash_transform_parity.DEFAULT_TENSOR
    shard_name = "model-00002-of-00131.safetensors"
    values = []
    for expert in range(2):
        for row in range(4):
            values.extend([0x3F80 + expert * 0x20 + row * 0x10 + (column % 4) for column in range(64)])
    payload = b"".join(struct.pack("<H", value) for value in values)
    header = json.dumps({tensor_name: {"dtype": "BF16", "shape": [2, 4, 64], "data_offsets": [0, len(payload)]}}, separators=(",", ":")).encode()
    (specimen / shard_name).write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (specimen / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor_name: shard_name}}), encoding="utf-8"
    )
    (manifests / f"{flash_transform_parity.LAKE_SLUG}.json").write_text(
        json.dumps({
            "repo": "Qwen/Qwen3.8-Flash-Next",
            "revision": PINNED_REVISION,
            "resolved_sha": PINNED_REVISION,
            "path": str(specimen),
            "n_files": 2,
        }), encoding="utf-8"
    )
    monkeypatch.setattr(flash_transform_parity, "LAKE_ROOT", lake)

    result = flash_transform_parity.run_flash_transform_parity(
        root=specimen,
        chunk_rows=2,
        emit=tmp_path / "transform.json",
    )

    assert result["status"] == "PASSED"
    assert result["source_tensor"]["bytes_read_pass_one"] == len(payload)
    assert result["source_tensor"]["bytes_read_pass_two"] == len(payload)
    assert result["transform_parity"]["complete_source_payload_read"] is True
    assert result["transform_parity"]["pack_unpack_parity"] is True
    assert result["comparison"]["full_payload_read"] is True
    assert result["whole_model_capability"] == "NOT_TESTED"
    assert result["model_loaded"] is False
    assert result["body_mutated"] is False


def test_flash_loader_roundtrip_serializes_descriptor_without_model_load(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    specimen = lake / "specimens" / flash_loader_roundtrip.LAKE_SLUG
    manifests = lake / "manifests"
    specimen.mkdir(parents=True)
    manifests.mkdir(parents=True)
    tensor_name = flash_loader_roundtrip.DEFAULT_TENSOR
    shard_name = "model-00002-of-00131.safetensors"
    values = []
    for expert in range(2):
        for row in range(4):
            values.extend([0x3F80 + expert * 0x20 + row * 0x10 + (column % 4) for column in range(64)])
    payload = b"".join(struct.pack("<H", value) for value in values)
    header = json.dumps({tensor_name: {"dtype": "BF16", "shape": [2, 4, 64], "data_offsets": [0, len(payload)]}}, separators=(",", ":")).encode()
    (specimen / shard_name).write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (specimen / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor_name: shard_name}}), encoding="utf-8"
    )
    (manifests / f"{flash_loader_roundtrip.LAKE_SLUG}.json").write_text(
        json.dumps({
            "repo": "Qwen/Qwen3.8-Flash-Next",
            "revision": PINNED_REVISION,
            "resolved_sha": PINNED_REVISION,
            "path": str(specimen),
            "n_files": 2,
        }), encoding="utf-8"
    )
    monkeypatch.setattr(flash_transform_parity, "LAKE_ROOT", lake)
    monkeypatch.setattr(flash_loader_roundtrip, "LAKE_ROOT", lake)
    transform_receipt = tmp_path / "transform.json"
    transform_result = flash_transform_parity.run_flash_transform_parity(
        root=specimen,
        chunk_rows=2,
        emit=transform_receipt,
    )
    assert transform_result["status"] == "PASSED"

    result = flash_loader_roundtrip.run_flash_loader_roundtrip(
        root=specimen,
        transform_receipt=transform_receipt,
        candidate_id="independent_q4_g64",
        row_count=2,
        emit=tmp_path / "loader.json",
    )

    assert result["status"] == "PASSED"
    assert result["native_loader"] == "BOUNDED_DESCRIPTOR_ROUNDTRIP_ONLY"
    assert result["loader_roundtrip"]["descriptor_json_roundtrip"] is True
    assert result["loader_roundtrip"]["code_pack_unpack_parity"] is True
    assert result["model_loaded"] is False
    assert result["body_mutated"] is False


def test_canonical_nomenclature_is_versioned_without_renaming_legacy_terms():
    assert NOMENCLATURE_VERSION == "HAWKING_NOMENCLATURE_V1"
    assert CANONICAL_PIPELINE[0] == "SourceSpecimen"
    assert CANONICAL_PIPELINE[-1] == "ResidentInstance"
    assert COMPATIBILITY_ALIASES["quantization"] == "GravityOperator"
    assert COMPATIBILITY_ALIASES["artifact"] == "SemanticInspectionRequired"
    graph = compile_physical_graph({"model_id": "flash-next", "organs": []})
    assert graph["nomenclature_version"] == NOMENCLATURE_VERSION
    assert graph["semantic_type"] == "PhysicalGraphPlan"
    assert graph["compiler_stage"] == "PhysicalGraphCompiler"


def test_flash_executable_ingests_bounded_kernel_evidence_without_promoting_it(tmp_path):
    repo = tmp_path / "repo"
    receipts = repo / "receipts" / "headless"
    receipts.mkdir(parents=True)
    source = "/Users/scammermike/Downloads/hawking/receipts/headless/HCLI_FLASH_NEXT_PRE_RUNTIME_SCIENCE.json"
    (receipts / "FLASH_NOETIC_Q4_KERNEL_PARITY.json").write_text(json.dumps({
        "status": "PASSED",
        "source_label": "[V]",
        "derived_label": "[D]",
        "native_kernel": {
            "kernel": "qwen_uniform_q4_group64_matvec",
            "whole_model_capability": "NOT_TESTED",
            "whole_model_runtime": "NOT_TESTED",
        },
        "source_tensor": {"selected_block_bytes": 128},
        "noetic_representation": {"candidate_id": "independent_q4_g64"},
        "gpu_timing": {"gpu_ns_median": 123},
        "parity": {"within_tolerance": True},
        "body_mutated": False,
        "model_loaded": False,
        "complete_system_ebpw": None,
        "flash_tps": None,
        "promotion_allowed": False,
    }), encoding="utf-8")
    result = run_flash_executable_scaffold(repo_root=repo, science_receipt=source)
    assert result["status"] == "PASSED"
    assert result["manifest"]["source_kernel_parity"]["status"] == "PASSED"
    assert result["manifest"]["native_kernels"]["status"] == "PLAN_ONLY"
    assert result["manifest"]["complete_token_timing"]["accepted_tps"] is None
    assert result["manifest"]["promotion_allowed"] is False


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
    assert parser.parse_args(["flash-representation-experiment"]).command == "flash-representation-experiment"
    assert parser.parse_args(["flash-transform-parity"]).command == "flash-transform-parity"
    assert parser.parse_args(["flash-loader-roundtrip"]).command == "flash-loader-roundtrip"
    assert parser.parse_args(["flash-executable", "--kernel-parity-receipt", "kernel.json"]).kernel_parity_receipt == "kernel.json"
    assert parser.parse_args(["protected-bench-watch"]).command == "protected-bench-watch"
