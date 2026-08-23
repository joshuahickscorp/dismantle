"""G011: production Noetic inference does not load parent weights.

Proved by observed file access (DYLD interpose of open/openat), with a
negative control that the detector must catch. See noetic_zero_parent.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from noetic_zero_parent import (
    RECEIPT,
    classify_path,
    parent_dir,
    parse_open_log,
    run_and_write,
)

PARENT = parent_dir()


@pytest.fixture(scope="session")
def receipt() -> dict:
    rec = run_and_write()
    assert RECEIPT.is_file(), f"harness did not write {RECEIPT}"
    return rec


def test_classifier_distinguishes_weight_tokenizer_config():
    weight = PARENT / "model-00001-of-00018.safetensors"
    tok = PARENT / "tokenizer.json"
    cfg = PARENT / "config.json"
    idx = PARENT / "model.safetensors.index.json"
    artifact = Path.home() / "models" / "qwen38-gravity-uniform-q4-v1" / "manifest.json"
    assert classify_path(str(weight), parent=PARENT) == "parent_weight"
    assert classify_path(str(tok), parent=PARENT) == "parent_tokenizer"
    assert classify_path(str(cfg), parent=PARENT) == "parent_config"
    assert classify_path(str(idx), parent=PARENT) == "parent_config"
    assert classify_path(str(artifact), parent=PARENT) == "not_parent"
    assert classify_path("/etc/hosts", parent=PARENT) == "not_parent"


def test_detector_flags_a_parent_safetensor_in_a_synthetic_log(tmp_path: Path):
    """The detector, given a log that DID touch a parent shard, must say so.

    This is the cheap form of the negative-control obligation: a detector that
    cannot classify a safetensors open as a weight cannot certify the clean
    case either.
    """
    log = tmp_path / "control.log"
    log.write_text(
        "\n".join(
            [
                f"open\t{PARENT / 'tokenizer.json'}",
                f"open\t{PARENT / 'model.safetensors.index.json'}",
                f"open\t{PARENT / 'model-00001-of-00018.safetensors'}",
                "open\t/etc/hosts",
            ]
        )
        + "\n"
    )
    obs = parse_open_log(log, parent=PARENT, cwd=tmp_path)
    assert obs["parent_weight"], "detector missed a parent .safetensors open"
    assert any(p.endswith(".safetensors") for p in obs["parent_weight"])
    assert obs["parent_tokenizer"]
    assert obs["parent_config"]
    assert "/etc/hosts" not in obs["parent_weight"]


def test_receipt_written(receipt: dict):
    assert RECEIPT.is_file()
    assert receipt["schema"] == "hawking.headless.noetic_zero_parent.v1"
    assert receipt["verdict"] == "PASS", receipt.get("pass_rule")


def test_production_run_opens_no_parent_weights(receipt: dict):
    prod = receipt["production_run"]
    assert prod["parent_weight_paths"] == []
    assert prod["n_parent_weight_opens"] == 0
    assert prod["verdict"] == "PASS"
    live = prod["live_native_decode"]
    infer = prod["catalog_weight_load_and_cpu_embed"]
    assert live["dylib_loaded"], "DYLD interpose did not load on the native decode"
    assert infer["dylib_loaded"], "DYLD interpose did not load on the catalog/CPU infer"
    cpu = infer["cpu_infer"] or {}
    assert cpu.get("n_opened") == cpu.get("n_catalog")
    assert cpu.get("n_catalog") == 755
    assert cpu.get("ok") is True
    assert cpu.get("followed_source_dir") is False


def test_negative_control_caught_a_parent_weight_open(receipt: dict):
    ctrl = receipt["negative_control"]
    assert ctrl["detector_caught"] is True, (
        "detector missed the composition teacher-scoring path; it cannot "
        "certify the clean case either"
    )
    assert ctrl["parent_weight_opens"], ctrl
    assert any(p.endswith(".safetensors") for p in ctrl["parent_weight_opens"])
    assert ctrl["verdict"] == "PASS"
    assert ctrl["did_not_load_27b"] is True


def test_tokenizer_dependency_is_not_a_weight_dependency(receipt: dict):
    prod = receipt["production_run"]
    dist = receipt["distinctions"]
    assert prod["parent_tokenizer_paths"], (
        "expected the parent tokenizer.json open (prior work; tokenizer "
        "dependency, not a weight dependency)"
    )
    assert all(p.endswith("tokenizer.json") for p in prod["parent_tokenizer_paths"])
    assert dist["parent_weights_at_inference"]["observed"] is False
    assert dist["parent_config_or_tokenizer"]["observed"] is True
    assert dist["parent_at_compile_time"]["observed"] is False
    found = " ".join(prod["found"]).lower()
    assert "tokenizer" in found
    assert "violation" not in found
