"""Odyssey pass: which runtime can actually execute a lake specimen.

Split out because five G048 receipts justified weight-space grading with a
sentence that conflated Hawking's Rust body with mlx_lm.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import odyssey_pass as op



# --- the blocker was misattributed, and a test is what stops it coming back ------

def test_mlx_lm_CAN_read_every_lake_specimen():
    """Five receipts justified weight-space grading with 'no runtime here executes a
    Qwen3-MoE, Falcon or Mamba forward pass'. mlx_lm is already a hard dependency of
    every AIR kernel in this program and carries all four classes. If this test ever
    fails the claim becomes true again -- until then it is false."""
    cov = op.runtime_coverage()
    assert cov["mlx_lm_covers_all_lake_specimens"] is True, cov["mlx_lm_module_present"]
    # ...AND THAT IS THE WEAKER CLAIM. Measured one block later: 3 of 4 actually
    # build and run; qwen3_vl_moe imports and does not construct. The import check
    # was right about three specimens and WRONG ABOUT ONE.
    assert op.execution_coverage()["built_and_ran"] == 3


def test_HAWKING_S_OWN_RUNTIME_still_covers_none_of_them():
    """The other half, and it is why the sentence sounded right: the Rust body really
    does dispatch only on llama/mistral/qwen2 families. Both halves are true and the
    receipts stated one."""
    assert op.runtime_coverage()["hawking_rust_covers_any_lake_specimen"] is False
    assert "qwen3_moe" not in op.HAWKING_RUST_ARCHS
    assert "falcon_h1" not in op.HAWKING_RUST_ARCHS


def test_the_gate_names_STORAGE_not_a_missing_reader():
    """A gap named 'no reader' invites weeks of Rust; a gap named 'contended bus'
    invites a quiesced window. Naming the wrong one parks the obligation forever."""
    cov = op.runtime_coverage()
    assert cov["blocked_on"] == "FAST_LOCAL_STORAGE"
    assert "reader" in cov["blocked_on_detail"]      # says what it is NOT


def test_IMPORTS_AND_EXECUTES_DISAGREE_ON_EXACTLY_ONE_SPECIMEN():
    """The whole reason the stronger check had to be run. If these two ever agree
    again the disagreement has been fixed or reintroduced, and either way somebody
    should look rather than inheriting a stale claim."""
    imports = op.runtime_coverage()["mlx_lm_module_present"]
    ex = op.execution_coverage()["measured"]
    assert all(imports.values()), imports
    disagree = [k for k, v in ex.items() if not v["builds"]]
    assert disagree == ["Qwen3-VL-30B-A3B"], disagree
    assert "tie_word_embeddings" in op.VL_GAP     # the gap is NAMED, not a shrug


def test_a_real_specimen_ACTUALLY_BUILDS_AND_RUNS_not_just_a_recorded_string():
    """execution_coverage returns recorded evidence, which is a table somebody could
    edit into agreement with anything. This rebuilds ONE specimen from its real
    config and pushes a token through, so the table has something behind it.
    Falcon-H1 is chosen because its config is on LOCAL storage -- the lake bus is
    the named gate and a test must not fight the operator's fill."""
    import json, importlib, pathlib
    cfg_path = pathlib.Path.home() / "noetic/stage/falcon-h1-7b/config.json"
    if not cfg_path.is_file():
        import pytest
        pytest.skip(f"{cfg_path} absent -- staged config is the input this needs")
    import mlx.core as mx
    cfg = json.loads(cfg_path.read_text())
    cfg["num_hidden_layers"] = 1                  # one layer: the graph, not the weights
    mod = importlib.import_module("mlx_lm.models." + cfg["model_type"])
    model = mod.Model(mod.ModelArgs.from_dict(cfg))
    y = model(mx.array([[1, 2, 3, 4]])); mx.eval(y)
    assert y.shape[:2] == (1, 4) and y.shape[2] > 1000
    assert bool(mx.all(mx.isfinite(y)).item()), "logits must be finite"
