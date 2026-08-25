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
