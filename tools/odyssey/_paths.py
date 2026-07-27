"""Shared path constants for Odyssey T0 tools."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ODYSSEY = ROOT / "odyssey"
T0_DIR = ODYSSEY / "t0"
T0_STATE = T0_DIR / "state"
HIDDEN_DIR = ODYSSEY / "evaluation" / "hidden"
PUBLIC_EVAL_DIR = ODYSSEY / "t0" / "public_eval"
CHECKPOINTS = ODYSSEY / "checkpoints"

# Sealed substrate facts (verified live; do not re-derive).
MATH_ARTIFACT = Path.home() / (
    "Library/Application Support/Hawking/Models/GLM-5.2/"
    "b4734de4facf877f85769a911abafc5283eab3d9/"
    "GLM-5.2-H0.98-Math-Preserve.gravity"
)
EXPECTED_INDEX_SHA256 = "33d40c254eb982d4a495f5f0792a116e9d9810d937f5f3969f4f84742b2364d9"
EXPECTED_MANIFEST_SHA256 = "b34596f5d4df0b09903845302648736ee2345d7662688176c851a4d749211a83"
EXPECTED_SHARD_COUNT = 282
EXPECTED_DECISION_COUNT = 59585
EXPECTED_BPW = 0.9774017488417455
EXPECTED_BYTES = 92_038_250_160

FENCE = ODYSSEY / "launch" / "ODYSSEY_LAUNCH_AUTHORIZED"
STOP = ODYSSEY / "launch" / "STOP"
