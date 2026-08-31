#!/usr/bin/env python3
"""What one decoded token actually costs, measured from the serving resident.

This exists because the numbers the campaign was reasoning from could not all
be true at once, and the release-profile resident settles which one was wrong.

Every figure here comes from ONE request against a `--profile release` build of
ascension_qwen38_resident (N=128, prompt 12 tokens). Timing was taken while four
Grok lanes ran at maximum, so it is DIAGNOSTIC_RELATIVE, not protected. Counts
and byte ledgers are exact regardless of contention; only the ns figures carry
the contamination caveat.

    python3 tools/future/resident_token_budget.py --record
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402

RECEIPT = REPO / "receipts" / "future" / "RESIDENT_TOKEN_BUDGET.json"

# --- the single observation everything below is derived from ---------------
OBS: dict[str, Any] = {
    "binary": "workspace/ops/build/rust/release/examples/ascension_qwen38_resident",
    "profile": "release (lto=fat, codegen-units=1) — the profile Cargo.toml says to benchmark with",
    "resident_identity": "sealed-3.14",
    "artifact_root": "/Users/scammermike/noetic/NOETIC_PARENT_A",
    "prompt_tokens": 12,
    "generated_tokens": 128,
    "decode_steps": 127,
    "fallbacks": 0,
    "decode_tps": 32.594308018571425,
    "complete_tps": 30.008027149606896,
    "dispatches_total": 133996,
    "dispatches_per_generated_token": 1046.84375,
    "prefill_dispatches": 11568,
    "decode_dispatches": 122428,
    "decode_gpu_ns": 3776318607,
    "decode_wall_ns": 3896385833,
    "wall_minus_gpu_ns_total": 137471693,
    "gpu_ns_per_generated_token": 32250937.9453125,
    "active_bytes_per_token": 10727793881.75,
    "resident_weight_bytes": 10554259456,
    "workspace_resident_bytes": 1232327492,
    "actual_read_bytes_per_token": None,
    "actual_read_bytes_status": "NOT_MEASURED_NO_METAL_MEMORY_COUNTER",
    "contamination": "4 Grok delegate lanes at maximum effort; CPU contended, GPU otherwise idle",
}

CLEAN_GEMV_GB_S = 703.5
PUBLISHED_PEAK_GB_S = 819.0
MARGINAL_DISPATCH_US = 15.0   # from the L40 host/catalog ceremony probe


def derived() -> dict[str, Any]:
    steps = OBS["decode_steps"]
    wall = OBS["decode_wall_ns"] / steps
    gpu = OBS["decode_gpu_ns"] / steps
    disp = OBS["decode_dispatches"] / steps
    gb = OBS["active_bytes_per_token"] / 1e9
    return {
        "decode_wall_ns_per_token": round(wall),
        "decode_wall_ms_per_token": round(wall / 1e6, 3),
        "decode_gpu_ns_per_token": round(gpu),
        "decode_gpu_ms_per_token": round(gpu / 1e6, 3),
        "host_gap_ms_per_token": round((wall - gpu) / 1e6, 3),
        "host_gap_share": round((wall - gpu) / wall, 4),
        "dispatches_per_decode_step": round(disp, 2),
        "dispatches_per_prefill_step": round(OBS["prefill_dispatches"] / OBS["prompt_tokens"], 2),
        "active_gb_per_token": round(gb, 4),
        "implied_effective_bandwidth_gb_s": round(gb / (wall / 1e9), 1),
        "roof_tps_at_clean_gemv": round(CLEAN_GEMV_GB_S / gb, 2),
        "roof_tps_at_published_peak": round(PUBLISHED_PEAK_GB_S / gb, 2),
    }


def findings() -> list[dict[str, Any]]:
    d = derived()
    disp = d["dispatches_per_decode_step"]
    additive_ms = disp * MARGINAL_DISPATCH_US / 1000.0
    return [
        {
            "id": "DISPATCH_COUNT_IS_964_PER_DECODE_STEP",
            "what": (
                f"{disp} dispatches per decoded token, and the same {d['dispatches_per_prefill_step']} "
                "per prefill step. Not 'hundreds' — about a thousand."
            ),
            "why_it_matters": (
                "This is the number S017 §5 asked for and it had never been "
                "measured on the serving binary, because the serving binary "
                "could not report it."
            ),
            "measured": True,
            "exact_regardless_of_contention": True,
        },
        {
            "id": "DISPATCH_COST_IS_NOT_ADDITIVE_HOST_TIME",
            "what": (
                f"At the L40 marginal figure of {MARGINAL_DISPATCH_US} us, "
                f"{disp} dispatches would be {additive_ms:.2f} ms, i.e. "
                f"{additive_ms / d['decode_wall_ms_per_token'] * 100:.0f}% of the "
                f"{d['decode_wall_ms_per_token']} ms token. But wall minus GPU is "
                f"only {d['host_gap_ms_per_token']} ms "
                f"({d['host_gap_share'] * 100:.1f}%)."
            ),
            "conclusion": (
                "The 15 us marginal cost does NOT apply additively to this class. "
                "Host submission is at most 3% of the token; the dispatch cost, "
                "if any, is inside gpu_ns as per-dispatch launch and teardown on "
                "device, not as host gap. Multiplying count by marginal cost "
                "would have produced a 47% attribution that the same run refutes."
            ),
            "supersedes": "any budget line that charges 964 x 15 us to the host",
        },
        {
            "id": "ACTIVE_BYTES_PER_TOKEN_IS_10.73_GB_NOT_9.88",
            "what": (
                f"The serving resident reports {OBS['active_bytes_per_token']:.0f} "
                f"active bytes per generated token ({d['active_gb_per_token']} GB). "
                "The anchor the ladder was built on was 9,878,901,136 — 8.6% low."
            ),
            "consequence": (
                f"The clean-GEMV roof is {d['roof_tps_at_clean_gemv']} TPS, not 71. "
                f"At the published 819 GB/s peak it is {d['roof_tps_at_published_peak']} TPS. "
                "71 is therefore NOT reachable by executor recovery alone — it is "
                "above the clean-addressing roof for this byte count and needs "
                "bytes to fall as well."
            ),
            "resolves": (
                "the 4% anchor inconsistency recorded OPEN in "
                "receipts/future/TOKEN_NS_OBJECTIVE.json — in the other direction. "
                f"10.7278 GB x 32.594 TPS = {d['implied_effective_bandwidth_gb_s']} GB/s, "
                "which matches the implied figure, so the outlier was the recorded "
                "337.3 GB/s and the low byte count, not the TPS."
            ),
        },
        {
            "id": "BUILD_PROFILE_IS_NOT_THE_TPS_LEVER",
            "what": (
                "release (lto=fat, cgu=1) decoded at 32.594 TPS; release-fast "
                "(lto=false, cgu=16) decoded at 32.774 TPS on the same box "
                "minutes apart. The difference is within run-to-run noise and "
                "the wrong sign for the 'benchmark profile is faster' story."
            ),
            "conclusion": (
                "The Cargo.toml warning is still correct policy — benchmark on "
                "release — but the profile was NOT hiding TPS. This resident is "
                "GPU-bound: gpu_ns is "
                f"{(1 - d['host_gap_share']) * 100:.1f}% of wall, so host codegen "
                "quality has almost nothing to work on."
            ),
        },
        {
            "id": "THE_TOKEN_IS_GPU_BOUND",
            "what": (
                f"{d['decode_gpu_ms_per_token']} ms of the "
                f"{d['decode_wall_ms_per_token']} ms token is GPU time. Host gap "
                f"is {d['host_gap_ms_per_token']} ms."
            ),
            "consequence": (
                "CPU submission, command buffer construction and host-side "
                "ceremony together cannot account for more than ~3% of the "
                "token. Any remaining win has to come from inside the GPU time: "
                "fewer bytes, fewer dispatches, or cheaper arithmetic per byte."
            ),
        },
    ]


def budget() -> dict[str, Any]:
    """The categories S017 §2 asked for. UNKNOWN stays UNKNOWN."""
    d = derived()
    return {
        "total_decode_ms_per_token": d["decode_wall_ms_per_token"],
        "categories": {
            "gpu_time_total": {
                "ms": d["decode_gpu_ms_per_token"],
                "share": round(1 - d["host_gap_share"], 4),
                "status": "MEASURED",
            },
            "host_gap_total": {
                "ms": d["host_gap_ms_per_token"],
                "share": d["host_gap_share"],
                "status": "MEASURED",
                "includes": "CPU submission, command buffer construction, "
                            "synchronization the GPU did not cover, python-side "
                            "protocol handling",
                "not_decomposed_further": True,
            },
            "weight_bytes_time": {
                "status": "UNKNOWN",
                "why": "actual_read_bytes_per_token is NOT_MEASURED_NO_METAL_"
                       "MEMORY_COUNTER. active_bytes_per_token is the catalog "
                       "sum, not what the GPU read. Without a memory counter "
                       "the byte-time cannot be separated from arithmetic time.",
                "bound": "at the clean-GEMV 703.5 GB/s, 10.7278 GB would take "
                         "15.25 ms, which is 50% of the token. That is a LOWER "
                         "BOUND on byte time only if every active byte is "
                         "actually read once.",
            },
            "mlp_bytes_time": {"status": "UNKNOWN", "why": "needs the per-organ "
                               "census; MLP share of bytes is recorded elsewhere "
                               "as ~54% but has not been re-derived against the "
                               "10.7278 GB figure"},
            "attention_bytes_time": {"status": "UNKNOWN"},
            "state_bytes_time": {"status": "UNKNOWN"},
            "other_model_bytes_time": {"status": "UNKNOWN"},
            "low_bit_decode_cost": {"status": "UNKNOWN", "why": "inside gpu_ns, "
                                    "not separable without per-kernel timing"},
            "useful_arithmetic": {"status": "UNKNOWN", "same_reason": True},
            "dispatch_cost": {
                "status": "BOUNDED_NOT_ATTRIBUTED",
                "count_per_token": d["dispatches_per_decode_step"],
                "upper_bound_if_host_side_ms": d["host_gap_ms_per_token"],
                "why": "964 dispatches is exact. Their cost is not: the host gap "
                       "bounds any host-side portion at 0.945 ms, and any "
                       "on-device launch cost is inside gpu_ns and unseparated.",
            },
            "command_encoder_cost": {"status": "UNKNOWN", "bounded_by": "host_gap_total"},
            "synchronization": {"status": "UNKNOWN", "bounded_by": "host_gap_total"},
            "cpu_submission": {"status": "UNKNOWN", "bounded_by": "host_gap_total"},
            "state_transition": {"status": "UNKNOWN"},
            "final_head_and_sampling": {"status": "UNKNOWN"},
        },
        "honest_remainder": (
            "Two categories are measured (GPU time, host gap) and they sum to "
            "the whole token. Everything inside GPU time is UNKNOWN and stays "
            "UNKNOWN: this run had no per-kernel timing and no Metal memory "
            "counter. That is a 96.9% unattributed remainder, and padding it "
            "into invented categories would make the receipt worse, not better."
        ),
        "what_would_close_it": [
            "per-kernel GPU timestamps, which separate byte time from arithmetic",
            "a Metal memory counter or a counted-read instrumented kernel, which "
            "turns actual_read_bytes_per_token from null into a number",
            "the per-organ byte census, which splits the byte time by organ",
        ],
    }


def build() -> dict[str, Any]:
    return {
        "schema": "hawking.future.resident_token_budget.v1",
        "version": 1,
        "recorded_by": "tools/future/resident_token_budget.py",
        "evidence_class": "DIAGNOSTIC_RELATIVE",
        "gpu_authority": False,
        "took_gpu_lease": False,
        "observation": OBS,
        "derived": derived(),
        "findings": findings(),
        "budget": budget(),
        "claim_boundary": (
            "One request against one build on a CPU-contended box. Dispatch "
            "counts and byte ledgers are exact and contention-independent. Every "
            "ns figure is DIAGNOSTIC_RELATIVE and must not be quoted as a "
            "protected absolute measurement. The budget attributes only what the "
            "run separated: GPU time and host gap. It does not claim to know "
            "what the GPU time is made of."
        ),
    }


def record() -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(build(), indent=1, sort_keys=True, default=str) + "\n")
    return RECEIPT


if __name__ == "__main__":
    if "--record" in sys.argv:
        print(f"wrote {record()}")
    else:
        print(json.dumps(build(), indent=1, default=str))
