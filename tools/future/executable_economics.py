#!/usr/bin/env python3
"""EXECUTABLE ECONOMICS — score a representation before anyone fits it.

A compression ratio without an execution story is not a candidate. This
module is the cost model that says, in advance, whether a proposed
representation can mathematically matter on the measured token.

Input: bytes removed, bytes added (generator, embeddings, residuals,
metadata, state), extra FLOPs per output element, dispatch delta, the
consuming primitive, and which bandwidth regime that primitive
plausibly runs in.

Output: predicted ms/token delta, predicted TPS, and MATERIAL or
IMMATERIAL against the S020 §20 bar — substantial work is deserved if
the candidate plausibly removes >= 1% of complete token time, or
creates a reusable representation family, or provides a high-information
falsifier.

A new representation may run in a different bandwidth regime than
affine-Q2. That is carried as a stated ASSUMPTION with a range, not a
point estimate dressed as a prediction.

    python3 tools/future/executable_economics.py --record
    python3 -m pytest tools/future/test_executable_economics.py -q

evidence_class STATIC_ONLY. No GPU. No bench lock. Does not touch crates/.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.future import causal_budget_71 as cb
from tools.future._common import REPO, load_json, write_receipt
from tools.future.physical_primitives import ATLAS_PRIMITIVES


RECEIPT = "EXECUTABLE_ECONOMICS.json"
SCHEMA = "hawking.future.executable_economics.v1"
VERSION = 1
RECORDED_BY = "tools/future/executable_economics.py"
EVIDENCE_CLASS = "STATIC_ONLY"

AUX_REL = "receipts/future/MLP_AUXILIARY_INFORMATION.json"
CODE_REL = "receipts/future/MLP_CODE_INFORMATION.json"
DN_REL = "receipts/future/DELTANET_REPRESENTATION.json"
SWEEP_REL = "receipts/future/DISPATCH_SIZE_SWEEP.json"
BA_REL = "receipts/future/BA_DELTA_AB.json"

# ---------------------------------------------------------------------------
# Cited measured constants. Arithmetic over these is STATIC_ONLY. None of
# them are re-measured here.
# ---------------------------------------------------------------------------

CITED_TOKEN_MS = 28.722
CITED_HOST_MS = cb.HOST_GAP_MS  # 0.989
CITED_GPU_MS = 27.733
TPS_71_TOKEN_MS = 1000.0 / 71.0  # 14.085
TPS_71_GPU_MS = TPS_71_TOKEN_MS - CITED_HOST_MS  # 13.096
GPU_REDUCTION_FOR_71 = 1.0 - TPS_71_GPU_MS / CITED_GPU_MS  # 52.8%

CLEAN_GEMV_GB_S = cb.CLEAN_GEMV_GB_S  # 703.5
LM_HEAD_GB_S = cb.DEMONSTRATED_GB_S  # 497.4

# Affine-Q2 size curve (DISPATCH_SIZE_SWEEP). Saturates ~377 at any
# per-dispatch size past the 20 MB knee. Does not reach the LM head.
AFFINE_Q2_GB_S_AT_5MB = 223.4
AFFINE_Q2_GB_S_AT_20MB = 325.9
AFFINE_Q2_GB_S_AT_338MB = 374.4
AFFINE_Q2_GB_S_AT_700MB = 376.7
AFFINE_Q2_SATURATED_GB_S = 376.7

MARGINAL_US_MLP_GQA_NORM = 6.25
MARGINAL_US_DELTANET_BA = 2.884

S020_SECTION_20_BAR_FRAC = 0.01
S020_SECTION_20_BAR_MS = CITED_TOKEN_MS * S020_SECTION_20_BAR_FRAC

# Geometry of sealed-3.14. Same numbers as representation_decode_fusion /
# mlp_auxiliary_information.
HIDDEN = 5120
INTERMEDIATE = 17408
LAYERS = 64
DN_LAYERS = 48
GQA_LAYERS = 16
VOCAB = 248320
QKVZ_ROWS = 16384
BA_ROWS = 96
GQA_Q_ROWS = 12288
GQA_KV_ROWS = 1024
F16_BYTES = 2

MLP_PARAMS = LAYERS * (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE)
MLP_ACTIVE_BYTES = 5_347_795_776
MLP_CODE_BYTES = 4_278_190_080
MLP_AUX_BYTES = 1_069_605_696
DN_ACTIVE_BYTES = 2_961_659_904
GQA_ACTIVE_BYTES = 891_292_160
LM_HEAD_ACTIVE_BYTES = 675_430_440
ACTIVE_BYTES = cb.ACTIVE_BYTES
MLP_MS = 15.541
DN_MS = 8.227
GQA_MS = 2.607
LM_HEAD_MS = 1.358
MLP_GB_S = 344.1
DN_GB_S = 360.0
GQA_GB_S = 341.9

ORGAN_BYTES: dict[str, int] = {
    "mlp": MLP_ACTIVE_BYTES,
    "deltanet": DN_ACTIVE_BYTES,
    "gqa": GQA_ACTIVE_BYTES,
    "lm_head": LM_HEAD_ACTIVE_BYTES,
    "token": ACTIVE_BYTES,
}
ORGAN_MS: dict[str, float] = {
    "mlp": MLP_MS,
    "deltanet": DN_MS,
    "gqa": GQA_MS,
    "lm_head": LM_HEAD_MS,
    "token": CITED_GPU_MS,
}
ORGAN_GB_S: dict[str, float] = {
    "mlp": MLP_GB_S,
    "deltanet": DN_GB_S,
    "gqa": GQA_GB_S,
    "lm_head": LM_HEAD_GB_S,
    "token": ACTIVE_BYTES / 1e9 / (CITED_GPU_MS / 1000.0),
}
ORGAN_OUTPUT_ELEMENTS: dict[str, int] = {
    "mlp": LAYERS * (INTERMEDIATE + INTERMEDIATE + HIDDEN),
    "deltanet": DN_LAYERS * (QKVZ_ROWS + HIDDEN + BA_ROWS),
    "gqa": GQA_LAYERS * (GQA_Q_ROWS + GQA_KV_ROWS + GQA_KV_ROWS + HIDDEN),
    "lm_head": VOCAB,
}
ORGAN_OUTPUT_ELEMENTS["token"] = sum(
    ORGAN_OUTPUT_ELEMENTS[k] for k in ("mlp", "deltanet", "gqa", "lm_head")
)
ORGAN_PARAMS: dict[str, int] = {
    "mlp": MLP_PARAMS,
    "lm_head": VOCAB * HIDDEN,
}

# Effective GEMV-dressed FLOP rate of the MLP organ. Extra FLOPs are
# scored at this rate as an ASSUMPTION: same cost-per-FLOP as the
# incumbent MAC, which is itself mostly wait-on-memory. Hidden-under-
# bandwidth is 0; decode-ALU-like is FLOP_EXPOSED_MULT times this.
EFFECTIVE_FLOP_S = (2.0 * MLP_PARAMS) / (MLP_MS / 1000.0)
FLOP_EXPOSED_MULT = 4.0

BYTES_ADDED_FIELDS: tuple[str, ...] = (
    "generator",
    "embeddings",
    "residuals",
    "metadata",
    "state",
)

LIVE_STATUSES = frozenset({"OPEN", "UNMEASURED", "EXISTING_LEVER"})
DEAD_STATUSES = frozenset(
    {
        "MEASURED_NEGATIVE",
        "ALREADY_FALSIFIED",
        "REJECTED_DENSE_REMAT",
        "AT_THE_FLOOR",
        "PER_DISPATCH_SIZE_REFUTED",
        "GRANULARITY_REFUTED",
        "BOUNDED_TOO_SMALL",
        "ALREADY_FUSED",
        "CLOSED",
    }
)

# Packing / codec / operator families that would transfer, not one-off hacks.
REUSABLE_FAMILY_IDS = frozenset(
    {
        "quantize_aux_u8",
        "larger_group_size",
        "group_size_256",
        "group_size_512",
        "group_size_1024",
        "entropy_coded_code_stream",
        "lower_bit_native",
        "heterogeneous_bit_allocation",
        "heterogeneous_qkvz_bits",
        "lower_bit_uniform_qkvz",
        "lower_bit_out_proj",
        "larger_q4_group",
        "function_replacement",
        "shared_input_transforms",
        "generated_programs",
        "generated_coefficients",
        "factorized_programs",
        "factorized_qkvz",
        "structured_transition_state",
        "recurrent_state_replacement",
        "lower_bit_recurrent_state",
        "direct_state_machine",
        "surviving_dispatch_size_amortize_sub20mb",
    }
)

# Cheap experiments that would teach a whole school, not just one row.
HIGH_INFORMATION_FALSIFIER_IDS = frozenset(
    {
        "surviving_dispatch_size_amortize_sub20mb",
        "dispatch_size_concat_to_lm_head_mb",
        "entropy_coded_code_stream",
        "function_replacement",
        "shared_input_transforms",
        "direct_state_machine",
        "heterogeneous_qkvz_bits",
    }
)

# Group-size curve points that causal_budget_71 ranked separately.
# larger_group_size in the aux receipt is G=128 (534,773,760).
GROUP_SIZE_EXTRA: tuple[tuple[int, int], ...] = (
    (256, 802_160_640),
    (512, 935_854_080),
    (1024, 1_002_700_800),
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement and no generate gate. "
    "Predicted ms/token and predicted TPS are arithmetic over cited organ "
    "times, cited byte shares, cited dispatch-class costs and a stated "
    "bandwidth-regime ASSUMPTION with a range. They are not a protected "
    "measurement and not a promise that a fit will hold capability. A new "
    "representation is not assumed to run at the affine-Q2 saturation "
    "(~377 GB/s), the LM-head demonstrated 497.4 GB/s, or the clean GEMV "
    "roof 703.5 GB/s: those are regimes, and the regime is an input. "
    "evidence_class is STATIC_ONLY. gpu_authority is false."
)

_UNSET: Any = object()


class IncompleteEconomics(ValueError):
    """bytes_removed without bytes_added is not a candidate."""


class EconomicsRefuse(ValueError):
    """The cost model refused rather than guessing."""


# ---------------------------------------------------------------------------
# Bandwidth regimes. Each is a named ASSUMPTION with a range. Affine-Q2
# saturates near 377 GB/s at any per-dispatch size; the LM head reaches
# 497.4 and nothing else has been shown to; the clean GEMV roof is 703.5.
# ---------------------------------------------------------------------------

BANDWIDTH_REGIMES: dict[str, dict[str, Any]] = {
    "production_mlp": {
        "gb_s_lo": MLP_GB_S,
        "gb_s_hi": AFFINE_Q2_SATURATED_GB_S,
        "gb_s_nominal": MLP_GB_S,
        "measured": True,
        "assumption": (
            "nominal is the production MLP organ rate. The high end is the "
            "affine-Q2 size-curve saturation, which isolated launches reach "
            "and production has not. Not 497.4, not 703.5."
        ),
        "source": "receipts/future/ORGAN_BANDWIDTH.json + DISPATCH_SIZE_SWEEP.json",
    },
    "production_deltanet": {
        "gb_s_lo": DN_GB_S,
        "gb_s_hi": LM_HEAD_GB_S,
        "gb_s_nominal": DN_GB_S,
        "measured": True,
        "assumption": (
            "nominal is the production DeltaNet organ rate (Q4). The high "
            "end is the LM head's demonstrated 497.4, also Q4, which DeltaNet "
            "has not been shown to reach. Including 497.4 in the range is an "
            "ASSUMPTION, not a prediction."
        ),
        "source": "receipts/future/ORGAN_BANDWIDTH.json",
    },
    "production_gqa": {
        "gb_s_lo": GQA_GB_S,
        "gb_s_hi": LM_HEAD_GB_S,
        "gb_s_nominal": GQA_GB_S,
        "measured": True,
        "assumption": (
            "nominal is production GQA. High end is the LM-head demonstrated "
            "rate; GQA has not been shown to reach it."
        ),
        "source": "receipts/future/ORGAN_BANDWIDTH.json",
    },
    "production_lm_head": {
        "gb_s_lo": LM_HEAD_GB_S,
        "gb_s_hi": LM_HEAD_GB_S,
        "gb_s_nominal": LM_HEAD_GB_S,
        "measured": True,
        "assumption": "the one organ that has been shown to reach 497.4 GB/s",
        "source": "receipts/future/ORGAN_BANDWIDTH.json",
    },
    "affine_q2_family": {
        "gb_s_lo": MLP_GB_S,
        "gb_s_hi": AFFINE_Q2_SATURATED_GB_S,
        "gb_s_nominal": MLP_GB_S,
        "measured": True,
        "assumption": (
            "production-shaped affine-Q2. The size curve saturates at "
            f"{AFFINE_Q2_SATURATED_GB_S} GB/s and does not rise toward "
            f"{LM_HEAD_GB_S}. A 5 MB launch is {AFFINE_Q2_GB_S_AT_5MB} GB/s; "
            "that is a different regime (affine_q2_unamortized)."
        ),
        "source": "receipts/future/DISPATCH_SIZE_SWEEP.json",
    },
    "affine_q2_unamortized": {
        "gb_s_lo": AFFINE_Q2_GB_S_AT_5MB,
        "gb_s_hi": AFFINE_Q2_GB_S_AT_20MB,
        "gb_s_nominal": AFFINE_Q2_GB_S_AT_5MB,
        "measured": True,
        "assumption": (
            "affine-Q2 launches around 5 MB. The surviving dispatch-size "
            "lever is this amortization to >= 20 MB, not the 350->497 gap."
        ),
        "source": "receipts/future/DISPATCH_SIZE_SWEEP.json",
    },
    "affine_q2_saturated": {
        "gb_s_lo": AFFINE_Q2_GB_S_AT_338MB,
        "gb_s_hi": AFFINE_Q2_GB_S_AT_700MB,
        "gb_s_nominal": AFFINE_Q2_SATURATED_GB_S,
        "measured": True,
        "assumption": (
            "the affine-Q2 family ceiling. Concatenating to the LM head's "
            "337.7 MB working set still sits here, not at 497.4."
        ),
        "source": "receipts/future/DISPATCH_SIZE_SWEEP.json",
    },
    "organ_cluster": {
        "gb_s_lo": GQA_GB_S,
        "gb_s_hi": DN_GB_S,
        "gb_s_nominal": 350.0,
        "measured": True,
        "assumption": "MLP, DeltaNet and GQA sit inside 5% of each other",
        "source": "receipts/future/ORGAN_BANDWIDTH.json",
    },
    "lm_head_demonstrated": {
        "gb_s_lo": LM_HEAD_GB_S,
        "gb_s_hi": LM_HEAD_GB_S,
        "gb_s_nominal": LM_HEAD_GB_S,
        "measured": True,
        "assumption": (
            "ASSUMPTION if applied to any organ other than the LM head. "
            "No other organ has been shown to reach 497.4 GB/s. Affine-Q2 "
            f"saturates at {AFFINE_Q2_SATURATED_GB_S}."
        ),
        "source": "receipts/future/ORGAN_BANDWIDTH.json",
    },
    "clean_gemv_roof": {
        "gb_s_lo": CLEAN_GEMV_GB_S,
        "gb_s_hi": CLEAN_GEMV_GB_S,
        "gb_s_nominal": CLEAN_GEMV_GB_S,
        "measured": True,
        "assumption": (
            "ASSUMPTION. Clean single-GEMV roof, not a packed-representation "
            "rate. Beating the LM head as well. Not a prediction."
        ),
        "source": "receipts/future/ORGAN_BANDWIDTH.json",
    },
    "unknown_new_representation": {
        "gb_s_lo": AFFINE_Q2_GB_S_AT_5MB,
        "gb_s_hi": LM_HEAD_GB_S,
        "gb_s_nominal": 350.0,
        "measured": False,
        "assumption": (
            "ASSUMPTION with a range, not a point estimate. A new "
            "representation is bounded below by the slowest measured "
            f"affine-Q2 launch ({AFFINE_Q2_GB_S_AT_5MB} GB/s at 5 MB) and "
            f"above by the fastest demonstrated packed GEMV ({LM_HEAD_GB_S} "
            "GB/s, the LM head). The clean GEMV roof is excluded: nothing "
            "packed has been shown to reach it."
        ),
        "source": "DISPATCH_SIZE_SWEEP.json + ORGAN_BANDWIDTH.json",
    },
}

DISPATCH_CLASSES: dict[str, dict[str, Any]] = {
    "mlp_gqa_norm_fusion": {
        "us": MARGINAL_US_MLP_GQA_NORM,
        "source": "receipts/future/RESIDENT_TOKEN_BUDGET.json derived.measured_marginal_dispatch_us",
    },
    "deltanet_ba": {
        "us": MARGINAL_US_DELTANET_BA,
        "source": "receipts/future/BA_DELTA_AB.json derived.us_per_removed_dispatch",
    },
}


def _r(value: float, n: int = 6) -> float:
    out = round(float(value), n)
    return 0.0 if out == 0.0 else out


def bytes_to_ms(n_bytes: float, gb_s: float) -> float:
    """Milliseconds to move n_bytes at gb_s GB/s."""
    if gb_s <= 0.0:
        raise EconomicsRefuse(f"bandwidth must be positive, got {gb_s}")
    return float(n_bytes) / float(gb_s) * 1e-6


def tps_from_ms(ms: float) -> float:
    if ms <= 0.0:
        raise EconomicsRefuse(f"token time must be positive, got {ms}")
    return 1000.0 / float(ms)


def normalize_bytes_added(bytes_added: Any) -> dict[str, int]:
    """Canonical five-field added-byte ledger. Total is the sum."""
    out = {k: 0 for k in BYTES_ADDED_FIELDS}
    if isinstance(bytes_added, bool):
        raise IncompleteEconomics(
            "bytes_added must be a number or a mapping of "
            + ", ".join(BYTES_ADDED_FIELDS)
        )
    if isinstance(bytes_added, (int, float)):
        if bytes_added < 0:
            raise EconomicsRefuse(f"bytes_added cannot be negative: {bytes_added}")
        out["generator"] = int(bytes_added)
        out["total"] = int(bytes_added)
        return out
    if not isinstance(bytes_added, Mapping):
        raise IncompleteEconomics(
            "bytes_added must be supplied as a number or a mapping of "
            f"{BYTES_ADDED_FIELDS}; a ratio without executable economics "
            "is not a candidate"
        )
    total_check = 0
    for key, raw in bytes_added.items():
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise EconomicsRefuse(f"bytes_added[{key!r}] is not a number: {raw!r}")
        if raw < 0:
            raise EconomicsRefuse(f"bytes_added[{key!r}] cannot be negative: {raw}")
        n = int(raw)
        total_check += n
        if key in out:
            out[key] += n
        else:
            # Extra named buckets (e.g. remat_dense) still count.
            out[str(key)] = out.get(str(key), 0) + n
    out["total"] = sum(int(v) for k, v in out.items() if k != "total")
    if out["total"] != total_check:
        out["total"] = total_check
    return out


def infer_regime(
    *,
    organ: str,
    consuming_primitive: str | None,
    bandwidth_regime: str | None,
) -> str:
    if bandwidth_regime:
        if bandwidth_regime not in BANDWIDTH_REGIMES:
            raise EconomicsRefuse(
                f"unknown bandwidth_regime {bandwidth_regime!r}. "
                f"known: {sorted(BANDWIDTH_REGIMES)}"
            )
        return bandwidth_regime
    if consuming_primitive is None or consuming_primitive not in ATLAS_PRIMITIVES:
        return "unknown_new_representation"
    if consuming_primitive in {"TiledProjection", "DirectRoutedAccumulate", "MoveOrRecompute"}:
        return "unknown_new_representation"
    if consuming_primitive == "LocalStateMachine":
        return "production_deltanet" if organ in {"deltanet", "token"} else "unknown_new_representation"
    if consuming_primitive == "LayoutTransform":
        return {
            "mlp": "production_mlp",
            "deltanet": "production_deltanet",
            "gqa": "production_gqa",
            "lm_head": "production_lm_head",
        }.get(organ, "unknown_new_representation")
    # FusedDecodeCompute and the rest of the packed-GEMV family.
    return {
        "mlp": "affine_q2_family",
        "deltanet": "production_deltanet",
        "gqa": "production_gqa",
        "lm_head": "production_lm_head",
    }.get(organ, "unknown_new_representation")


def infer_dispatch_class(organ: str, dispatch_class: str | None) -> str | None:
    if dispatch_class:
        if dispatch_class not in DISPATCH_CLASSES:
            raise EconomicsRefuse(
                f"unknown dispatch_class {dispatch_class!r}. "
                f"known: {sorted(DISPATCH_CLASSES)}"
            )
        return dispatch_class
    if organ in {"mlp", "gqa"}:
        return "mlp_gqa_norm_fusion"
    if organ == "deltanet":
        return "deltanet_ba"
    return None


def _flop_ms(
    extra_flops_per_output_element: float,
    n_output_elements: int,
    *,
    exposed_mult: float = 1.0,
) -> float:
    if extra_flops_per_output_element == 0.0:
        return 0.0
    extra = float(extra_flops_per_output_element) * int(n_output_elements)
    return extra / EFFECTIVE_FLOP_S * 1000.0 * float(exposed_mult)


def score(
    *,
    bytes_removed: Any = _UNSET,
    bytes_added: Any = _UNSET,
    extra_flops_per_output_element: float = 0.0,
    dispatch_delta: float = 0.0,
    consuming_primitive: str | None = None,
    bandwidth_regime: str | None = None,
    organ: str = "mlp",
    n_output_elements: int | None = None,
    dispatch_class: str | None = None,
    retime_organ: bool = False,
    reusable_family: bool = False,
    high_information_falsifier: bool = False,
    candidate_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Score a proposed representation. Does not fit anything.

    Refuses if bytes_removed is supplied without bytes_added: a ratio
    without executable economics is not a candidate.
    """
    if bytes_removed is not _UNSET and bytes_added is _UNSET:
        raise IncompleteEconomics(
            "bytes_removed without bytes_added: a ratio without executable "
            "economics is not a candidate"
        )
    if bytes_removed is not _UNSET and bytes_added is None:
        raise IncompleteEconomics(
            "bytes_removed without bytes_added: a ratio without executable "
            "economics is not a candidate"
        )

    if bytes_removed is _UNSET or bytes_removed is None:
        removed = 0
    else:
        if isinstance(bytes_removed, bool) or not isinstance(bytes_removed, (int, float)):
            raise EconomicsRefuse(f"bytes_removed is not a number: {bytes_removed!r}")
        if bytes_removed < 0:
            raise EconomicsRefuse(f"bytes_removed cannot be negative: {bytes_removed}")
        removed = int(bytes_removed)

    if bytes_added is _UNSET or bytes_added is None:
        added = normalize_bytes_added(0)
        added_was_supplied = False
    else:
        added = normalize_bytes_added(bytes_added)
        added_was_supplied = True

    if organ not in ORGAN_BYTES:
        raise EconomicsRefuse(
            f"unknown organ {organ!r}. known: {sorted(ORGAN_BYTES)}"
        )
    organ_bytes = ORGAN_BYTES[organ]
    if removed > organ_bytes:
        raise EconomicsRefuse(
            f"bytes_removed {removed} exceeds {organ} active bytes {organ_bytes}"
        )

    added_total = int(added.get("total", 0))
    net_bytes = added_total - removed

    regime_name = infer_regime(
        organ=organ,
        consuming_primitive=consuming_primitive,
        bandwidth_regime=bandwidth_regime,
    )
    regime = BANDWIDTH_REGIMES[regime_name]
    gb_s_nom = float(regime["gb_s_nominal"])
    gb_s_lo = float(regime["gb_s_lo"])
    gb_s_hi = float(regime["gb_s_hi"])
    if gb_s_lo > gb_s_hi:
        gb_s_lo, gb_s_hi = gb_s_hi, gb_s_lo

    # For same-family packing changes, remaining bytes keep the organ's
    # measured rate (causal_budget_71). For a claimed regime shift of the
    # whole organ, retime remaining bytes at the assumed rate.
    organ_gb_s = ORGAN_GB_S[organ]
    organ_ms = ORGAN_MS[organ]

    def _byte_delta_at(gb_s: float) -> float:
        if retime_organ:
            new_bytes = organ_bytes + net_bytes
            if new_bytes < 0:
                raise EconomicsRefuse("retime would produce negative organ bytes")
            return bytes_to_ms(new_bytes, gb_s) - organ_ms
        return bytes_to_ms(net_bytes, gb_s)

    # Incremental packing uses the organ's own measured rate as nominal,
    # even when the named regime's nominal differs slightly.
    rate_for_terms = gb_s_nom if retime_organ else organ_gb_s
    if retime_organ:
        byte_nom = _byte_delta_at(gb_s_nom)
        byte_at_lo = _byte_delta_at(gb_s_lo)
        byte_at_hi = _byte_delta_at(gb_s_hi)
    else:
        byte_nom = _byte_delta_at(organ_gb_s)
        byte_at_lo = _byte_delta_at(gb_s_lo)
        byte_at_hi = _byte_delta_at(gb_s_hi)
    byte_removed_ms = -bytes_to_ms(removed, rate_for_terms)
    byte_added_ms = bytes_to_ms(added_total, rate_for_terms)

    if extra_flops_per_output_element < 0:
        raise EconomicsRefuse(
            f"extra_flops_per_output_element cannot be negative: "
            f"{extra_flops_per_output_element}"
        )
    if n_output_elements is None:
        n_output_elements = ORGAN_OUTPUT_ELEMENTS[organ]
    if extra_flops_per_output_element != 0.0 and n_output_elements <= 0:
        raise EconomicsRefuse(
            "extra FLOPs require n_output_elements > 0 (pass n_output_elements "
            "or a known organ)"
        )

    flop_nom = _flop_ms(extra_flops_per_output_element, n_output_elements, exposed_mult=1.0)
    flop_hidden = 0.0
    flop_exposed = _flop_ms(
        extra_flops_per_output_element, n_output_elements, exposed_mult=FLOP_EXPOSED_MULT
    )

    dclass = infer_dispatch_class(organ, dispatch_class)
    if dispatch_delta != 0.0 and dclass is None:
        # Class unknown: carry the measured class range as ASSUMPTION.
        us_nom = 0.5 * (MARGINAL_US_MLP_GQA_NORM + MARGINAL_US_DELTANET_BA)
        us_lo, us_hi = MARGINAL_US_DELTANET_BA, MARGINAL_US_MLP_GQA_NORM
        dispatch_assumption = (
            "ASSUMPTION: dispatch class not named; range is the two measured "
            f"classes ({MARGINAL_US_DELTANET_BA} us DeltaNet BA, "
            f"{MARGINAL_US_MLP_GQA_NORM} us MLP/GQA/norm fusion)"
        )
    elif dclass is None:
        us_nom = us_lo = us_hi = 0.0
        dispatch_assumption = "no dispatch delta"
    else:
        us_nom = float(DISPATCH_CLASSES[dclass]["us"])
        us_lo = us_hi = us_nom
        dispatch_assumption = (
            f"measured class {dclass} at {us_nom} us; class-dependent, "
            "not a single marginal"
        )
    dispatch_nom = float(dispatch_delta) * us_nom / 1000.0
    dispatch_at_lo = float(dispatch_delta) * us_lo / 1000.0
    dispatch_at_hi = float(dispatch_delta) * us_hi / 1000.0

    delta_nom = byte_nom + flop_nom + dispatch_nom

    # Range: bandwidth lo/hi × flop hidden/exposed × dispatch class lo/hi.
    # "Plausibly" for the S020 time bar uses the most negative (most
    # saving) combination inside the stated assumptions.
    candidates_delta = []
    for b in (byte_at_lo, byte_at_hi, byte_nom):
        for f in (flop_hidden, flop_nom, flop_exposed):
            for d in (dispatch_at_lo, dispatch_at_hi, dispatch_nom):
                candidates_delta.append(b + f + d)
    delta_lo = min(candidates_delta)
    delta_hi = max(candidates_delta)

    token_nom = CITED_TOKEN_MS + delta_nom
    token_fast = CITED_TOKEN_MS + delta_lo
    token_slow = CITED_TOKEN_MS + delta_hi
    # Host gap is a floor: this model does not delete host work.
    token_nom = max(token_nom, CITED_HOST_MS)
    token_fast = max(token_fast, CITED_HOST_MS)
    token_slow = max(token_slow, CITED_HOST_MS)

    predicted_tps = tps_from_ms(token_nom)
    tps_fast = tps_from_ms(token_fast)
    tps_slow = tps_from_ms(token_slow)

    plausible_ms_saved = max(0.0, -delta_lo)
    clears_time_bar = plausible_ms_saved >= S020_SECTION_20_BAR_MS

    live = True if status is None else status in LIVE_STATUSES
    reasons: list[str] = []
    if live and clears_time_bar:
        reasons.append("clears_s020_section_20_time_bar")
    if live and reusable_family:
        reasons.append("reusable_representation_family")
    if live and high_information_falsifier:
        reasons.append("high_information_falsifier")
    if not live:
        reasons.append(f"not_live:{status}")

    verdict = "MATERIAL" if (live and reasons and any(
        r in {
            "clears_s020_section_20_time_bar",
            "reusable_representation_family",
            "high_information_falsifier",
        }
        for r in reasons
    )) else "IMMATERIAL"

    bandwidth_is_assumption = (not regime["measured"]) or (
        regime_name
        in {
            "unknown_new_representation",
            "lm_head_demonstrated",
            "clean_gemv_roof",
            "affine_q2_saturated",
        }
        and organ != "lm_head"
    )
    if regime_name == "lm_head_demonstrated" and organ != "lm_head":
        bandwidth_is_assumption = True
    if regime_name == "clean_gemv_roof":
        bandwidth_is_assumption = True
    if regime_name == "unknown_new_representation":
        bandwidth_is_assumption = True
    # Even a measured organ rate is an ASSUMPTION when applied to a new
    # primitive: the new kernel may not sit on that curve.
    if consuming_primitive not in {
        "FusedDecodeCompute",
        "LayoutTransform",
        None,
    } or (
        consuming_primitive == "FusedDecodeCompute"
        and organ == "mlp"
        and regime_name not in {"production_mlp", "affine_q2_family"}
    ):
        bandwidth_is_assumption = True
    if consuming_primitive and consuming_primitive not in ATLAS_PRIMITIVES:
        bandwidth_is_assumption = True

    return {
        "ok": True,
        "id": candidate_id,
        "organ": organ,
        "consuming_primitive": consuming_primitive,
        "status": status,
        "live": live,
        "verdict": verdict,
        "verdict_reasons": reasons,
        "bytes_removed": removed,
        "bytes_added": added,
        "bytes_added_supplied": added_was_supplied,
        "net_bytes": net_bytes,
        "extra_flops_per_output_element": float(extra_flops_per_output_element),
        "n_output_elements": int(n_output_elements),
        "dispatch_delta": float(dispatch_delta),
        "retime_organ": retime_organ,
        "predicted_ms_delta": delta_nom,
        "predicted_ms_delta_range": [delta_lo, delta_hi],
        "predicted_ms_saved": -delta_nom,
        "predicted_token_ms": token_nom,
        "predicted_token_ms_range": [token_fast, token_slow],
        "predicted_tps": predicted_tps,
        "predicted_tps_range": [tps_slow, tps_fast],
        "terms": {
            "byte_ms_delta": byte_nom,
            "byte_removed_ms": byte_removed_ms,
            "byte_added_ms": byte_added_ms,
            "flop_ms_delta": flop_nom,
            "dispatch_ms_delta": dispatch_nom,
        },
        "assumptions": {
            "bandwidth_regime": regime_name,
            "bandwidth_gb_s_nominal": gb_s_nom,
            "bandwidth_gb_s_range": [gb_s_lo, gb_s_hi],
            "bandwidth_is_assumption": bandwidth_is_assumption,
            "bandwidth_note": regime["assumption"],
            "incremental_byte_rate_gb_s": organ_gb_s if not retime_organ else gb_s_nom,
            "flop_model": (
                "extra FLOPs scored as extra_flops * n_output_elements / "
                "EFFECTIVE_FLOP_S, the MLP GEMV-dressed rate "
                f"({EFFECTIVE_FLOP_S / 1e12:.3f} TFLOP/s). ASSUMPTION: same "
                "cost-per-FLOP as the incumbent MAC (mostly wait-on-memory). "
                f"Range [0, {FLOP_EXPOSED_MULT}x] covers hidden-under-bandwidth "
                "to decode-ALU-like."
            ),
            "flop_ms_range": [flop_hidden, flop_exposed],
            "dispatch_class": dclass,
            "dispatch_us_nominal": us_nom,
            "dispatch_us_range": [us_lo, us_hi],
            "dispatch_note": dispatch_assumption,
            "cited_token_ms": CITED_TOKEN_MS,
            "host_gap_floor_ms": CITED_HOST_MS,
        },
        "s020_section_20": {
            "bar_frac": S020_SECTION_20_BAR_FRAC,
            "bar_ms": S020_SECTION_20_BAR_MS,
            "plausible_ms_saved": plausible_ms_saved,
            "clears_time_bar": clears_time_bar,
            "reusable_family": reusable_family,
            "high_information_falsifier": high_information_falsifier,
        },
    }


def score_proposal(row: Mapping[str, Any]) -> dict[str, Any]:
    """Score a recorded or ad-hoc proposal mapping."""
    kwargs: dict[str, Any] = {}
    if "bytes_removed" in row:
        kwargs["bytes_removed"] = row["bytes_removed"]
    if "bytes_added" in row:
        kwargs["bytes_added"] = row["bytes_added"]
    elif "bytes_removed" in row:
        kwargs["bytes_added"] = 0
    for key in (
        "extra_flops_per_output_element",
        "dispatch_delta",
        "consuming_primitive",
        "bandwidth_regime",
        "organ",
        "n_output_elements",
        "dispatch_class",
        "retime_organ",
        "reusable_family",
        "high_information_falsifier",
        "status",
    ):
        if key in row and row[key] is not _UNSET:
            kwargs[key] = row[key]
    kwargs["candidate_id"] = row.get("id")
    return score(**kwargs)


# ---------------------------------------------------------------------------
# Place every candidate already on record onto the curve.
# ---------------------------------------------------------------------------


def _load_future(rel: str) -> dict[str, Any]:
    path = REPO / rel
    if not path.is_file():
        raise EconomicsRefuse(f"required receipt missing: {rel} ({path})")
    doc = load_json(path)
    if not isinstance(doc, dict):
        raise EconomicsRefuse(f"{rel} is not an object")
    return doc


def _proposal_from_receipt_row(
    row: Mapping[str, Any],
    *,
    source: str,
    organ: str,
    default_regime: str,
) -> dict[str, Any]:
    cid = str(row["id"])
    status = str(row.get("status") or "OPEN")
    prim = row.get("physical_primitive")
    if prim is not None:
        prim = str(prim)
    raw_elim = row.get("bytes_eliminated_if_true")
    if raw_elim is None:
        removed = 0
        incomplete = True
    else:
        removed = int(raw_elim)
        incomplete = False
        if removed < 0:
            # Smaller group size grows the auxiliary. Not an attack.
            removed = 0
            incomplete = True

    added: dict[str, int] | int = 0
    remat = row.get("dense_rematerialization")
    if status == "REJECTED_DENSE_REMAT":
        params = ORGAN_PARAMS.get(organ)
        if params:
            added = {"generator": int(params) * F16_BYTES}

    dispatch_delta = float(row.get("dispatch_delta") or 0.0)
    extra_flops = float(row.get("extra_flops_per_output_element") or 0.0)
    retime = bool(row.get("retime_organ") or False)
    regime = row.get("bandwidth_regime") or default_regime
    dclass = row.get("dispatch_class")

    return {
        "id": cid,
        "name": row.get("name"),
        "source": source,
        "status": status,
        "organ": organ,
        "consuming_primitive": prim,
        "bytes_removed": removed,
        "bytes_added": added,
        "byte_model_incomplete": incomplete,
        "extra_flops_per_output_element": extra_flops,
        "dispatch_delta": dispatch_delta,
        "dispatch_class": dclass,
        "bandwidth_regime": regime,
        "retime_organ": retime,
        "reusable_family": cid in REUSABLE_FAMILY_IDS,
        "high_information_falsifier": cid in HIGH_INFORMATION_FALSIFIER_IDS,
        "capability": row.get("capability"),
        "dense_rematerialization": remat,
    }


def recorded_proposals() -> list[dict[str, Any]]:
    """Every candidate already on record, plus the surviving size lever
    and the group-size curve points causal_budget ranked separately.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(p: dict[str, Any]) -> None:
        cid = p["id"]
        if cid in seen:
            raise EconomicsRefuse(f"duplicate recorded candidate id {cid}")
        seen.add(cid)
        rows.append(p)

    aux = _load_future(AUX_REL)
    for cand in aux.get("candidates") or []:
        _add(
            _proposal_from_receipt_row(
                cand,
                source=AUX_REL,
                organ="mlp",
                default_regime="affine_q2_family",
            )
        )
    for g, eliminated in GROUP_SIZE_EXTRA:
        _add(
            {
                "id": f"group_size_{g}",
                "name": f"MLP affine-Q2 group size {g} (byte curve exact; capability UNMEASURED)",
                "source": AUX_REL,
                "status": "OPEN",
                "organ": "mlp",
                "consuming_primitive": "FusedDecodeCompute",
                "bytes_removed": eliminated,
                "bytes_added": 0,
                "byte_model_incomplete": False,
                "extra_flops_per_output_element": 0.0,
                "dispatch_delta": 0.0,
                "dispatch_class": "mlp_gqa_norm_fusion",
                "bandwidth_regime": "affine_q2_family",
                "retime_organ": False,
                "reusable_family": True,
                "high_information_falsifier": False,
                "capability": "UNMEASURED",
                "dense_rematerialization": "DIRECT_CONSUME",
            }
        )

    code = _load_future(CODE_REL)
    for cand in code.get("candidates") or []:
        _add(
            _proposal_from_receipt_row(
                cand,
                source=CODE_REL,
                organ="mlp",
                default_regime="affine_q2_family",
            )
        )

    dn = _load_future(DN_REL)
    for cand in dn.get("candidates") or []:
        p = _proposal_from_receipt_row(
            cand,
            source=DN_REL,
            organ="deltanet",
            default_regime="production_deltanet",
        )
        if p["id"] == "fused_update_consume":
            p["dispatch_delta"] = -48.0
            p["dispatch_class"] = "deltanet_ba"
            p["name"] = cand.get("name") or p.get("name")
        if p["consuming_primitive"] is None:
            p["bandwidth_regime"] = "unknown_new_representation"
        _add(p)

    # The size sweep killed "concatenate to 338 MB and reach 497". What
    # survives is the 5 MB -> 20 MB amortization on THIS kernel family,
    # and the fact that isolated affine-Q2 saturates near 377.
    _add(
        {
            "id": "surviving_dispatch_size_amortize_sub20mb",
            "name": (
                "amortize affine-Q2 launches from ~5 MB (223 GB/s) to >= 20 MB "
                "(326 GB/s); the only size effect the sweep left standing"
            ),
            "source": SWEEP_REL,
            "status": "OPEN",
            "organ": "mlp",
            "consuming_primitive": "FusedDecodeCompute",
            "bytes_removed": 0,
            "bytes_added": 0,
            "byte_model_incomplete": False,
            "extra_flops_per_output_element": 0.0,
            "dispatch_delta": 0.0,
            "dispatch_class": "mlp_gqa_norm_fusion",
            "bandwidth_regime": "affine_q2_unamortized",
            "retime_organ": False,
            "reusable_family": True,
            "high_information_falsifier": True,
            "capability": "MEASURED_ON_AFFINE_Q2",
            "dense_rematerialization": "DIRECT_CONSUME",
            "note": (
                "Production organs already average >= 8.8 MB/dispatch and sit "
                "in the 342-360 cluster, so this is not a save on today's "
                "graph. It is a constraint on new representations: do not "
                "shard into 5 MB launches of this kernel family. MATERIAL "
                "as a reusable family / high-information falsifier, not as "
                "a 1% token-time lever on the current token."
            ),
        }
    )
    _add(
        {
            "id": "dispatch_size_concat_to_lm_head_mb",
            "name": (
                "concatenate affine-Q2 working set to the LM head's 337.7 MB "
                "in hope of 497.4 GB/s"
            ),
            "source": SWEEP_REL,
            "status": "PER_DISPATCH_SIZE_REFUTED",
            "organ": "mlp",
            "consuming_primitive": "FusedDecodeCompute",
            "bytes_removed": 0,
            "bytes_added": 0,
            "byte_model_incomplete": False,
            "extra_flops_per_output_element": 0.0,
            "dispatch_delta": 0.0,
            "dispatch_class": "mlp_gqa_norm_fusion",
            "bandwidth_regime": "affine_q2_saturated",
            "retime_organ": True,
            "reusable_family": False,
            "high_information_falsifier": True,
            "capability": "REFUTED_AS_497_PATH",
            "dense_rematerialization": "DIRECT_CONSUME",
            "note": (
                "The sweep reached 374 GB/s at 338 MB and 377 GB/s at 700 MB, "
                "not 497. Layer-N+1 dependency means production cannot take "
                "the isolated concat. Scored at the measured affine-Q2 "
                "ceiling so the leftover 344->377 is not re-proposed as 497."
            ),
        }
    )
    return rows


def _compact(score_row: dict[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    s20 = score_row["s020_section_20"]
    assumptions = score_row["assumptions"]
    return {
        "id": proposal["id"],
        "name": proposal.get("name"),
        "source": proposal.get("source"),
        "status": proposal.get("status"),
        "live": score_row["live"],
        "organ": score_row["organ"],
        "consuming_primitive": score_row["consuming_primitive"],
        "bytes_removed": score_row["bytes_removed"],
        "bytes_added_total": int(score_row["bytes_added"].get("total", 0)),
        "bytes_added": {
            k: int(score_row["bytes_added"].get(k, 0)) for k in BYTES_ADDED_FIELDS
        },
        "byte_model_incomplete": bool(proposal.get("byte_model_incomplete")),
        "net_bytes": score_row["net_bytes"],
        "extra_flops_per_output_element": score_row["extra_flops_per_output_element"],
        "dispatch_delta": score_row["dispatch_delta"],
        "verdict": score_row["verdict"],
        "verdict_reasons": list(score_row["verdict_reasons"]),
        "predicted_ms_delta": _r(score_row["predicted_ms_delta"], 4),
        "predicted_ms_delta_range": [
            _r(score_row["predicted_ms_delta_range"][0], 4),
            _r(score_row["predicted_ms_delta_range"][1], 4),
        ],
        "predicted_ms_saved": _r(score_row["predicted_ms_saved"], 4),
        "predicted_token_ms": _r(score_row["predicted_token_ms"], 4),
        "predicted_token_ms_range": [
            _r(score_row["predicted_token_ms_range"][0], 4),
            _r(score_row["predicted_token_ms_range"][1], 4),
        ],
        "predicted_tps": _r(score_row["predicted_tps"], 3),
        "predicted_tps_range": [
            _r(score_row["predicted_tps_range"][0], 3),
            _r(score_row["predicted_tps_range"][1], 3),
        ],
        "terms": {k: _r(v, 4) for k, v in score_row["terms"].items()},
        "assumptions": {
            "bandwidth_regime": assumptions["bandwidth_regime"],
            "bandwidth_gb_s_nominal": _r(assumptions["bandwidth_gb_s_nominal"], 2),
            "bandwidth_gb_s_range": [
                _r(assumptions["bandwidth_gb_s_range"][0], 2),
                _r(assumptions["bandwidth_gb_s_range"][1], 2),
            ],
            "bandwidth_is_assumption": assumptions["bandwidth_is_assumption"],
            "bandwidth_note": assumptions["bandwidth_note"],
            "flop_ms_range": [
                _r(assumptions["flop_ms_range"][0], 4),
                _r(assumptions["flop_ms_range"][1], 4),
            ],
            "dispatch_class": assumptions["dispatch_class"],
            "dispatch_us_nominal": _r(assumptions["dispatch_us_nominal"], 3),
            "dispatch_note": assumptions["dispatch_note"],
        },
        "s020_section_20": {
            "bar_ms": _r(s20["bar_ms"], 4),
            "plausible_ms_saved": _r(s20["plausible_ms_saved"], 4),
            "clears_time_bar": s20["clears_time_bar"],
            "reusable_family": s20["reusable_family"],
            "high_information_falsifier": s20["high_information_falsifier"],
        },
        "capability": proposal.get("capability"),
        "note": proposal.get("note"),
    }


def rank_recorded() -> list[dict[str, Any]]:
    proposals = recorded_proposals()
    scored: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for p in proposals:
        s = score_proposal(p)
        scored.append((p, s))
    # Economic curve: largest predicted save first. Stable by id.
    scored.sort(key=lambda pair: (-pair[1]["predicted_ms_saved"], pair[0]["id"]))
    out = []
    for rank, (p, s) in enumerate(scored, start=1):
        row = _compact(s, p)
        row["rank"] = rank
        out.append(row)
    return out


def build() -> dict[str, Any]:
    ranked = rank_recorded()
    live_material = [r["id"] for r in ranked if r["live"] and r["verdict"] == "MATERIAL"]
    live_immaterial = [
        r["id"] for r in ranked if r["live"] and r["verdict"] == "IMMATERIAL"
    ]
    dead = [r["id"] for r in ranked if not r["live"]]
    by_id = {r["id"]: r for r in ranked}

    # Pin the two load-bearing guards into the receipt so a closer that
    # dropped them is visible without rerunning pytest.
    refused = False
    try:
        score(bytes_removed=1)
    except IncompleteEconomics:
        refused = True

    both_terms = score(
        bytes_removed=MLP_ACTIVE_BYTES,
        bytes_added=0,
        extra_flops_per_output_element=100.0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
    )
    bytes_only = score(
        bytes_removed=MLP_ACTIVE_BYTES,
        bytes_added=0,
        extra_flops_per_output_element=0.0,
        organ="mlp",
        consuming_primitive="FusedDecodeCompute",
    )

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "evidence_class": EVIDENCE_CLASS,
        "gpu_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "s020_section_20_bar": {
            "frac_of_complete_token_time": S020_SECTION_20_BAR_FRAC,
            "bar_ms": _r(S020_SECTION_20_BAR_MS, 4),
            "cited_token_ms": CITED_TOKEN_MS,
            "or_reusable_representation_family": True,
            "or_high_information_falsifier": True,
            "verbatim": (
                "a candidate deserves substantial work if it plausibly "
                "removes >= 1% of complete token time, or creates a "
                "reusable representation family, or provides a "
                "high-information falsifier"
            ),
        },
        "measured_constants_cited": {
            "token_ms": CITED_TOKEN_MS,
            "host_ms": CITED_HOST_MS,
            "gpu_ms": CITED_GPU_MS,
            "tps_71_needs_token_ms": _r(TPS_71_TOKEN_MS, 3),
            "tps_71_needs_gpu_ms": _r(TPS_71_GPU_MS, 3),
            "gpu_reduction_for_71_at_current_executor": _r(GPU_REDUCTION_FOR_71, 4),
            "affine_q2_gb_s_at_5mb": AFFINE_Q2_GB_S_AT_5MB,
            "affine_q2_gb_s_at_20mb": AFFINE_Q2_GB_S_AT_20MB,
            "affine_q2_gb_s_at_338mb": AFFINE_Q2_GB_S_AT_338MB,
            "affine_q2_gb_s_at_700mb": AFFINE_Q2_GB_S_AT_700MB,
            "affine_q2_saturated_gb_s": AFFINE_Q2_SATURATED_GB_S,
            "lm_head_gb_s": LM_HEAD_GB_S,
            "clean_gemv_gb_s": CLEAN_GEMV_GB_S,
            "mlp_code_independent_fraction": 0.93509,
            "marginal_dispatch_us_mlp_gqa_norm": MARGINAL_US_MLP_GQA_NORM,
            "marginal_dispatch_us_deltanet_ba": MARGINAL_US_DELTANET_BA,
            "organs": [
                {
                    "organ": o["organ"],
                    "bytes": ORGAN_BYTES[o["organ"]],
                    "ms": o["ms"],
                    "gb_s": o["gb_s"],
                    "dispatches": o["dispatches"],
                }
                for o in cb.ORGANS
            ],
            "mlp_output_elements": ORGAN_OUTPUT_ELEMENTS["mlp"],
            "effective_flop_s_mlp_dressed": EFFECTIVE_FLOP_S,
        },
        "model": {
            "formula": (
                "predicted_token_ms = cited_token_ms "
                "+ ms(net_bytes, rate) "
                "+ extra_flops * n_output_elements / EFFECTIVE_FLOP_S * 1000 "
                "+ dispatch_delta * class_us / 1000; "
                "net_bytes = bytes_added.total - bytes_removed; "
                "rate is the organ's measured GB/s for incremental packing, "
                "or the named regime if retime_organ; "
                "the regime is an ASSUMPTION with a range"
            ),
            "refuses": "bytes_removed without bytes_added",
            "bandwidth_regimes": {
                name: {
                    "gb_s_lo": spec["gb_s_lo"],
                    "gb_s_hi": spec["gb_s_hi"],
                    "gb_s_nominal": spec["gb_s_nominal"],
                    "measured": spec["measured"],
                    "assumption": spec["assumption"],
                }
                for name, spec in BANDWIDTH_REGIMES.items()
            },
            "dispatch_classes": DISPATCH_CLASSES,
            "flop_exposed_multiplier": FLOP_EXPOSED_MULT,
            "bytes_added_fields": list(BYTES_ADDED_FIELDS),
        },
        "guards": {
            "bytes_removed_without_bytes_added_refused": refused,
            "mlp_full_removal_byte_ms": _r(bytes_only["terms"]["byte_ms_delta"], 4),
            "mlp_full_removal_flop_ms_at_100": _r(both_terms["terms"]["flop_ms_delta"], 4),
            "mlp_full_removal_scores_both_terms": (
                both_terms["terms"]["byte_ms_delta"] < 0.0
                and both_terms["terms"]["flop_ms_delta"] > 0.0
                and abs(
                    both_terms["terms"]["byte_ms_delta"]
                    - bytes_only["terms"]["byte_ms_delta"]
                )
                < 1e-9
                and both_terms["predicted_ms_delta"] > bytes_only["predicted_ms_delta"]
            ),
        },
        "n_candidates": len(ranked),
        "n_live_material": len(live_material),
        "n_live_immaterial": len(live_immaterial),
        "n_dead": len(dead),
        "live_material_ranked": live_material,
        "live_immaterial_ranked": live_immaterial,
        "candidates_ranked": ranked,
        "what_the_model_cannot_know": [
            "whether a new representation runs in the affine-Q2 family, the LM-head Q4 regime, or somewhere off both curves — that is an ASSUMPTION with a range",
            "capability: no generate gate lives here. A MATERIAL byte lever can still be generation-incoherent",
            "actual_read_bytes_per_token: the catalog sum is not what the GPU read (no Metal memory counter on this box)",
            "extra decode ALU of a codec that is not the incumbent: the FLOP term is GEMV-dressed, range [0, 4x]",
            "whether remaining bytes keep the organ's measured rate or retiming the whole organ is the right counterfactual — retime_organ is an explicit switch",
            "host work a new generator might add; the host gap is a floor, not a claim that host stays 0.989 ms",
        ],
        "sources": [AUX_REL, CODE_REL, DN_REL, SWEEP_REL, BA_REL],
        "top_live_material": [
            {
                "id": cid,
                "predicted_ms_saved": by_id[cid]["predicted_ms_saved"],
                "predicted_tps": by_id[cid]["predicted_tps"],
                "verdict_reasons": by_id[cid]["verdict_reasons"],
                "status": by_id[cid]["status"],
            }
            for cid in live_material[:12]
        ],
    }


def record() -> Path:
    return write_receipt(RECEIPT, build(), RECORDED_BY)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Input:")[0].strip())
    parser.add_argument("--record", action="store_true", help=f"write receipts/future/{RECEIPT}")
    parser.add_argument("--build", action="store_true", help="alias of --record")
    args = parser.parse_args(list(argv) if argv is not None else None)
    doc = build()
    if args.record or args.build:
        path = record()
        print(f"wrote {path}")
    print(
        f"  {doc['n_candidates']} on the curve; "
        f"{doc['n_live_material']} live MATERIAL; "
        f"{doc['n_live_immaterial']} live IMMATERIAL; "
        f"{doc['n_dead']} dead"
    )
    for row in doc["top_live_material"][:8]:
        print(
            f"  {row['predicted_ms_saved']:+7.3f} ms  "
            f"{row['predicted_tps']:6.2f} pred-TPS  "
            f"{row['status']:16s}  {row['id']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
