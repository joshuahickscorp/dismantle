"""Fail-closed reconciliation for the source-body-free K11 schedule."""

from __future__ import annotations


# --- archive path fixup (lane A1): resolve roots as if still in tools/condense/ ---
import sys as _sys_a1
from pathlib import Path as _Path_a1
_A1_HERE = _Path_a1(__file__).resolve().parent
_A1_CONDENSE = _A1_HERE.parent if _A1_HERE.name == "archive" else _A1_HERE
_A1_REPO = _A1_CONDENSE.parents[1]  # repo root (condense -> tools -> repo)
if str(_A1_CONDENSE) not in _sys_a1.path:
    _sys_a1.path.insert(0, str(_A1_CONDENSE))
# --- end archive path fixup ---
from typing import Any, Mapping

from tools.condense import tg_active_byte_budget as budget
from tools.condense import tg_k11_synthetic_schedule as schedule

SCHEMA = "hawking.tg_k11_reconcile.v1"
TOUCH_FIELDS = {
    "touch_id",
    "token_index",
    "layer",
    "kind",
    "tensor_logical_id",
    "projection",
    "bytes",
    "ledger_category",
    "budget_category",
    "hook",
    "cache_generation",
    "address_generation",
    "role",
    "measurement",
}
SCHEDULE_FIELDS = {
    "schema",
    "mode",
    "artifact_binding",
    "physical_dram_claim",
    "geometry",
    "synthetic_dense_layers",
    "synthetic_full_indexer_layers",
    "token_index",
    "cache_generation",
    "address_generation_base",
    "kv_cache_bytes",
    "transfer_bytes",
    "touches",
    "category_bytes_ledger",
    "category_bytes_budget",
    "claims",
    "fences",
}


class ReconcileError(ValueError):
    pass


def _refuse(condition: bool, message: str) -> None:
    if condition:
        raise ReconcileError(message)


def _assert_false_map(value: Any, field: str) -> None:
    _refuse(not isinstance(value, Mapping), f"{field}: object required")
    _refuse(not value, f"{field}: empty")
    _refuse(any(item is not False for item in value.values()), f"{field}: all must be false")


def reconcile_schedule(
    candidate: Mapping[str, Any],
    *,
    bandwidth_gbps: Any,
    headroom_fraction: Any = 0,
    bandwidth_provenance: str = "planning_peak_label",
    bandwidth_measurement_identity: str | None = None,
    diagnostic_target_ms: Any | None = None,
) -> dict[str, Any]:
    _refuse(not isinstance(candidate, Mapping), "schedule: object required")
    _refuse(set(candidate) != SCHEDULE_FIELDS, "schedule: closed schema mismatch")
    _refuse(candidate["schema"] != schedule.SCHEMA, "schedule: wrong schema")
    _refuse(candidate["mode"] != "source_body_free_planning_only", "schedule: wrong mode")
    _refuse(candidate["artifact_binding"] != "none", "schedule: artifact binding forbidden")
    _refuse(candidate["physical_dram_claim"] is not False, "schedule: physical claim forbidden")
    _assert_false_map(candidate["claims"], "schedule.claims")
    _assert_false_map(candidate["fences"], "schedule.fences")
    _refuse(candidate["claims"] != schedule.CLAIMS, "schedule.claims: closed schema mismatch")
    _refuse(candidate["fences"] != schedule.FENCES, "schedule.fences: closed schema mismatch")
    _refuse(
        bandwidth_provenance not in {"planning_peak_label", "measured_sustained_fixture"},
        "bandwidth: unknown provenance",
    )
    if bandwidth_provenance == "measured_sustained_fixture":
        _refuse(
            not isinstance(bandwidth_measurement_identity, str)
            or len(bandwidth_measurement_identity) != 64
            or any(c not in "0123456789abcdef" for c in bandwidth_measurement_identity),
            "bandwidth: measured fixture requires lowercase SHA-256 identity",
        )
    else:
        _refuse(
            bandwidth_measurement_identity is not None,
            "bandwidth: peak label cannot carry measurement identity",
        )

    token_index = candidate["token_index"]
    _refuse(isinstance(token_index, bool) or not isinstance(token_index, int), "token index")
    try:
        canonical = schedule.emit_token_schedule(
            token_index,
            kv_cache_bytes=candidate["kv_cache_bytes"],
            transfer_bytes=candidate["transfer_bytes"],
            cache_generation=candidate["cache_generation"],
            address_generation_base=candidate["address_generation_base"],
        )
    except ValueError as exc:
        raise ReconcileError(f"schedule inputs: {exc}") from exc
    _refuse(candidate["geometry"] != canonical["geometry"], "geometry mismatch")
    _refuse(
        candidate["synthetic_dense_layers"] != canonical["synthetic_dense_layers"],
        "dense layer identity mismatch",
    )
    _refuse(
        candidate["synthetic_full_indexer_layers"]
        != canonical["synthetic_full_indexer_layers"],
        "indexer layer identity mismatch",
    )

    touches = candidate["touches"]
    _refuse(not isinstance(touches, list), "touches: array required")
    _refuse(len(touches) != 2_563, "touches: exact count 2563 required")
    seen_ids: set[str] = set()
    seen_bills: set[tuple[Any, ...]] = set()
    for index, (touch, expected) in enumerate(zip(touches, canonical["touches"], strict=True)):
        _refuse(not isinstance(touch, Mapping), f"touch {index}: object required")
        _refuse(set(touch) != TOUCH_FIELDS, f"touch {index}: closed schema mismatch")
        _refuse(touch != expected, f"touch {index}: canonical schedule mismatch")
        touch_id = touch["touch_id"]
        _refuse(touch_id in seen_ids, f"touch {index}: duplicate touch_id")
        seen_ids.add(touch_id)
        bill = (
            touch["token_index"],
            touch["tensor_logical_id"],
            touch["role"],
            touch["address_generation"],
        )
        _refuse(bill in seen_bills, f"touch {index}: duplicate active bill")
        seen_bills.add(bill)
        _refuse(touch["hook"] != "record_active_bytes_for", f"touch {index}: wrong hook")
        _refuse(touch["role"] != "active_weight", f"touch {index}: wrong role")
        _refuse(
            touch["measurement"] != "synthetic_static_source_extent",
            f"touch {index}: wrong measurement",
        )
        _refuse(touch["cache_generation"] < 0, f"touch {index}: cache generation")
        _refuse(touch["address_generation"] <= 0, f"touch {index}: address generation")

    ledger = schedule.rollup_ledger_categories(touches)
    _refuse(ledger["other"] != 0, "unclassified other bytes")
    _refuse(candidate["category_bytes_ledger"] != ledger, "ledger category mismatch")
    try:
        converted = schedule.to_budget_categories(
            ledger,
            kv_cache_bytes=candidate["kv_cache_bytes"],
            transfer_bytes=candidate["transfer_bytes"],
        )
    except ValueError as exc:
        raise ReconcileError(f"budget category conversion: {exc}") from exc
    _refuse(candidate["category_bytes_budget"] != converted, "budget category mismatch")
    weight_active = sum(ledger.values())
    _refuse(
        weight_active != candidate["geometry"]["static_total_weight_bytes"],
        "weight active total mismatch",
    )
    try:
        receipt = budget.evaluate_budget(
            bandwidth_gbps=bandwidth_gbps,
            headroom_fraction=headroom_fraction,
            category_bytes=converted,
            diagnostic_target_ms=diagnostic_target_ms,
        )
    except budget.BudgetError as exc:
        raise ReconcileError(f"budget input: {exc}") from exc
    _assert_false_map(receipt["claims"], "budget.claims")
    _assert_false_map(receipt["fences"], "budget.fences")
    return {
        "schema": SCHEMA,
        "mode": "source_body_free_planning_only",
        "ok": True,
        "planning_admission_only": True,
        "bandwidth": {
            "value_gbps": str(bandwidth_gbps),
            "unit": "decimal_GB_per_s",
            "provenance": bandwidth_provenance,
            "measurement_identity": bandwidth_measurement_identity,
        },
        "geometry": candidate["geometry"],
        "identities": {
            "touch_count": len(touches),
            "weight_active_bytes": weight_active,
            "kv_cache_bytes": candidate["kv_cache_bytes"],
            "transfer_bytes": candidate["transfer_bytes"],
            "all_budget_bytes": sum(converted.values()),
            "routed_ideal_78_eq_contract_floor": (
                candidate["geometry"]["routed_historical_ideal_78_bytes"]
                == budget.ROUTED_WEIGHT_FLOOR_BYTES
            ),
            "routed_scheduled_75_disclosed": (
                candidate["geometry"]["routed_scheduled_75_bytes"]
                != budget.ROUTED_WEIGHT_FLOOR_BYTES
            ),
            "other_is_zero": ledger["other"] == 0,
            "no_double_touch": True,
            "every_touch_generation_bound": True,
            "transfer_not_in_weight_active": True,
            "kv_cache_not_in_weight_active": True,
        },
        "category_bytes_ledger": ledger,
        "category_bytes_budget": converted,
        "budget_receipt": receipt,
        "claims": dict(schedule.CLAIMS),
        "fences": dict(schedule.FENCES),
    }
