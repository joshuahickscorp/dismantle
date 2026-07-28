#!/usr/bin/env python3
"""Source-body-free active-byte physics for the Temporal Gravity ladder.

This calculator is planning evidence only.  It never reads a model artifact,
measures TPS, or promotes a TG milestone.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Any, Mapping

SCHEMA = "hawking.tg_active_byte_budget.v1"
ROUTED_WEIGHT_FLOOR_BYTES = 2_580_304_896
BYTE_CATEGORIES = (
    "routed_experts",
    "shared_experts",
    "dense",
    "attention",
    "indexer",
    "router",
    "head",
    "kv_cache",
    "transfer",
    "other",
)
MAX_BYTE_COUNT = (1 << 63) - 1
MILESTONE_TARGET_MS = {
    "TG20": Decimal("20"),
    "TG10": Decimal("10"),
    "TG5": Decimal("5"),
    "TG2": Decimal("2"),
    "TG1": Decimal("1"),
}


class BudgetError(ValueError):
    """A typed refusal for malformed or physically meaningless inputs."""


def _decimal(value: Any, field: str, *, allow_zero: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise BudgetError(f"{field}: bool is not numeric authority")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise BudgetError(f"{field}: invalid decimal") from exc
    if not result.is_finite():
        raise BudgetError(f"{field}: nonfinite")
    if result < 0 or (result == 0 and not allow_zero):
        raise BudgetError(f"{field}: must be {'nonnegative' if allow_zero else 'positive'}")
    return result


def _byte_count(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BudgetError(f"{field}: must be an integer byte count")
    if value < 0:
        raise BudgetError(f"{field}: must be nonnegative")
    if value > MAX_BYTE_COUNT:
        raise BudgetError(f"{field}: exceeds signed-u64 interoperability bound")
    return value


def normalize_categories(category_bytes: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(category_bytes, Mapping):
        raise BudgetError("category_bytes: must be an object")
    observed = set(category_bytes)
    expected = set(BYTE_CATEGORIES)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise BudgetError(f"category_bytes: closed schema missing={missing} extra={extra}")
    return {
        category: _byte_count(category_bytes[category], f"category_bytes.{category}")
        for category in BYTE_CATEGORIES
    }


def active_byte_ceiling(
    bandwidth_gbps: Any,
    target_ms: Any,
    *,
    headroom_fraction: Any = 0,
) -> int:
    """Return the conservative integer byte ceiling for one token."""

    bandwidth = _decimal(bandwidth_gbps, "bandwidth_gbps")
    duration = _decimal(target_ms, "target_ms")
    headroom = _decimal(headroom_fraction, "headroom_fraction", allow_zero=True)
    if headroom >= 1:
        raise BudgetError("headroom_fraction: must be below 1")
    raw = bandwidth * Decimal(1_000_000_000) * duration / Decimal(1000)
    raw *= Decimal(1) - headroom
    return int(raw.to_integral_value(rounding=ROUND_FLOOR))


def required_bandwidth_gbps(active_bytes: Any, target_ms: Any) -> Decimal:
    byte_count = _byte_count(active_bytes, "active_bytes")
    duration = _decimal(target_ms, "target_ms")
    return Decimal(byte_count) * Decimal(1000) / duration / Decimal(1_000_000_000)


def _target_receipt(
    *,
    name: str | None,
    target_ms: Decimal,
    bandwidth_gbps: Decimal,
    headroom_fraction: Decimal,
    categories: Mapping[str, int],
) -> dict[str, Any]:
    ceiling = active_byte_ceiling(
        bandwidth_gbps,
        target_ms,
        headroom_fraction=headroom_fraction,
    )
    total = sum(categories.values())
    routed = categories["routed_experts"]
    non_routed = total - routed
    max_routed = max(0, ceiling - non_routed)
    if max_routed == 0:
        collapse_factor: str | None = None
    else:
        collapse_factor = str(
            max(
                Decimal(1),
                Decimal(ROUTED_WEIGHT_FLOOR_BYTES) / Decimal(max_routed),
            ).normalize()
        )
    usable_fraction = Decimal(1) - headroom_fraction
    current_minimum = required_bandwidth_gbps(total, target_ms)
    routed_minimum = required_bandwidth_gbps(ROUTED_WEIGHT_FLOOR_BYTES, target_ms)
    return {
        "milestone": name,
        "planning_admission_only": True,
        "not_sufficient_for_tg": True,
        "target_ms": str(target_ms),
        "ceiling_bytes": ceiling,
        "total_active_bytes": total,
        "slack_bytes": ceiling - total,
        "admitted_by_bytes": total <= ceiling,
        "non_routed_bytes": non_routed,
        "max_routed_bytes_after_non_routed": max_routed,
        "full_routed_geometry_fits": ROUTED_WEIGHT_FLOOR_BYTES <= max_routed,
        "routed_collapse_factor_required": collapse_factor,
        "minimum_physical_bandwidth_gbps_for_current_total": str(current_minimum),
        "minimum_physical_bandwidth_gbps_for_routed_floor_only": str(routed_minimum),
        "required_rated_bandwidth_gbps_at_headroom_for_current_total": str(
            current_minimum / usable_fraction
        ),
        "required_rated_bandwidth_gbps_at_headroom_for_routed_floor_only": str(
            routed_minimum / usable_fraction
        ),
    }


def evaluate_budget(
    *,
    bandwidth_gbps: Any,
    headroom_fraction: Any,
    category_bytes: Mapping[str, Any],
    diagnostic_target_ms: Any | None = None,
) -> dict[str, Any]:
    bandwidth = _decimal(bandwidth_gbps, "bandwidth_gbps")
    headroom = _decimal(headroom_fraction, "headroom_fraction", allow_zero=True)
    if headroom >= 1:
        raise BudgetError("headroom_fraction: must be below 1")
    categories = normalize_categories(category_bytes)
    targets = [
        _target_receipt(
            name=name,
            target_ms=target,
            bandwidth_gbps=bandwidth,
            headroom_fraction=headroom,
            categories=categories,
        )
        for name, target in MILESTONE_TARGET_MS.items()
    ]
    diagnostic = None
    if diagnostic_target_ms is not None:
        target = _decimal(diagnostic_target_ms, "diagnostic_target_ms")
        diagnostic = _target_receipt(
            name=None,
            target_ms=target,
            bandwidth_gbps=bandwidth,
            headroom_fraction=headroom,
            categories=categories,
        )
    return {
        "schema": SCHEMA,
        "mode": "source_body_free_planning_only",
        "planning_admission_only": True,
        "bandwidth_provenance": "caller_declared_planning_input",
        "bandwidth_unit": "decimal_GB_per_s",
        "byte_unit": "byte",
        "bandwidth_gbps": str(bandwidth),
        "headroom_fraction": str(headroom),
        "usable_bandwidth_gbps": str(bandwidth * (Decimal(1) - headroom)),
        "routed_weight_floor_bytes": ROUTED_WEIGHT_FLOOR_BYTES,
        "routed_weight_floor_semantics": (
            "historical_ideal_78_sparse_layer_contract_floor_not_live_schedule"
        ),
        "category_bytes": categories,
        "category_sum_bytes": sum(categories.values()),
        "targets": targets,
        "diagnostic_target": diagnostic,
        "claims": {
            "base_true_tps": False,
            "tg_milestone": False,
            "capable_artifact": False,
            "real_source_access": False,
        },
        "fences": {
            "RAMANUJAN_RESEARCH_AUTHORIZED": False,
            "HIDE_KERNEL_TURN": False,
            "ODYSSEY_LAUNCH_AUTHORIZED": False,
            "full_traversal": False,
            "mop_touched": False,
        },
    }


def _parse_categories(raw: str) -> Mapping[str, Any]:
    def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BudgetError(f"categories JSON: duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise BudgetError(f"categories JSON: nonfinite {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=closed_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise BudgetError(f"categories JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise BudgetError("categories JSON: expected object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandwidth-gbps", required=True)
    parser.add_argument("--headroom-fraction", default="0")
    parser.add_argument("--categories-json", required=True)
    parser.add_argument("--diagnostic-target-ms")
    args = parser.parse_args(argv)
    try:
        receipt = evaluate_budget(
            bandwidth_gbps=args.bandwidth_gbps,
            headroom_fraction=args.headroom_fraction,
            category_bytes=_parse_categories(args.categories_json),
            diagnostic_target_ms=args.diagnostic_target_ms,
        )
    except BudgetError as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
