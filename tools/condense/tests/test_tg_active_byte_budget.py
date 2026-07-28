from decimal import Decimal

import pytest

from tools.condense import tg_active_byte_budget as budget


def categories(**updates: int) -> dict[str, int]:
    values = {name: 0 for name in budget.BYTE_CATEGORIES}
    values.update(updates)
    return values


def target(receipt: dict, name: str) -> dict:
    return next(row for row in receipt["targets"] if row["milestone"] == name)


def test_peak_800_gbps_has_exact_tg2_tg1_absolute_ceilings():
    receipt = budget.evaluate_budget(
        bandwidth_gbps=800,
        headroom_fraction=0,
        category_bytes=categories(),
    )
    assert target(receipt, "TG2")["ceiling_bytes"] == 1_600_000_000
    assert target(receipt, "TG1")["ceiling_bytes"] == 800_000_000
    assert (
        target(receipt, "TG2")[
            "minimum_physical_bandwidth_gbps_for_routed_floor_only"
        ]
        == "1290.152448"
    )
    assert (
        target(receipt, "TG1")[
            "minimum_physical_bandwidth_gbps_for_routed_floor_only"
        ]
        == "2580.304896"
    )
    assert target(receipt, "TG20")["routed_collapse_factor_required"] == "1"


def test_headroom_and_non_routed_bytes_tighten_routed_allowance():
    receipt = budget.evaluate_budget(
        bandwidth_gbps=800,
        headroom_fraction=Decimal("0.2"),
        category_bytes=categories(
            routed_experts=2_580_304_896,
            attention=200_000_000,
            head=200_000_000,
        ),
    )
    tg2 = target(receipt, "TG2")
    tg1 = target(receipt, "TG1")
    assert tg2["ceiling_bytes"] == 1_280_000_000
    assert tg2["max_routed_bytes_after_non_routed"] == 880_000_000
    assert Decimal(tg2["routed_collapse_factor_required"]) > Decimal("2.93")
    assert tg1["ceiling_bytes"] == 640_000_000
    assert tg1["max_routed_bytes_after_non_routed"] == 240_000_000
    assert Decimal(tg1["routed_collapse_factor_required"]) > Decimal("10.75")
    assert not tg2["full_routed_geometry_fits"]
    assert not tg1["admitted_by_bytes"]


@pytest.mark.parametrize("delta,admitted", [(-1, True), (0, True), (1, False)])
def test_tg2_byte_boundary_immediately_below_equal_above(delta: int, admitted: bool):
    receipt = budget.evaluate_budget(
        bandwidth_gbps=800,
        headroom_fraction=0,
        category_bytes=categories(routed_experts=1_600_000_000 + delta),
    )
    assert target(receipt, "TG2")["admitted_by_bytes"] is admitted


def test_measured_sustained_bandwidth_is_stricter_than_peak_label():
    receipt = budget.evaluate_budget(
        bandwidth_gbps=418,
        headroom_fraction="0.2",
        category_bytes=categories(head=200_000_000, attention=200_000_000),
    )
    assert target(receipt, "TG2")["ceiling_bytes"] == 668_800_000
    assert target(receipt, "TG2")["max_routed_bytes_after_non_routed"] == 268_800_000
    assert target(receipt, "TG1")["ceiling_bytes"] == 334_400_000
    assert target(receipt, "TG1")["max_routed_bytes_after_non_routed"] == 0
    assert target(receipt, "TG1")["routed_collapse_factor_required"] is None


def test_diagnostic_target_never_mints_a_milestone():
    receipt = budget.evaluate_budget(
        bandwidth_gbps=800,
        headroom_fraction=0,
        category_bytes=categories(),
        diagnostic_target_ms="0.5",
    )
    assert receipt["diagnostic_target"]["milestone"] is None
    assert receipt["diagnostic_target"]["ceiling_bytes"] == 400_000_000
    assert not receipt["claims"]["tg_milestone"]
    assert not receipt["claims"]["base_true_tps"]


def test_category_schema_is_closed_and_byte_counts_are_exact_ints():
    with pytest.raises(budget.BudgetError, match="missing"):
        budget.normalize_categories({"routed_experts": 1})
    bad = categories()
    bad["attacker"] = 1
    with pytest.raises(budget.BudgetError, match="extra"):
        budget.normalize_categories(bad)
    with pytest.raises(budget.BudgetError, match="integer"):
        budget.normalize_categories(categories(router=True))
    with pytest.raises(budget.BudgetError, match="integer"):
        budget.normalize_categories(categories(router=1.5))
    with pytest.raises(budget.BudgetError, match="nonnegative"):
        budget.normalize_categories(categories(router=-1))
    with pytest.raises(budget.BudgetError, match="interoperability"):
        budget.normalize_categories(categories(router=1 << 63))


def test_categories_json_rejects_duplicate_keys_and_nonfinite_values():
    with pytest.raises(budget.BudgetError, match="duplicate"):
        budget._parse_categories('{"routed_experts":0,"routed_experts":1}')
    with pytest.raises(budget.BudgetError, match="nonfinite"):
        budget._parse_categories('{"routed_experts":NaN}')


@pytest.mark.parametrize(
    "bandwidth,target_ms,headroom",
    [
        (True, 2, 0),
        ("nan", 2, 0),
        (800, "inf", 0),
        (800, 2, 1),
        (0, 2, 0),
    ],
)
def test_invalid_physics_inputs_refuse(bandwidth, target_ms, headroom):
    with pytest.raises(budget.BudgetError):
        budget.active_byte_ceiling(
            bandwidth,
            target_ms,
            headroom_fraction=headroom,
        )


def test_receipt_carries_all_false_fences():
    receipt = budget.evaluate_budget(
        bandwidth_gbps=800,
        headroom_fraction=0,
        category_bytes=categories(),
    )
    assert receipt["schema"] == "hawking.tg_active_byte_budget.v1"
    assert receipt["planning_admission_only"]
    assert receipt["bandwidth_unit"] == "decimal_GB_per_s"
    assert receipt["byte_unit"] == "byte"
    assert all(row["not_sufficient_for_tg"] for row in receipt["targets"])
    assert set(receipt["category_bytes"]) == set(budget.BYTE_CATEGORIES)
    assert receipt["category_sum_bytes"] == 0
    assert all(value is False for value in receipt["claims"].values())
    assert all(value is False for value in receipt["fences"].values())
