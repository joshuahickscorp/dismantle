import copy

import pytest

from tools.condense import tg_active_byte_budget as budget
from tools.condense import tg_k11_reconcile as reconcile
from tools.condense import tg_k11_synthetic_schedule as schedule


def test_geometry_and_touch_identities_are_exact():
    geometry = schedule.geometry_constants()
    assert geometry["routed_historical_ideal_78_bytes"] == 2_580_304_896
    assert geometry["routed_historical_ideal_78_bytes"] == budget.ROUTED_WEIGHT_FLOOR_BYTES
    assert geometry["routed_scheduled_75_bytes"] == 2_481_062_400
    assert geometry["shared_scheduled_bytes"] == 310_132_800
    assert geometry["dense_mlp_scheduled_bytes"] == 74_336_832
    assert geometry["attention_scheduled_bytes"] == 1_408_647_552
    assert geometry["indexer_scheduled_bytes"] == 787_218_432
    assert geometry["router_scheduled_bytes"] == 471_859_200
    assert geometry["lm_head_scheduled_bytes"] == 3_806_330_880
    assert geometry["static_total_weight_bytes"] == 9_339_588_096
    assert geometry["total_weight_touches"] == 2_563


def test_schedule_is_deterministic_closed_and_false_fenced():
    first = schedule.emit_token_schedule(0)
    second = schedule.emit_token_schedule(0)
    assert first == second
    assert len(first["touches"]) == 2_563
    assert sum(first["category_bytes_ledger"].values()) == 9_339_588_096
    assert set(first["category_bytes_budget"]) == set(budget.BYTE_CATEGORIES)
    assert all(value is False for value in first["claims"].values())
    assert all(value is False for value in first["fences"].values())


def test_happy_reconciliation_feeds_only_planning_budget():
    result = reconcile.reconcile_schedule(
        schedule.emit_token_schedule(0),
        bandwidth_gbps=800,
        headroom_fraction="0.2",
    )
    assert result["ok"]
    assert result["identities"]["touch_count"] == 2_563
    assert result["identities"]["weight_active_bytes"] == 9_339_588_096
    assert result["identities"]["routed_ideal_78_eq_contract_floor"]
    assert result["identities"]["routed_scheduled_75_disclosed"]
    targets = {row["milestone"]: row for row in result["budget_receipt"]["targets"]}
    assert not targets["TG2"]["admitted_by_bytes"]
    assert not targets["TG1"]["admitted_by_bytes"]
    assert all(value is False for value in result["claims"].values())
    assert all(value is False for value in result["fences"].values())


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda s: s["touches"].append(copy.deepcopy(s["touches"][0])), "count"),
        (lambda s: s["touches"][0].update(bytes=s["touches"][0]["bytes"] + 1), "canonical"),
        (lambda s: s["touches"][0].update(ledger_category="other"), "canonical"),
        (lambda s: s["touches"][0].pop("address_generation"), "closed"),
        (lambda s: s["touches"][0].update(address_generation=0), "canonical"),
        (lambda s: s["geometry"].update(n_layers=77), "geometry"),
        (lambda s: s["category_bytes_ledger"].update(attention=0), "ledger category"),
        (lambda s: s["category_bytes_budget"].update(head=0), "budget category"),
        (lambda s: s["claims"].update(attacker=False), "closed schema"),
    ],
)
def test_schedule_mutations_refuse(mutation, match):
    value = schedule.emit_token_schedule(0)
    mutation(value)
    with pytest.raises(reconcile.ReconcileError, match=match):
        reconcile.reconcile_schedule(value, bandwidth_gbps=800)


def test_kv_and_transfer_are_separate_margin_not_weight_active():
    value = schedule.emit_token_schedule(
        0,
        kv_cache_bytes=123,
        transfer_bytes=456,
    )
    result = reconcile.reconcile_schedule(value, bandwidth_gbps=418)
    assert result["identities"]["weight_active_bytes"] == 9_339_588_096
    assert result["identities"]["kv_cache_bytes"] == 123
    assert result["identities"]["transfer_bytes"] == 456
    assert result["identities"]["all_budget_bytes"] == 9_339_588_675


def test_bandwidth_provenance_is_closed():
    value = schedule.emit_token_schedule(0)
    with pytest.raises(reconcile.ReconcileError, match="unknown provenance"):
        reconcile.reconcile_schedule(
            value,
            bandwidth_gbps=800,
            bandwidth_provenance="device_claim",
        )
    with pytest.raises(reconcile.ReconcileError, match="requires lowercase"):
        reconcile.reconcile_schedule(
            value,
            bandwidth_gbps=418,
            bandwidth_provenance="measured_sustained_fixture",
        )
    result = reconcile.reconcile_schedule(
        value,
        bandwidth_gbps=418,
        bandwidth_provenance="measured_sustained_fixture",
        bandwidth_measurement_identity="a" * 64,
    )
    assert result["bandwidth"]["measurement_identity"] == "a" * 64


def test_adapter_refuses_other_instead_of_absorbing_gap():
    ledger = {key: 0 for key in schedule.LEDGER_CATEGORIES}
    ledger["other"] = 1
    with pytest.raises(ValueError, match="unclassified"):
        schedule.to_budget_categories(ledger, kv_cache_bytes=0, transfer_bytes=0)
