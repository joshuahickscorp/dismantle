"""Deterministic source-body-free K11 weight-touch schedule.

The schedule mirrors historical General-R0 header geometry already frozen in
``cost_ledger.rs``.  It is static planning evidence, not physical DRAM or TG
evidence.
"""

from __future__ import annotations

from typing import Any

from tools.condense import tg_active_byte_budget as byte_budget

SCHEMA = "hawking.tg_k11_synthetic_schedule.v1"
N_LAYERS = 78
N_DENSE_MLP_LAYERS = 3
N_SPARSE_LAYERS = 75
N_FULL_INDEXER_LAYERS = 21
EXPERTS_PER_TOKEN = 8
PROJECTIONS_PER_EXPERT = 3

EXPERT_PROJECTION_BYTES = 1_378_368
DENSE_MLP_PROJECTION_BYTES = 8_259_648
ATTENTION_PROJECTIONS = {
    "q_a": 1_378_368,
    "q_b": 3_672_128,
    "kv_a": 389_184,
    "kv_b": 1_607_744,
    "o": 11_012_160,
}
INDEXER_PROJECTIONS = {
    "wq_b": 16_777_216 * 2,
    "wk": 1_572_864 * 2,
    "weights_proj": 393_216 * 2,
}
ROUTER_BYTES = 3_145_728 * 2
LM_HEAD_BYTES = 154_880 * 6_144 * 4

DENSE_LAYERS = frozenset({0, 1, 2})
# Synthetic identities only; count and extents are authoritative for this
# fixture, not the locations of a capable artifact's full indexer layers.
FULL_INDEXER_LAYERS = frozenset(range(0, 63, 3))

LEDGER_TO_BUDGET = {
    "routed_experts": "routed_experts",
    "shared_experts": "shared_experts",
    "dense_mlp": "dense",
    "attention": "attention",
    "indexer": "indexer",
    "router": "router",
    "lm_head": "head",
    "other": "other",
}
LEDGER_CATEGORIES = tuple(LEDGER_TO_BUDGET)
PROJECTION_ORDER = ("gate", "up", "down")

CLAIMS = {
    "base_true_tps": False,
    "tg_milestone": False,
    "capable_artifact": False,
    "real_source_access": False,
    "physical_dram": False,
}
FENCES = {
    "RAMANUJAN_RESEARCH_AUTHORIZED": False,
    "HIDE_KERNEL_TURN": False,
    "ODYSSEY_LAUNCH_AUTHORIZED": False,
    "full_traversal": False,
    "mop_touched": False,
}


def geometry_constants() -> dict[str, int]:
    routed_ideal = EXPERTS_PER_TOKEN * 3 * EXPERT_PROJECTION_BYTES * N_LAYERS
    routed_scheduled = (
        EXPERTS_PER_TOKEN * 3 * EXPERT_PROJECTION_BYTES * N_SPARSE_LAYERS
    )
    shared = 3 * EXPERT_PROJECTION_BYTES * N_SPARSE_LAYERS
    dense = 3 * DENSE_MLP_PROJECTION_BYTES * N_DENSE_MLP_LAYERS
    attention = sum(ATTENTION_PROJECTIONS.values()) * N_LAYERS
    indexer = sum(INDEXER_PROJECTIONS.values()) * N_FULL_INDEXER_LAYERS
    router = ROUTER_BYTES * N_SPARSE_LAYERS
    total = routed_scheduled + shared + dense + attention + indexer + router + LM_HEAD_BYTES
    return {
        "n_layers": N_LAYERS,
        "n_sparse_layers": N_SPARSE_LAYERS,
        "n_dense_mlp_layers": N_DENSE_MLP_LAYERS,
        "n_full_indexer_layers": N_FULL_INDEXER_LAYERS,
        "experts_per_token": EXPERTS_PER_TOKEN,
        "projections_per_expert": PROJECTIONS_PER_EXPERT,
        "routed_historical_ideal_78_bytes": routed_ideal,
        "routed_scheduled_75_bytes": routed_scheduled,
        "shared_scheduled_bytes": shared,
        "dense_mlp_scheduled_bytes": dense,
        "attention_scheduled_bytes": attention,
        "indexer_scheduled_bytes": indexer,
        "router_scheduled_bytes": router,
        "lm_head_scheduled_bytes": LM_HEAD_BYTES,
        "static_total_weight_bytes": total,
        "total_weight_touches": 2_563,
    }


def _touch(
    *,
    token_index: int,
    touch_id: str,
    layer: int | None,
    kind: str,
    logical_id: str,
    projection: str,
    byte_count: int,
    ledger_category: str,
    address_generation: int,
    cache_generation: int,
) -> dict[str, Any]:
    return {
        "touch_id": touch_id,
        "token_index": token_index,
        "layer": layer,
        "kind": kind,
        "tensor_logical_id": logical_id,
        "projection": projection,
        "bytes": byte_count,
        "ledger_category": ledger_category,
        "budget_category": LEDGER_TO_BUDGET[ledger_category],
        "hook": "record_active_bytes_for",
        "cache_generation": cache_generation,
        "address_generation": address_generation,
        "role": "active_weight",
        "measurement": "synthetic_static_source_extent",
    }


def emit_token_schedule(
    token_index: int = 0,
    *,
    kv_cache_bytes: int = 0,
    transfer_bytes: int = 0,
    cache_generation: int = 0,
    address_generation_base: int = 1,
) -> dict[str, Any]:
    for field, value in (
        ("token_index", token_index),
        ("kv_cache_bytes", kv_cache_bytes),
        ("transfer_bytes", transfer_bytes),
        ("cache_generation", cache_generation),
        ("address_generation_base", address_generation_base),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field}: integer required")
    if min(token_index, kv_cache_bytes, transfer_bytes, cache_generation) < 0:
        raise ValueError("token/cache/transfer/generation values must be nonnegative")
    if address_generation_base <= 0:
        raise ValueError("address_generation_base must be positive")

    touches: list[dict[str, Any]] = []
    generation = address_generation_base

    def add(**kwargs: Any) -> None:
        nonlocal generation
        touches.append(
            _touch(
                token_index=token_index,
                address_generation=generation,
                cache_generation=cache_generation,
                **kwargs,
            )
        )
        generation += 1

    for layer in range(N_LAYERS):
        for projection, byte_count in ATTENTION_PROJECTIONS.items():
            add(
                touch_id=f"L{layer:03d}.attn.{projection}",
                layer=layer,
                kind="attention_projection",
                logical_id=f"L{layer}.attention.{projection}",
                projection=projection,
                byte_count=byte_count,
                ledger_category="attention",
            )
        if layer in FULL_INDEXER_LAYERS:
            for projection, byte_count in INDEXER_PROJECTIONS.items():
                add(
                    touch_id=f"L{layer:03d}.indexer.{projection}",
                    layer=layer,
                    kind="indexer_projection",
                    logical_id=f"L{layer}.indexer.{projection}",
                    projection=projection,
                    byte_count=byte_count,
                    ledger_category="indexer",
                )
        if layer in DENSE_LAYERS:
            for projection in PROJECTION_ORDER:
                add(
                    touch_id=f"L{layer:03d}.dense.{projection}",
                    layer=layer,
                    kind="dense_mlp_projection",
                    logical_id=f"L{layer}.dense.{projection}",
                    projection=projection,
                    byte_count=DENSE_MLP_PROJECTION_BYTES,
                    ledger_category="dense_mlp",
                )
        else:
            add(
                touch_id=f"L{layer:03d}.router",
                layer=layer,
                kind="router_projection",
                logical_id=f"L{layer}.router",
                projection="router",
                byte_count=ROUTER_BYTES,
                ledger_category="router",
            )
            for expert_slot in range(EXPERTS_PER_TOKEN):
                for projection in PROJECTION_ORDER:
                    add(
                        touch_id=f"L{layer:03d}.E{expert_slot:02d}.{projection}",
                        layer=layer,
                        kind="routed_expert_projection",
                        logical_id=f"L{layer}.E{expert_slot}.{projection}",
                        projection=projection,
                        byte_count=EXPERT_PROJECTION_BYTES,
                        ledger_category="routed_experts",
                    )
            for projection in PROJECTION_ORDER:
                add(
                    touch_id=f"L{layer:03d}.shared.{projection}",
                    layer=layer,
                    kind="shared_expert_projection",
                    logical_id=f"L{layer}.shared.{projection}",
                    projection=projection,
                    byte_count=EXPERT_PROJECTION_BYTES,
                    ledger_category="shared_experts",
                )
    add(
        touch_id="token.lm_head",
        layer=None,
        kind="lm_head_projection",
        logical_id="lm_head",
        projection="lm_head",
        byte_count=LM_HEAD_BYTES,
        ledger_category="lm_head",
    )

    ledger_categories = rollup_ledger_categories(touches)
    budget_categories = to_budget_categories(
        ledger_categories,
        kv_cache_bytes=kv_cache_bytes,
        transfer_bytes=transfer_bytes,
    )
    return {
        "schema": SCHEMA,
        "mode": "source_body_free_planning_only",
        "artifact_binding": "none",
        "physical_dram_claim": False,
        "geometry": geometry_constants(),
        "synthetic_dense_layers": sorted(DENSE_LAYERS),
        "synthetic_full_indexer_layers": sorted(FULL_INDEXER_LAYERS),
        "token_index": token_index,
        "cache_generation": cache_generation,
        "address_generation_base": address_generation_base,
        "kv_cache_bytes": kv_cache_bytes,
        "transfer_bytes": transfer_bytes,
        "touches": touches,
        "category_bytes_ledger": ledger_categories,
        "category_bytes_budget": budget_categories,
        "claims": dict(CLAIMS),
        "fences": dict(FENCES),
    }


def rollup_ledger_categories(touches: list[dict[str, Any]]) -> dict[str, int]:
    totals = {category: 0 for category in LEDGER_CATEGORIES}
    for touch in touches:
        category = touch.get("ledger_category")
        if category not in totals:
            raise ValueError(f"unknown ledger category {category!r}")
        byte_count = touch.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError("touch bytes must be a nonnegative integer")
        totals[category] += byte_count
    return totals


def to_budget_categories(
    ledger_categories: dict[str, int],
    *,
    kv_cache_bytes: int,
    transfer_bytes: int,
) -> dict[str, int]:
    if set(ledger_categories) != set(LEDGER_CATEGORIES):
        raise ValueError("ledger category schema mismatch")
    if ledger_categories["other"] != 0:
        raise ValueError("unclassified active bytes refuse")
    result = {category: 0 for category in byte_budget.BYTE_CATEGORIES}
    for ledger, target in LEDGER_TO_BUDGET.items():
        result[target] = ledger_categories[ledger]
    result["kv_cache"] = kv_cache_bytes
    result["transfer"] = transfer_bytes
    return byte_budget.normalize_categories(result)
