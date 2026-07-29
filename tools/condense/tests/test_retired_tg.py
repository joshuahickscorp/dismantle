#!/usr/bin/env python3.12
"""Retired-controller engine shells for family tg (H1/J2). Bodies in _retired_shell."""
from __future__ import annotations
from pathlib import Path
from tools.condense.tests._retired_shell import fences_run, lifecycle
FAMILY = 'tg'

def test_peak_800_gbps_has_exact_tg2_tg1_absolute_ceilings(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_peak_800_gbps_has_exact_tg2_tg1_absolute_ceilings')
def test_headroom_and_non_routed_bytes_tighten_routed_allowance(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_headroom_and_non_routed_bytes_tighten_routed_allowance')
def test_tg2_byte_boundary_immediately_below_equal_above(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_tg2_byte_boundary_immediately_below_equal_above')
def test_measured_sustained_bandwidth_is_stricter_than_peak_label(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_measured_sustained_bandwidth_is_stricter_than_peak_label')
def test_diagnostic_target_never_mints_a_milestone(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_diagnostic_target_never_mints_a_milestone')
def test_category_schema_is_closed_and_byte_counts_are_exact_ints(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_category_schema_is_closed_and_byte_counts_are_exact_ints')
def test_categories_json_rejects_duplicate_keys_and_nonfinite_values(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_categories_json_rejects_duplicate_keys_and_nonfinite_values')
def test_invalid_physics_inputs_refuse(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_invalid_physics_inputs_refuse')
def test_receipt_carries_all_false_fences(tmp_path: Path) -> None: fences_run(tmp_path, FAMILY, 'test_receipt_carries_all_false_fences')
def test_geometry_and_touch_identities_are_exact(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_geometry_and_touch_identities_are_exact')
def test_schedule_is_deterministic_closed_and_false_fenced(tmp_path: Path) -> None: fences_run(tmp_path, FAMILY, 'test_schedule_is_deterministic_closed_and_false_fenced')
def test_happy_reconciliation_feeds_only_planning_budget(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_happy_reconciliation_feeds_only_planning_budget')
def test_schedule_mutations_refuse(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_schedule_mutations_refuse')
def test_kv_and_transfer_are_separate_margin_not_weight_active(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_kv_and_transfer_are_separate_margin_not_weight_active')
def test_bandwidth_provenance_is_closed(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_bandwidth_provenance_is_closed')
def test_adapter_refuses_other_instead_of_absorbing_gap(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_adapter_refuses_other_instead_of_absorbing_gap')
def test_public_reconciler_returns_one_typed_refusal_surface(tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_public_reconciler_returns_one_typed_refusal_surface')
