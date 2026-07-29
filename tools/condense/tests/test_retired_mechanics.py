#!/usr/bin/env python3.12
"""Retired-controller engine shells for family mechanics (H1/J2). Bodies in _retired_shell."""
from __future__ import annotations
from pathlib import Path
from tools.condense.tests._retired_shell import lifecycle, reseal, spec_seal_run
FAMILY = 'mechanics'

class TestMechMeasure:
    def test_mech_vector_populated_and_labelled(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_mech_vector_populated_and_labelled')
    def test_mech_vector_does_not_hide_arithmetic_in_lookups(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_mech_vector_does_not_hide_arithmetic_in_lookups')
    def test_m1_equals_b1_same_artifact(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_m1_equals_b1_same_artifact')
    def test_b0_b1_m1_all_execute_same_recon(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_b0_b1_m1_all_execute_same_recon')
    def test_m1_batch_matches_b1_batch(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_m1_batch_matches_b1_batch')
    def test_no_dense_shadow_bounded_temporary(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_no_dense_shadow_bounded_temporary')
    def test_m1_reads_less_than_full_dense_bytes(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_m1_reads_less_than_full_dense_bytes')
    def test_deterministic_same_seed(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_deterministic_same_seed')
    def test_quality_parity_matches_dense_definition(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_quality_parity_matches_dense_definition')
    def test_seal_roundtrip_self_sha256(self, tmp_path: Path) -> None: reseal(tmp_path, FAMILY, 'test_seal_roundtrip_self_sha256')
    def test_m1_cpu_metal_parity(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_m1_cpu_metal_parity')
    def test_b1_cpu_metal_parity(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_b1_cpu_metal_parity')
    def test_module_selftest_green(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_module_selftest_green')

class TestMechRunAll:
    def test_module_selftest_green(self, tmp_path: Path) -> None: spec_seal_run(tmp_path, FAMILY, 'test_module_selftest_green')
    def test_staged_execution_matches_frozen_shared_grammar(self, tmp_path: Path) -> None: spec_seal_run(tmp_path, FAMILY, 'test_staged_execution_matches_frozen_shared_grammar')
    def test_M2_runs_and_measured(self, tmp_path: Path) -> None: spec_seal_run(tmp_path, FAMILY, 'test_M2_runs_and_measured')
    def test_M3_M4_M5_M6_run_and_measured(self, tmp_path: Path) -> None: spec_seal_run(tmp_path, FAMILY, 'test_M3_M4_M5_M6_run_and_measured')
    def test_no_dense_shadow_M2_through_M6(self, tmp_path: Path) -> None: spec_seal_run(tmp_path, FAMILY, 'test_no_dense_shadow_M2_through_M6')
    def test_quality_gate_rejects_lower_quality(self, tmp_path: Path) -> None: spec_seal_run(tmp_path, FAMILY, 'test_quality_gate_rejects_lower_quality')
    def test_fake_win_ban_candidate_marked_inadmissible(self, tmp_path: Path) -> None: spec_seal_run(tmp_path, FAMILY, 'test_fake_win_ban_candidate_marked_inadmissible')
    def test_conditional_false_negative_gate_fires(self, tmp_path: Path) -> None: spec_seal_run(tmp_path, FAMILY, 'test_conditional_false_negative_gate_fires')
    def test_pareto_excludes_dominated_and_dense(self, tmp_path: Path) -> None: spec_seal_run(tmp_path, FAMILY, 'test_pareto_excludes_dominated_and_dense')
    def test_pareto_champions_present(self, tmp_path: Path) -> None: spec_seal_run(tmp_path, FAMILY, 'test_pareto_champions_present')
    def test_deterministic_same_seed(self, tmp_path: Path) -> None: spec_seal_run(tmp_path, FAMILY, 'test_deterministic_same_seed')
    def test_cluster_ledger_shared_cheaper_than_independent(self, tmp_path: Path) -> None: spec_seal_run(tmp_path, FAMILY, 'test_cluster_ledger_shared_cheaper_than_independent')
    def test_islands_are_billed_and_increase_bytes(self, tmp_path: Path) -> None: spec_seal_run(tmp_path, FAMILY, 'test_islands_are_billed_and_increase_bytes')
