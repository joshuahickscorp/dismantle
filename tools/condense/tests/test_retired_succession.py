#!/usr/bin/env python3.12
"""Retired-controller engine shells for family succession (H1/J2). Bodies in _retired_shell."""
from __future__ import annotations
from pathlib import Path
from tools.condense.tests._retired_shell import checkpoint, fences_run, lifecycle, reseal, spec_repro
FAMILY = 'succession'

class TestForgeIntegration:
    def test_forge_program_materializes_sealed_and_launch_refused(self, tmp_path: Path) -> None: reseal(tmp_path, FAMILY, 'test_forge_program_materializes_sealed_and_launch_refused')
    def test_forge_program_carries_required_section8_fields(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_forge_program_carries_required_section8_fields')
    def test_pre_run_readiness_receipt_is_derived_not_static(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_pre_run_readiness_receipt_is_derived_not_static')
    def test_giant_adapter_contracts_stable_and_composed_from_authority(self, tmp_path: Path) -> None: fences_run(tmp_path, FAMILY, 'test_giant_adapter_contracts_stable_and_composed_from_authority')
    def test_block0_forward_shapes_and_rmsnorm(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_block0_forward_shapes_and_rmsnorm')

class TestSuccCli:
    def test_all_required_commands_are_dispatched(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_all_required_commands_are_dispatched')
    def test_parser_accepts_every_command(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_parser_accepts_every_command')
    def test_unknown_command_rejected(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_unknown_command_rejected')
    def test_no_documented_command_missing_from_parser(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_no_documented_command_missing_from_parser')

class TestSuccCore:
    def test_events_selftest(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_events_selftest')
    def test_state_selftest(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_state_selftest')
    def test_queue_selftest(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_queue_selftest')
    def test_engine_selftest(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_engine_selftest')
    def test_event_chain_and_resume(self, tmp_path: Path) -> None: checkpoint(tmp_path, FAMILY, 'test_event_chain_and_resume')
    def test_event_tamper_detected(self, tmp_path: Path) -> None: reseal(tmp_path, FAMILY, 'test_event_tamper_detected')
    def test_illegal_transition_refused(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_illegal_transition_refused')
    def test_checkpoint_tamper_refused(self, tmp_path: Path) -> None: reseal(tmp_path, FAMILY, 'test_checkpoint_tamper_refused')
    def test_split_brain_resume_refused(self, tmp_path: Path) -> None: checkpoint(tmp_path, FAMILY, 'test_split_brain_resume_refused')
    def test_full_lifecycle_waits_then_imports(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_full_lifecycle_waits_then_imports')
    def test_default_rows_are_honestly_blocked(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_default_rows_are_honestly_blocked')
    def test_queue_row_tamper_detected_on_reload(self, tmp_path: Path) -> None: reseal(tmp_path, FAMILY, 'test_queue_row_tamper_detected_on_reload')
    def test_invalid_status_refused(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_invalid_status_refused')
    def test_unbound_program_fails_validation(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_unbound_program_fails_validation')
    def test_heavy_dispatch_is_gated(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_heavy_dispatch_is_gated')

class TestSuccEmpirical:
    def test_harvest_selftest(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_harvest_selftest')
    def test_retire_selftest(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_retire_selftest')
    def test_classification_never_conflates_deferral_with_collapse(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_classification_never_conflates_deferral_with_collapse')
    def test_eta_observations_are_per_segment_not_global(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_eta_observations_are_per_segment_not_global')
    def test_engine_skips_retired_experiments(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_engine_skips_retired_experiments')
    def test_retirement_receipts_preserve_evidence_and_reopening(self, tmp_path: Path) -> None: spec_repro(tmp_path, FAMILY, 'test_retirement_receipts_preserve_evidence_and_reopening')

class TestSuccGravity:
    def test_policy_selftest(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_policy_selftest')
    def test_receipts_selftest(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_receipts_selftest')
    def test_engine_selftest(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_engine_selftest')
    def test_starts_below_one_bpw(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_starts_below_one_bpw')
    def test_cannot_finalize_above_one_bpw_without_receipt(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_cannot_finalize_above_one_bpw_without_receipt')
    def test_missing_experiment_not_justification(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_missing_experiment_not_justification')
    def test_scheduler_deferral_not_justification(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_scheduler_deferral_not_justification')
    def test_failed_scalar_codec_not_justification(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_failed_scalar_codec_not_justification')
    def test_unsupported_treatment_never_selected(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_unsupported_treatment_never_selected')
    def test_doctor_bytes_counted_in_whole(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_doctor_bytes_counted_in_whole')
    def test_exact_rational_identity(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_exact_rational_identity')
    def test_tries_other_representation_before_ascending(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_tries_other_representation_before_ascending')
    def test_collapse_triggers_ascent(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_collapse_triggers_ascent')
    def test_pass_triggers_descent(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_pass_triggers_descent')
    def test_physical_impossibility_authorizes_ascent(self, tmp_path: Path) -> None: fences_run(tmp_path, FAMILY, 'test_physical_impossibility_authorizes_ascent')
    def test_state_survives_crash_resume(self, tmp_path: Path) -> None: checkpoint(tmp_path, FAMILY, 'test_state_survives_crash_resume')
    def test_duplicate_heavy_launch_refused(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_duplicate_heavy_launch_refused')
    def test_cannot_become_second_heavy_controller(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_cannot_become_second_heavy_controller')
    def test_queue_and_telegram_survive_restart(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_queue_and_telegram_survive_restart')
    def test_no_conflicting_parent_identities(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_no_conflicting_parent_identities')
    def test_escape_receipt_tamper_detected(self, tmp_path: Path) -> None: reseal(tmp_path, FAMILY, 'test_escape_receipt_tamper_detected')
    def test_f0_f2_cannot_masquerade_as_f4(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_f0_f2_cannot_masquerade_as_f4')
    def test_whole_artifact_bpw_authoritative(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_whole_artifact_bpw_authoritative')
    def test_synthetic_event_horizon_is_half(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_synthetic_event_horizon_is_half')
    def test_gravity_bonus_pulls_toward_subbit(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_gravity_bonus_pulls_toward_subbit')
    def test_physically_impossible_candidate_penalized(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_physically_impossible_candidate_penalized')
    def test_real_parent_defers_subbit_no_false_floor(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_real_parent_defers_subbit_no_false_floor')
    def test_deliverables_seal(self, tmp_path: Path) -> None: reseal(tmp_path, FAMILY, 'test_deliverables_seal')
    def test_selector_binding_applies_gravity_priority(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_selector_binding_applies_gravity_priority')
    def test_finalize_extreme_gate_refuses_unjustified(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_finalize_extreme_gate_refuses_unjustified')
    def test_gravity_notifications_compose(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_gravity_notifications_compose')
    def test_gravity_daily_summary_fields(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_gravity_daily_summary_fields')
    def test_arm_frontier_is_launch_gated(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_arm_frontier_is_launch_gated')
    def test_default_off_until_all_gates(self, tmp_path: Path) -> None: lifecycle(tmp_path, FAMILY, 'test_default_off_until_all_gates')
