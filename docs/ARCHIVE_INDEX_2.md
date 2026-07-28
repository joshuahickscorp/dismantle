# Archive index — floor prune (2026-07-28)

Working-tree floor prune of no-reader root receipts and later closed-campaign
artifacts. **Nothing is lost:** full content lives in git history at the annotated
tag `pre-floor-prune-20260728`. Restore any file with:

```
git checkout pre-floor-prune-20260728 -- <path>
# or
git show pre-floor-prune-20260728:<path>
```

## T1 — root receipts with no executable-code readers — 196 files

Pruned **196 tracked root files** (~58,934,431 bytes,
~2,068,345 lines). Reader map was built by scanning every `.py`/`.rs`/`.sh`/`.toml`/`.ts`/`.tsx`
source under the repo (excluding `.git`, `target`, `models`, `node_modules`) for the
exact basename. Files with any hit stayed. Also kept: build/config, the hard path-loaded
receipts (`tools/odyssey/known_failures.py`, `crates/hawking-adapters/src/evidence.rs`,
campaign/prometheus/condense path constants), and the 14 `must_not_delete` themes from
`HAWKING_CONSOLIDATION_INVENTORY.json` (acceleration verdicts, KIMI scientific law /
long-run / gravity finals, GLM52 functional + Math-Preserve + corrected law, ascension/
odyssey chain, DEEPSEEK contraction pilot, doctor negative atlas, HIDE/Fabric scaffolds, etc.).

## T1 / HAWKING — 100 files

- `HAWKING_ASCENSION_FINAL.json` — FUNCTIONAL_PARTIAL_ONLY on DeepSeek, same terminal category as GLM by a different path
- `HAWKING_ASCENSION_FINAL.md` — Hawking Ascension — final position
- `HAWKING_ASCENSION_FUNCTIONAL_GRAVITY_CONTINUATION.md` — HAWKING ASCENSION: FUNCTIONAL GRAVITY CONTINUATION
- `HAWKING_ASCENSION_STATUS.json` — JSON receipt
- `HAWKING_ASCENSION_STATUS.md` — Hawking Ascension status
- `HAWKING_ATTENTION_PRELUDE_ICB_REPLAY_20260727.json` — ATTENTION_PRELUDE_ICB_REPLAY_INTEGRATED_DEFAULT_OFF
- `HAWKING_ATTENTION_PRE_SCORE_ICB_GROUPING_20260727.json` — ATTENTION_PRE_SCORE_ICB_GROUPING_INTEGRATED_DEFAULT_OFF
- `HAWKING_BASE_RUNTIME_G2_RESULT.json` — hawking.gravity.base_runtime_g2.v1
- `HAWKING_BASE_RUNTIME_ROOT_CAUSE.json` — hawking.gravity.base_runtime_root_cause.v1
- `HAWKING_BASE_RUNTIME_WARM_RESULT.json` — hawking.gravity.base_runtime_warm.v1
- `HAWKING_BF16_NATIVE_RECEIPT.json` — lm_head.weight
- `HAWKING_BRIDGE_ADAPTERS_FINISH.json` — FINISHED
- `HAWKING_BYTE_CATEGORY_LEDGER.json` — JSON receipt
- `HAWKING_BYTE_SURPLUS_ROOT_CAUSE.json` — hawking.gravity.byte_surplus_root_cause.v1
- `HAWKING_CANONICAL_AUTHORITY_MAP.json` — MULTIPLE_LIVE
- `HAWKING_COMMAND_GRAPH_AUDIT_20260726.json` — hawking.gravity.command_graph_audit.v1
- `HAWKING_COMPACT_ATTENTION_ICB_REPLAY_20260727.json` — COMPACT_ATTENTION_ICB_REPLAY_INTEGRATED_DEFAULT_OFF
- `HAWKING_COMPACT_ATTENTION_REPLAY_GRID_AUDIT_20260727.json` — COMPACT_ATTENTION_REPLAY_BOUNDARY_ENUMERATED_NO_OVERDISPATCH_ADMITTED
- `HAWKING_COMPACT_MLA_ADMISSION_PREFLIGHT_20260727.json` — QUALIFIED_DEFAULT_OFF_PREALLOCATION_ADMISSION
- `HAWKING_COMPACT_MLA_APPEND_CANDIDATE_20260727.json` — PREREQUISITE_ONLY_NOT_PROMOTED
- `HAWKING_COMPACT_MLA_CACHE_OWNER_20260727.json` — PRODUCTION_UNREACHABLE_OWNER_NOT_PROMOTED
- `HAWKING_COMPACT_MLA_FIVE_DISPATCH_DAG_20260727.json` — SYNTHETIC_COMPLETE_ATTENTION_DAG_NOT_PROMOTED
- `HAWKING_COMPACT_MLA_LIVE_CANDIDATE_20260727.json` — PARITY_QUALIFIED_DEFAULT_OFF_NOT_PERFORMANCE_PROMOTED
- `HAWKING_COMPACT_MLA_PARITY_PROOF_20260727.json` — hawking.gravity.compact_mla.cpu_architecture_proof.v1
- `HAWKING_COMPACT_MLA_PRODUCTION_FEASIBILITY_AUDIT_20260727.json` — hawking.gravity.compact_mla.production_feasibility_audit.v1
- `HAWKING_COMPACT_MLA_THREE_DISPATCH_CHAIN_20260727.json` — SYNTHETIC_CHAIN_PROOF_NOT_PROMOTED
- `HAWKING_COMPACT_RANKED_ATTENTION_CANDIDATE_20260727.json` — PREREQUISITE_ONLY_NOT_PROMOTED
- `HAWKING_CONSOLIDATION_INVENTORY.md` — Hawking consolidation inventory
- `HAWKING_DEVICE_ATTENTION_PRELUDE_CLOSED_20260727.json` — COMPLETE_TOKEN_PARITY_QUALIFIED_DEFAULT_OFF_NOT_FLAGSHIP_PERFORMANCE_PROMOTED
- `HAWKING_DEVICE_ATTENTION_RESIDUAL_CLOSED_20260727.json` — PARITY_QUALIFIED_DEFAULT_OFF_DEVICE_GRAPH
- `HAWKING_DEVICE_DSA_CLOSED_INDEXER_GRAPH_20260727.json` — COMPLETE_TOKEN_PARITY_QUALIFIED_DEFAULT_OFF_NOT_FLAGSHIP_PERFORMANCE_PROMOTED
- `HAWKING_DEVICE_DSA_LIVE_CANDIDATE_20260727.json` — PARITY_QUALIFIED_GRAPH_CLOSED_DEFAULT_OFF_NOT_PERFORMANCE_PROMOTED
- `HAWKING_DEVICE_DSA_POST_SCORE_ICB_REPLAY_20260727.json` — DEVICE_DSA_POST_SCORE_ICB_REPLAY_INTEGRATED_DEFAULT_OFF
- `HAWKING_DEVICE_DSA_PRE_SCORE_ICB_REPLAY_20260727.json` — DEVICE_DSA_PRE_SCORE_ICB_REPLAY_INTEGRATED_DEFAULT_OFF
- `HAWKING_DEVICE_DSA_RADIX_TOPK_20260727.json` — BOUNDED_32K_EXACT_AND_MEASURED_DEFAULT_OFF_NOT_FLAGSHIP_PROMOTED
- `HAWKING_DEVICE_EXPERT_ADDRESSING_AUDIT_20260727.json` — READ_ONLY_AUDIT_CACHE_INDEXED_HIT_PATH_RECOMMENDED
- `HAWKING_DEVICE_EXPERT_EXECUTION_PERMUTATION_20260727.json` — BOUNDED_PREREQUISITE_QUALIFIED_NO_PERFORMANCE_CLAIM
- `HAWKING_DEVICE_EXPERT_RESIDUAL_CLOSED_20260727.json` — BOUNDED_COMPLETE_TOKEN_QUALIFIED_DEFAULT_OFF
- `HAWKING_DEVICE_EXPERT_TABLE_COMPLETE_WAVE_20260727.json` — BOUNDED_COMPLETE_WAVE_PROOF_QUALIFIED_DEFAULT_UNUSED
- `HAWKING_DEVICE_EXPERT_TABLE_HETEROGENEOUS_20260727.json` — HETEROGENEOUS_SELECTED_ROUTE_INTEGRATED_DEFAULT_OFF
- `HAWKING_DEVICE_EXPERT_TABLE_HIT_PROOF_20260727.json` — BOUNDED_INDIRECT_PROJECTION_PROOF_QUALIFIED_DEFAULT_UNUSED
- `HAWKING_DEVICE_EXPERT_TABLE_ICB_REPLAY_20260727.json` — DEVICE_EXPERT_WAVE_ICB_REPLAY_INTEGRATED_DEFAULT_OFF
- `HAWKING_DEVICE_EXPERT_TABLE_NATIVE_BF16_20260727.json` — NATIVE_BF16_INDIRECT_PROJECTION_QUALIFIED_LOW_LEVEL_ONLY
- `HAWKING_DEVICE_EXPERT_TABLE_PACKED_R0_20260727.json` — PACKED_R0_INDIRECT_PROJECTION_QUALIFIED_LOW_LEVEL_ONLY
- `HAWKING_DEVICE_EXPERT_TABLE_PERSISTENT_LEASE_20260727.json` — PERSISTENT_SELECTED_ROUTE_LEASE_QUALIFIED_DEFAULT_OFF
- `HAWKING_DEVICE_EXPERT_TABLE_PRODUCTION_WIRING_20260727.json` — PRODUCTION_WIRED_QUALIFIED_DEFAULT_OFF
- `HAWKING_DEVICE_FINAL_HEAD_ICB_REPLAY_20260727.json` — DEVICE_FINAL_HEAD_ICB_REPLAY_INTEGRATED_DEFAULT_OFF
- `HAWKING_DEVICE_FINAL_NORM_HEAD_GRAPH_20260727.json` — BOUNDED_PARITY_QUALIFIED_DEFAULT_OFF_WITH_INHERITED_NATIVE_HEAD_BLOCKER
- `HAWKING_DEVICE_ROUTER_SELECTION_20260727.json` — PARITY_QUALIFIED_DEFAULT_OFF_DEVICE_ROUTER
- `HAWKING_DSA_INDEX_OWNERSHIP_SPLIT_20260727.json` — OWNERSHIP_PREREQUISITE_NOT_PROMOTED
- `HAWKING_EXPERT_WAVE_CONCURRENT_RUNTIME_20260727.json` — SYNTHETIC_REPLICATED_AND_BOUNDED_COMPLETE_TOKEN_QUALIFIED_DEFAULT_OFF
- `HAWKING_EXPERT_WAVE_PERSISTENT_SCRATCH_CANDIDATE_20260727.json` — hawking.gravity.expert_wave_persistent_scratch_candidate.v1
- `HAWKING_FINAL_ASCENT_PROCESS_QUIESCENCE_RECEIPT.json` — hawking.final_ascent.process_quiescence_receipt.v1
- `HAWKING_G2_DOMINANT_COST.json` — hawking.gravity.g2_dominant_decode_cost.v1
- `HAWKING_GPU_CACHE_REVIEW.json` — VERIFIED BY CLAUDE. Tests re-run independently. Not merged; end-to-end residency still unmeasured.
- `HAWKING_GRAVITY_CROSS_MODEL_TRANSFER.md` — Cross-model transfer: the next-parent protocol is now amplification-first
- `HAWKING_GRAVITY_RUNTIME_GAPS.json` — hawking.gravity.base_runtime_gaps.v1
- `HAWKING_GRAVITY_SERVE_FIRST_GENERATION.json` — hawking.serve.gravity.first_generation.v1
- `HAWKING_GRAVITY_SERVE_PROMPT_PATH_AUDIT_20260727.json` — PROMPT_DROP_HYPOTHESIS_EXCLUDED
- `HAWKING_HIDE_HANDOFF_CONTRACT.json` — FROZEN_AT_THE_BOUNDARY, implementation deferred until a parent admits a servable compact representation
- `HAWKING_HIDE_RUNTIME_CONTRACT.json` — JSON receipt
- `HAWKING_ICB_COMPLETE_TOKEN_PERFORMANCE_GATE_20260727.json` — ICB_COMPLETE_TOKEN_DEFAULT_ON_REJECTED
- `HAWKING_LADDER_V3.md` — Hawking Ladder V3
- `HAWKING_LEDGER_BF16.json` — JSON receipt
- `HAWKING_LM_HEAD_ROOT_CAUSE.json` — hawking.gravity.lm_head_root_cause.v1
- `HAWKING_LONG_CONTEXT_SCRATCH_RECEIPT_20260727.json` — hawking.gravity.long_context_scratch.v1
- `HAWKING_MEMO_VERIFIED_RESULT.json` — JSON receipt
- `HAWKING_MODEL_FEEL_PARITY_RESULTS.json` — hawking.model_feel_parity.v1
- `HAWKING_MODEL_VAULT.json` — CONTRACT_SEALED_NOT_EXECUTED
- `HAWKING_MOTHERLOAD_REPORT.md` — Hawking Motherload Completion — Session Report
- `HAWKING_NEXT_PARENT_ADMISSION.json` — hawking.next_parent_admission.v1
- `HAWKING_NULL_CORRECTED_METRIC_CONTRACT.md` — Hawking null-corrected metric contract
- `HAWKING_ORCHESTRATION_POLICY.md` — Orchestration policy — standing delegation zones
- `HAWKING_PARALLEL_GATE_ASSESSMENT.json` — substrate sealed
- `HAWKING_PHYSICAL_ACCOUNTING_PHASE1_20260727.json` — hawking.gravity.physical_accounting.phase1.v1
- `HAWKING_POSITIONED_ROPE_REPLAY_ABI_20260727.json` — n_heads
- `HAWKING_POST_ATTENTION_FUSION_REJECTED_20260727.json` — REJECTED_NUMERIC_PARITY_V2_1_NOT_PROMOTED
- `HAWKING_PQ_K_TRANSPOSE_CANDIDATE_20260727.json` — PREREQUISITE_ONLY_NOT_PROMOTED
- `HAWKING_PQ_V_ROWS_CANDIDATE_20260727.json` — PREREQUISITE_ONLY_NOT_PROMOTED
- `HAWKING_PROFILER_RECOVERY_RECEIPT.json` — hawking.gravity.profiler_recovery.v1
- `HAWKING_PROVIDER_CAPABILITY_MATRIX.json` — hawking.provider_capability_matrix.v1
- `HAWKING_RECLAMATION_SURVEY.md` — I'll survey the Python/Rust split read-only and write `RECLAMATION_SURVEY.md` + `.json`. Starting with layout and the gr
- `HAWKING_RECLAMATION_VERDICT.json` — REVISED DOWN -- optimistic
- `HAWKING_REPLAYABLE_COMPUTE_GRAPH_ICB_20260727.json` — REPLAYABLE_COMPUTE_GRAPH_SUBSTRATE_DEFAULT_OFF
- `HAWKING_REPRESENTATION_READINESS_MATRIX.json` — native functional organ replacement
- `HAWKING_RESIDENT_ATTENTION_LAYOUT_VARIANT_20260727.json` — DEFAULT_INERT_LAYOUT_PREREQUISITE_NOT_PROMOTED
- `HAWKING_RESIDENT_KV_STATE_STATIC_20260727.json` — hawking.gravity.resident_kv_state.static_projection.v1
- `HAWKING_RESIDENT_TPS.json` — JSON receipt
- `HAWKING_RESOURCE_MODE.json` — hawking.resource_mode.v1
- `HAWKING_RESUME_NEXT_SESSION.sh` — #!/usr/bin/env bash
- `HAWKING_ROUTE_SEGMENT_PRIMITIVES_20260726.json` — hawking.route_segment_primitives.v1
- `HAWKING_ROUTE_SEGMENT_REORDER_20260726.json` — hawking.route_segment_reorder.v1
- `HAWKING_RUNTIME_ASCENSION_AUDIT.json` — PROVEN
- `HAWKING_SHARED_TREE_PROTOCOL.json` — hawking.shared_tree_protocol.v1
- `HAWKING_STORAGE_CLEANUP_RECEIPT.json` — hawking.storage_cleanup_receipt.v1
- `HAWKING_TG_COST_LEDGER.json` — JSON receipt
- `HAWKING_TG_LEDGER_FINDINGS.json` — hawking.gravity.tg_ledger_findings.v1
- `HAWKING_TG_LEDGER_POSTMEMO.json` — JSON receipt
- `HAWKING_TG_LEDGER_POSTMEMO_FINDINGS.json` — hawking.gravity.tg_ledger_postmemo.v1
- `HAWKING_WARM_NOHASH_RESULT.json` — JSON receipt

## T1 / KIMI — 27 files

- `KIMI_K26_CAUSAL_ATLAS.json` — JSON receipt
- `KIMI_K26_DEVICE_CLEANSE_LEDGER.jsonl` — JSONL ledger
- `KIMI_K26_DOCTOR_BYTE_AUCTION.json` — JSON receipt
- `KIMI_K26_FINAL_GC_LEDGER.jsonl` — JSONL ledger
- `KIMI_K26_FINAL_STORAGE_REPORT.md` — Kimi K2.6 Final Storage Report
- `KIMI_K26_FIRST_CHECKPOINT.json` — PASS
- `KIMI_K26_HANDOFF_PRECHECK.json` — hawking.kimi_k26.handoff_precheck.v1
- `KIMI_K26_HANDOFF_PRECHECK.md` — Kimi K2.6 handoff precheck
- `KIMI_K26_LOCAL_SOURCE_DETECTION.json` — GREEN
- `KIMI_K26_NEXT_EXPERIMENT.json` — ADVANCING
- `KIMI_K26_P1_F1_FUNCTIONAL_CODEBOOKS.json` — JSON receipt
- `KIMI_K26_P1_F1_GRAMMAR_ISLANDS_BRACKET.json` — JSON receipt
- `KIMI_K26_P1_F2_SPARSE_PROPAGATION.json` — JSON receipt
- `KIMI_K26_PARALLEL_EXECUTION_LEDGER.jsonl` — JSONL ledger
- `KIMI_K26_PARALLEL_EXECUTION_PLAN.json` — contextual_native_seam
- `KIMI_K26_PARALLEL_EXECUTION_PLAN_PE02.json` — nonlinear_f0_f1_tournament
- `KIMI_K26_PARALLEL_EXECUTION_PLAN_PE03.json` — m5_exact_rate_ladder
- `KIMI_K26_RESIDENT_STORAGE_GATE.json` — hawking.kimi_k26.resident_storage_gate.v1
- `KIMI_K26_ROLLBACK.json` — SEALED
- `KIMI_K26_SCIENCE_PUBLICATION_MANIFEST.json` — hawking.kimi_k26.science_publication_manifest.v1
- `KIMI_K26_SOURCE_ADMISSION.json` — Modified MIT License
- `KIMI_K26_SOURCE_FORMAT_LEDGER.json` — PASS
- `KIMI_K26_STATE.json` — JSON receipt
- `KIMI_K26_TEXT_CORE_CLAIM.json` — PASS
- `KIMI_K26_TOURNAMENT.json` — SEALED
- `KIMI_PHONE_STATUS.json` — RUNNING
- `KIMI_PHONE_STATUS.md` — Kimi K2.6 Doctor Prime

## T1 / ROOT_MD — 22 files

- `120B_GATE_F_CLOSEOUT_PRECHECK.md` — 120B GATE-F CLOSEOUT PRECHECK
- `ADAPTIVE_TRANSFER_LADDER_PRECHECK.md` — ADAPTIVE TRANSFER LADDER PRECHECK
- `BASE_RUNTIME_MAXIMIZED_GATE.md` — BASE_RUNTIME_MAXIMIZED — hard promotion gate
- `CONDENSE_AUDIT.md` — CONDENSE_AUDIT.md · /goal condense · branch condense/run-20260703 · 2026-07-03
- `CONDENSE_DOCS_REVIEW.md` — CONDENSE_DOCS_REVIEW.md · staged doc deletions for the grader · condense/run-20260703
- `CONDENSE_LEDGER.md` — CONDENSE_LEDGER.md · /goal condense run · branch condense/run-20260703
- `M1ULTRA_RUN_REPORT.md` — M1 Ultra Run Report: the hawking maximization run
- `MODEL_LADDER_IGNITION_PRECHECK.md` — MODEL LADDER IGNITION PRECHECK
- `MOP_RAMANUJAN_ASSESSMENT_SURVEY.md` — MOP → Ramanujan Reuse Assessment
- `NUCLEAR_PASTA_LEDGER.md` — NUCLEAR PASTA execution ledger
- `NUMERIC_PARITY_V2.md` — Numeric Parity Contract V2
- `PROVIDER_FOUNDRY_V2_PRECHECK.md` — PROVIDER FOUNDRY V2 PRECHECK
- `RAMANUJAN_COGNITION_AMENDMENT.md` — Continuum amendment — the discovery loop
- `RAMANUJAN_COGNITION_CLAUDE.md` — Ramanujan's cognitive architecture — independent pass (Claude)
- `RAMANUJAN_COGNITION_GROK.md` — Ramanujan cognition — independent frontier ideation
- `RAMANUJAN_COGNITION_PROPOSAL.md` — Ramanujan cognitive architecture — consolidated proposal
- `SPEED_RESEARCH_WALL_CLOCK.md` — Speed research: where the wall clock goes, and what removes it
- `SUBBIT_CAPABILITY_DENSITY_PRECHECK.md` — Sub-bit capability-density reset — precheck
- `SUBBIT_CLOSURE_LEDGER.md` — Sub-bit closure ledger — Qwen3-235B-A22B-Instruct-2507
- `TEMPORAL_GRAVITY.md` — Temporal Gravity — minimize causal execution
- `VULTURE_HANDOFF_PRECHECK.md` — VULTURE HANDOFF PRECHECK
- `WATCHLIST.md` — WATCHLIST.md — lapping-condition spec

## T1 / GLM52 — 19 files

- `GLM52_A4_REAL_EXPERT_BANK.json` — hawking.representation.a4_real_bank.v1
- `GLM52_AA_PACK_ALLOCATOR_FIX.json` — hawking.glm52.aa_pack_allocator_fix.v1
- `GLM52_AA_PACK_DRY_RUN.json` — lm_head.weight
- `GLM52_ACTIVATION_AWARE_METAL_RUNTIME_20260727.json` — hawking.glm52.activation_aware_metal_runtime.v1
- `GLM52_ACTIVATION_AWARE_RUNTIME_ABI_AUDIT_20260727.json` — hawking.glm52.activation_aware_runtime_abi_audit.v1
- `GLM52_ASSEMBLY_COVERAGE.json` — hawking.glm52.assembly_coverage.v1
- `GLM52_ASSEMBLY_RESULT.json` — JSON receipt
- `GLM52_BASIS_TRANSFER.json` — hawking.representation.basis_transfer.v1
- `GLM52_CLAIM_A_STATUS.json` — The iso-memory frontier and the RandomPolicy control
- `GLM52_EXPERT_SWEEP.json` — hawking.representation.expert_sweep.v1
- `GLM52_GENERATION_A_DELETION_RECEIPT.json` — hawking.glm52.generation_a_deletion_receipt.v1
- `GLM52_GPU_BASE_TPS.json` — hawking.gravity.glm_base_tps.v1
- `GLM52_PASS3_BYTE_PRECLEARANCE.json` — hawking.prometheus.pass3_byte_preclearance.v1
- `GLM52_PASS3_OPTIMIZATION_AUDIT.json` — hawking.prometheus.pass3_optimization_audit.v1
- `GLM52_POSTPACK_RUNTIME_GATES_20260727.json` — PREPARED_NOT_EXECUTED
- `GLM52_RATE_SWEEP_ALL_ORGANS.json` — hawking.representation.rate_sweep.v1
- `GLM52_SELECTED_GENERAL_ARTIFACT.json` — hawking.glm52.selected_artifact.v1
- `GLM52_TABULA_RASA_LIVE_AUDIT.json` — JSON receipt
- `GLM52_TABULA_RASA_LIVE_AUDIT.md` — GLM-5.2 tabula rasa live audit

## T1 / OTHER — 14 files

- `120B_GATE_F_CLOSEOUT_PRECHECK.json` — hawking.gpt_oss_120b.gate_f_closeout_precheck.v1
- `ADAPTIVE_TRANSFER_LADDER_PRECHECK.json` — hawking.adaptive_transfer_ladder.precheck.v1
- `FIT_KERNEL_V3_REVIEW.json` — VERIFIED BY CLAUDE. Benchmarked and tested independently. Not merged; opt-in only.
- `MODEL_LADDER_IGNITION_PRECHECK.json` — hawking.model_ladder.ignition_precheck.v1
- `MODEL_RELEASE_INTERIM_PARENT.json` — GREEN
- `MOP_RAMANUJAN_TRANSFER_ASSESSMENT.json` — hawking.ramanujan.mop_transfer_assessment.v1
- `NUCLEAR_PASTA_STATE.json` — JSON receipt
- `NUMERIC_PARITY_V2_1_HARNESS.json` — absolute_error_near_zero
- `PROMETHEUS_ARCHITECTURE.json` — NOT_SEALED
- `PROVIDER_FOUNDRY_V2_PRECHECK.json` — hawking.provider_foundry_v2.precheck.v1
- `S1_FAILURE_DECOMPOSITION.json` — JSON receipt
- `SUBBIT_CAPABILITY_DENSITY_PRECHECK.json` — LEVER_ALIVE
- `SUBBIT_CLOSURE_STATE.json` — admitted_next
- `VULTURE_HANDOFF_PRECHECK.json` — RUNNING (do not touch; do not modify its scientific code while alive)

## T1 / OTHER_MODEL — 6 files

- `QWEN235B_DEGRADATION_ATLAS.json` — CLOSURE. This is a PRIOR LIBRARY for the next parent, not a set of universal truths. Every entry names the parent it was
- `QWEN235B_DOCTOR_HANDOFF.json` — hawking.doctor.handoff.v1
- `QWEN235B_TREATMENT_RESPONSE_ATLAS.json` — CLOSURE. Priors for the next parent, not universal truths.
- `QWEN35_397B_ORGAN_INVENTORY.json` — hawking.qwen35_moe.inventory.v1
- `QWEN35_397B_WINDOW_PLAN.json` — hawking.qwen35_moe.window_plan.v1
- `QWEN_FULL_COURSE_PRECHECK.json` — SEALED

## T1 / DOCTOR — 4 files

- `DOCTOR_DIAGNOSIS_ONTOLOGY.json` — JSON receipt
- `DOCTOR_GENERATION_3_REGISTRY.json` — JSON receipt
- `DOCTOR_PRIME_PRECHECK.json` — FIXED and GUARDED. corpus_integrity.py splits by document/context-hash and refuses a shared embedding row via assert_lay
- `DOCTOR_TREATMENT_LIBRARY.json` — hawking.doctor.treatment_library.v1

## T1 / ROOT_YAML — 1 files

- `pnpm-lock.yaml` — lockfileVersion: '9.0'

## T1 / DEEPSEEK — 1 files

- `DEEPSEEK_V4_FLASH_MOE_PROBE_DECISION.json` — the embedding-seeded shortcut is INVALID for DeepSeek's functional existence test, proven by the shared-expert control; 

## T1 / GRAVITY — 1 files

- `GRAVITY_DEGENERATE_ATTRIBUTION.json` — hawking.gravity.degenerate_attribution.v1

## T1 / ROOT_SH — 1 files

- `run_7b_ladder.sh` — #!/usr/bin/env bash

## T2 — superseded campaign narration — 127 files

Pruned **127 tracked markdown files** (~2,073,871 bytes, ~30,329 lines).
Includes root `HAWKING_*_STATUS.md` (tools rewrite on next run), one-shot `docs/plans/**` campaign plans
without executable readers, `docs/hide-impl/**` campaign writeups without readers, other non-living `docs/*.md`
narrative without readers, and superseded `control/final_ascent/contracts/*-revision-N.md` (kept highest N and all
contracts with no revision suffix). Living reference kept: README/ARCHITECTURE/MODELS, docs/{dead_levers,serve,
BENCHMARKS,env_flags,kernels}, docs/gravity/**, docs/hide-bible/**, docs/ARCHIVE_INDEX*.md.
Restore: `git checkout pre-floor-prune-20260728 -- <path>`.

## T2 / control/final_ascent/contracts (superseded revisions) — 27 files

- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-1.md` — Revision 1: close rank, leakage, byte, and lifecycle gaps
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-10-adversarial-ledger-identity.md` — GLM-5 rare-route pilot — Revision 10 adversarial ledger/identity closure
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-11-science-witness-closure.md` — GLM-5 rare-route pilot — Revision 11 science/witness closure
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-12-immutable-closure.md` — GLM-5.2 rare-route basis pilot — Revision 12 immutable closure
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-2.md` — Revision 2: close deployed-representation, provenance, and recovery gaps
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-3.md` — Revision 3: eliminate false-positive science and lifecycle gates
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-4-lifecycle.md` — Revision 4B: implement the real reader and recoverable lifecycle
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-4-science.md` — Revision 4A: make science and provenance validation semantic
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-5-science.md` — Revision 5: close remaining science and provenance false positives
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-6-lifecycle.md` — Revision 6: close remaining lifecycle, recovery, and replay false positives
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-7-science-cleanup.md` — Revision 7: remove remaining science/provenance fail-open paths
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-8-lifecycle-science-closure.md` — Revision 8: close lifecycle recovery and streamed-reader evidence
- `control/final_ascent/contracts/glm52-rare-route-basis-pilot-revision-9-atomic-epoch-replay.md` — GLM-5 2-shard rare-route pilot — Revision 9 atomic epoch/replay closure
- `control/final_ascent/contracts/tg-cheap-hotpath-residual-router-scalars-revision-1.md` — Temporal Gravity cheap hot-path closure — Revision 1 physics/no-regression
- `control/final_ascent/contracts/tg-device-resident-three-batch-mlp-revision-1.md` — TG device-resident ordinary three-batch MLP — Revision 1
- `control/final_ascent/contracts/tg-device-resident-three-batch-mlp-revision-2.md` — TG device-resident three-batch MLP — Revision 2 live-Metal correction
- `control/final_ascent/contracts/tg-device-resident-three-batch-mlp-revision-3.md` — TG device-resident three-batch MLP — Revision 3 device-only correction
- `control/final_ascent/contracts/tg-hide-glm-live-token-path-revision-1.md` — Temporal Gravity GLM live-token path into HIDE — Revision 1
- `control/final_ascent/contracts/tg-hide-glm-live-token-path-revision-2.md` — Temporal Gravity GLM live-token path into HIDE — Revision 2
- `control/final_ascent/contracts/tg-hide-glm-live-token-path-revision-3.md` — Temporal Gravity → HIDE live-token path — Revision 3
- `control/final_ascent/contracts/tg-numeric-parity-near-tie-fallback-revision-1.md` — Near-tie fallback Revision 1: real FP64 authority and durable evidence
- `control/final_ascent/contracts/tg-numeric-parity-near-tie-fallback-revision-2.md` — Temporal Gravity Numeric Parity near-tie fallback — Revision 2
- `control/final_ascent/contracts/tg-numeric-parity-near-tie-fallback-revision-3.md` — Temporal Gravity Numeric Parity near-tie fallback — Revision 3
- `control/final_ascent/contracts/tg-router-bias-residency-bind-once-revision-1.md` — Temporal Gravity router-bias residency — Revision 1
- `control/final_ascent/contracts/tg-runtime-receipt-profiler-hardening-revision-1.md` — Temporal Gravity runtime receipt/profiler hardening — Revision 1
- `control/final_ascent/contracts/tg-runtime-receipt-profiler-hardening-revision-2.md` — Temporal Gravity runtime receipt/profiler hardening — Revision 2
- `control/final_ascent/contracts/tg-runtime-receipt-profiler-hardening-revision-3.md` — Temporal Gravity runtime receipt/profiler hardening — Revision 3

## T2 / docs (other narrative) — 4 files

- `docs/architecture.md` — Hawking architecture map (2026-06-21)
- `docs/design/continuous_batching.md` — Continuous Request Batching for `hawking serve`
- `docs/mixtral.md` — Mixtral 8×7B support
- `docs/rwkv7.md` — On-device instruct post-train: RWKV7-g1-0.4B (M3 Pro, $0, overnight)

## T2 / docs/hide-impl — 16 files

- `docs/hide-impl/FIRST_RECEIPT_HARNESS.md` — First Mandatory Implementation Receipt - Harness Spec
- `docs/hide-impl/PHASE0_CRATE_MAP.md` — Phase 0: Crate Map (target vs current, reconciled)
- `docs/hide-impl/PHASE0_SOURCE_LEGAL_INTAKE.md` — Phase 0: Source and Legal Intake Ledger
- `docs/hide-impl/consolidation/HIDE_CODEX_DEPTH_MAP.md` — HIDE Codex Depth Map
- `docs/hide-impl/consolidation/HIDE_CONTROL_DENSITY_SCORECARD.md` — HIDE Control Density Scorecard
- `docs/hide-impl/consolidation/HIDE_DEAD_DUPLICATE_CONTROL_REPORT.md` — HIDE Dead / Duplicate / Misleading / Mock Control Report
- `docs/hide-impl/consolidation/HIDE_FINAL_CONSOLIDATION_REPORT.md` — HIDE Final Consolidation Report
- `docs/hide-impl/consolidation/HIDE_GROK_BUILD_DEPTH_MAP.md` — HIDE x grok-build depth map
- `docs/hide-impl/consolidation/HIDE_OPENCODE_DEPTH_MAP.md` — HIDE opencode Depth Map
- `docs/hide-impl/consolidation/HIDE_PRODUCTIVITY_DENSITY_BASELINE.md` — HIDE Productivity Density Baseline
- `docs/hide-impl/consolidation/HIDE_SURFACE_WITHOUT_BACKEND_REPORT.md` — HIDE Surface Without Backend Report
- `docs/hide-impl/consolidation/HIDE_UI_CONTROL_CENSUS.md` — HIDE UI Control Census
- `docs/hide-impl/consolidation/HIDE_WORKFLOW_BEFORE_AFTER.md` — HIDE Workflow Before and After
- `docs/hide-impl/preview/HIDE_FOURTH_ADVERSARIAL_REPORT.md` — HIDE fourth adversarial report
- `docs/hide-impl/preview/HIDE_PREVIEW_CLOSEOUT_REPORT.md` — HIDE Preview Closeout Report
- `docs/hide-impl/preview/HIDE_PREVIEW_GALLERY.md` — HIDE Preview Gallery

## T2 / docs/plans — 75 files

- `docs/plans/APPENDIX.md` — The Appendix
- `docs/plans/APPENDIX_HANDOFF.md` — Appendix handoff and incorporation contract
- `docs/plans/CONDENSATION_DOCTOR_V2.md` — Condensation Doctor v2 — Program Synthesis for Capability Restoration
- `docs/plans/DOCTOR_V5.md` — Condensation Doctor v5 — Canonical Capability-First Specification
- `docs/plans/DOCTOR_V5_3BPW_RESOURCE_STOP_RECOVERY.md` — Doctor V5 14B/3bpw resource-stop recovery staging
- `docs/plans/DOCTOR_V5_AGGRESSIVE_ADMISSION_V2.md` — Doctor V5 aggressive admission v2 (unbound scaffold)
- `docs/plans/DOCTOR_V5_BLOCKED_CELL_RECOVERY.md` — Doctor V5 14B/4bpw blocked-cell recovery
- `docs/plans/DOCTOR_V5_ELASTIC_HOST_SPRINT.md` — Doctor V5 elastic phases and host sprint isolation
- `docs/plans/DOCTOR_V5_MOUNTAIN_LADDER.md` — Doctor V5 post-120B mountain ladder
- `docs/plans/DOCTOR_V5_PARALLEL_ACCELERATION_HANDOFF.md` — Doctor V5 unbound GPT-OSS and higher-tier acceleration handoff
- `docs/plans/DOCTOR_V5_PHYSICAL_ADAPTER_AUTHORITY.md` — Doctor V5 physical adapter authority
- `docs/plans/DOCTOR_V5_PHYSICAL_POST120_NEXT_GOAL_PROMPT.md` — Restart-proof next-goal prompt
- `docs/plans/DOCTOR_V5_POST120_ACCELERATION.md` — Doctor V5 120B and post-120B acceleration handoff
- `docs/plans/DOCTOR_V5_POST_120B.md` — Doctor V5 post-120B handoff
- `docs/plans/DOCTOR_V5_REMAINING_SCRATCH_LEDGER.md` — Doctor V5 phase-aware remaining-scratch ledger
- `docs/plans/DOCTOR_V5_RESEARCH_PASSES.md` — Condensation Doctor v5 — Three Expansion Passes and Adversarial Proof Plan
- `docs/plans/DOCTOR_V5_SINGLE_DEVICE_SPRINT.md` — Doctor V5 single-device sprint handoff
- `docs/plans/DOCTOR_V5_TELEGRAM_NOTIFICATIONS.md` — Doctor V5 Telegram rung notifications
- `docs/plans/FRONTIER_ECOSYSTEM_SCAFFOLD.md` — Condenser Ecosystem Frontier: scaffold status
- `docs/plans/HAWKING_GLM52_GENERATION_B_STREAMING_FRONTIER_ACTION_PLAN.md` — /goal
- `docs/plans/HAWKING_GRAVITY_FORGE.md` — Hawking Gravity Forge
- `docs/plans/HAWKING_GRAVITY_RUNBOOK.md` — Hawking Gravity Runbook
- `docs/plans/HAWKING_LAPTOP_WAVE_2026_07_08.md` — Hawking laptop wave - 2026-07-08
- `docs/plans/HAWKING_SEED_ARCHITECTURE.md` — Hawking Seed — architecture specification + 3-candidate design (Phase 1 oracle)
- `docs/plans/HIDE_CONDENSER_GOAL_PROMPT.md` — HIDE condenser goal prompt
- `docs/plans/HIDE_REFINEMENT_RUN_REPORT.md` — HIDE refinement run report
- `docs/plans/M1ULTRA_GOAL_PROMPT.md` — M1 ULTRA GOAL PROMPT: the iterative maximization loop for the delivered box
- `docs/plans/M1ULTRA_POTENTIAL_AUDIT.md` — M1 ULTRA POTENTIAL AUDIT: every evaluated category graded on the delivered box's ceiling
- `docs/plans/STUDIO_DEEP_AUDIT_2026_07_08.md` — Studio deep audit - 2026-07-08
- `docs/plans/STUDIO_GO.md` — STUDIO GO — the one-command entry point for the Hawking frontier program
- `docs/plans/STUDIO_MODEL_LADDER.md` — Studio Model Ladder — Condensation, Not Quantization
- `docs/plans/SUCCESSOR_RUNBOOK.md` — Successor condenser: operational + rollback runbook
- `docs/plans/TABULA_RASA_SHIPPING_PLAN.md` — Hawking Tabula Rasa Shipping Plan
- `docs/plans/TRAINING_LADDER_V5.md` — Hawking Training Ladder v5 — Capability-First Condensation
- `docs/plans/agentic_tool_system_audit_2026_07_11.md` — Agentic Tool System: scaffold audit + rating (2026-07-11)
- `docs/plans/apple_fit_frontier_2026_06_22.md` — Apple Fit Frontier (2026-06-22)
- `docs/plans/bible_active.md` — > **THROUGHPUT BIBLE — ACTIVE (lean working doc).** Split 2026-05-31 from `throughput_bible_2026_05_30.md`. This is the 
- `docs/plans/bible_archive.md` — > **ARCHIVE / REFERENCE — the long companion to [bible_active.md](bible_active.md)** (split 2026-05-31 from this file's 
- `docs/plans/computational_efficiency_paradigms_2026_07_11.md` — Beyond FLOPS: a cross-layer research agenda for computational efficiency
- `docs/plans/condense_autopilot_2026_06_27.md` — Condense Autopilot - 2026-06-27
- `docs/plans/condense_naming_migration_2026_06_22.md` — Condense Naming Migration (2026-06-22)
- `docs/plans/doctor_capability_and_speed_roadmap.md` — Doctor → ~1:1 quality + Speed, BEFORE the 32B (2026-06-23)
- `docs/plans/doctor_maximization_plan.md` — Doctor Maximization Plan — Recovery to Outrun Compression's Decay
- `docs/plans/g1a_v2_expansion_results_2026_06_20.md` — G1a V2 Expansion Chain Results
- `docs/plans/h_autoverify_protocol.md` — Event Horizon Auto-Handoff Protocol (2026-06-21)
- `docs/plans/hawking_capability_frontier_2026_06_28.md` — Hawking IDE — Capability Frontier & Build Roadmap
- `docs/plans/hawking_capability_frontier_2026_06_28_HANDOFF_PROMPT.md` — Build-session handoff prompt (paste into the integrated/coding session)
- `docs/plans/hawking_event_horizon_phase0_blueprint.md` — I have everything grounded against the real source. The skeletons in the designs match the actual signatures (`verify_dr
- `docs/plans/hawking_event_horizon_proposal_engine.md` — Hawking Event Horizon — Unified Speculative Proposal Engine
- `docs/plans/hawking_event_horizon_status.md` — Hawking Event Horizon — As-Built Status (2026-06-20)
- `docs/plans/hawking_gravity_maximal_fidelity_ladder.md` — Hawking Gravity: maximal-fidelity ladder (as close to raw models as the box allows)
- `docs/plans/hawking_handoff_2026_06_28.md` — Hawking — Maximal Handoff & Codebase Understanding (2026-06-28)
- `docs/plans/hawking_ide_claude_research_handoff_2026_07_19.md` — Claude handoff: independent Hawking IDE frontier pass
- `docs/plans/hawking_ide_frontier_2026_07_19.md` — Hawking IDE frontier dossier
- `docs/plans/hawking_shippability_masterplan_2026_06_22.md` — Hawking — Shippability Master Plan (2026-06-22)
- `docs/plans/hide_command_system_maximalist_2026_06_29.md` — HIDE — The Maximalist Command System (trigger grammar + local-unlimited depth)
- `docs/plans/hide_deep_audit_2026_07_16.md` — HIDE deep audit and facet ladder
- `docs/plans/hide_executor_plan_v2_enriched_2026_06_29.md` — HIDE Executor — Enriched Master Plan (v2)
- `docs/plans/hide_handoff_2026_06_28.md` — HIDE handoff prompt (paste into the dedicated HIDE chat) — 2026-06-28
- `docs/plans/hide_master_build_plan_2026_06_29.md` — HIDE / Hawking — Master Build Plan (2026-06-29)
- `docs/plans/hide_refinement_roadmap_2026_07_05.md` — HIDE refinement roadmap
- `docs/plans/hide_research_menu_2026_06_29.md` — HIDE Agentic-Frontier Research Menu (2026-06-29)
- `docs/plans/hide_ship_readiness.md` — HIDE — Ship Readiness (living status, 2026-06-29)
- `docs/plans/hide_ship_status_2026_06_29.md` — HIDE — Ship Status (2026-06-29)
- `docs/plans/hide_sota_frontier_and_regrade_2026_07_16.md` — HIDE SOTA frontier read + regraded ladder (condenser pass)
- `docs/plans/hide_ux_audit_and_research_2026_06_29.md` — HIDE — Deep Audit + UX Research (2026-06-29)
- `docs/plans/parameter_sweep_pipeline.md` — Hawking Condense — Parameter-Sweep Testing Pipeline (2026-06-23)
- `docs/plans/q6k_predec_design.md` — Q6_K predec ffn_down — implementation design (campaign R-design, 2026-06-21)
- `docs/plans/quintessential_engine_2026_06_29.md` — Hawking — the quintessential local inference engine (unified plan, 2026-06-29)
- `docs/plans/ratios_roadmap_2026_06_21.md` — Ratios Roadmap — speed · compression · density (validated 2026-06-21)
- `docs/plans/spec_decode_reentry_appendix_2026_07_14.md` — Speculative-decode re-entry appendix — 2026-07-14
- `docs/plans/spec_decode_studio_readiness_2026_07_12.md` — Speculative decoding on the 96 GB Studio: evidence and readiness
- `docs/plans/storage_stripdown_resident_first_2026_07_20.md` — Storage stripdown and the resident-first resequencing
- `docs/plans/throughput_pivot_campaign.md` — Throughput-Pivot Campaign — live autonomous run (started 2026-06-21)
- `docs/plans/tq_compute_for_memory_appendix_2026_07_14.md` — TQ compute-for-memory appendix — 2026-07-14

## T2 / root HAWKING_*_STATUS.md — 5 files

- `HAWKING_CONTINUUM_STATUS.md` — HAWKING CONTINUUM STATUS
- `HAWKING_FINAL_ASCENT_STATUS.md` — HAWKING FINAL ASCENT STATUS
- `HAWKING_LIGHT_ONLY_STATUS.md` — HAWKING LIGHT-ONLY STATUS
- `HAWKING_MOTHERLOAD_STATUS.md` — Hawking Motherload Completion Status
- `HAWKING_PARALLEL_STATUS.md` — HAWKING PARALLEL CONTINUATION STATUS

