# Hawking Motherload Completion Status

endpoint: `IN_PROGRESS`  
gates closed: 5/20  
ODYSSEY_LAUNCH_AUTHORIZED: `false`  
updated: 2026-07-24T20:50:10Z

| gate | state | condition | note |
|---|---|---|---|
| M01 | running | GLM traversal is 282/282 | traversal live at 59/282 (W003); eviction had authorized nothing for 3 windows because every teacher capsule was sealed under the retired 8-token calibration -- stale capsules archived with a withdrawal receipt and the chain re-seeded from layer 0 under the live 256-token corpus calibration |
| M02 | open | complete local GLM .gravity artifact exists |  |
| M03 | open | lowest broad parity rate is sealed, sub-bit preferred and H15 maximum |  |
| M04 | running | full GLM token executes from .gravity | production path closed on the instrument: load_engine dispatches on container magic, GravityEngine streams tokens through the reviewed registry with no source weights; GLM adapter green on fixture and awaiting the flagship artifact |
| M05 | running | measured base TPS and prefill exist | base measured on the instrument: decode 105.8/68.8/29.2/13.3 tok/s at ctx 128/512/2048/8192, prefill 116.5/92.8/48.8/19.1, 1 command buffer + 210 dispatches per token; end-to-end generation 60.7 tok/s with incremental decode bit-identical to full replay; GLM numbers gated on the flagship artifact |
| M06 | open | acceleration stack is terminal and measured separately |  |
| M07 | open | GLM runs end to end inside HIDE |  |
| M08 | open | Prometheus S0 and source decision are sealed |  |
| M09 | green | Prometheus architecture and profiles are implemented | all 14 Revision 3 §7 components implemented and wired: 8 measured, 5 gated with named gates, 1 sealer; profiles general/math/uniform/random compiled and hashed; equal-budget solver matches all four arms to 0.0175% at 46.70 GB |
| M10 | running | equal-budget Claim A is sealed | Claim A NOT_SEALED and correctly so: allocation plans are byte-matched and ready, but retention is deliberately null -- at equal bytes retention IS the claim. Blocked on S0.8 cartography membership and the served flagship. |
| M11 | open | General and Math artifacts are selected and verified |  |
| M12 | running | Forge, continuity, sovereignty, and Limit Registry are sealed | sovereignty sealed for the live artifact: continuity manifest names the exact body hash, fallback_allowed=false, hidden_intervention 0.0, model_continuity 1.0, attribution_completeness 1.0, 13 invariants green; false_refusal and boundary_error remain GATED on a served flagship and never appear as numbers |
| M13 | running | Odyssey substrate and training bundle are complete | training bundle complete: plan T0-T5, objective/checkpoint/evaluation contracts, data + teacher-trace manifests, profile manifest; substrate itself still GATED on M11 and declared so rather than named speculatively |
| M14 | green | sandbox, roles, Ledger, verifiers, Tribunal, and retrieval are scaffolded | sandbox policy (network deny-by-default, filesystem allowlist, emergency stop), 12 roles with promotion held only by verifier and Tribunal, Ledger contract, 4-tier lattice, 7 memory stores, Tribunal + prior-art protocol, retrieval against a pinned snapshot, branch economics, Graveyard |
| M15 | green | Lean/Mathlib and evidence environment are pinned | Lean leanprover/lean4:v4.15.0 and Mathlib v4.15.0 pinned to concrete revisions; validator rejects 'latest'; container digest declared with gate ODYSSEY-ENV-01 |
| M16 | green | Odyssey dry-run validation passes | odyssey_package.py validate: 86 checks, 0 failed, DRY_RUN_PASS; selftest proves a flipped fence FAILS validation and the runner exits 1 |
| M17 | green | ODYSSEY_LAUNCH_AUTHORIZED remains false | ODYSSEY_LAUNCH_AUTHORIZED=false; the builder reads the fence and never writes it, so rebuilding cannot authorize a run |
| M18 | running | rollback/source lifecycle is green | eviction VERIFIED FIRING: free disk 274.6 -> 404.8 GiB at the W003/W004 boundary, 1 EVICT event, 0 faults; traversal now sustainable to 282/282 |
| M19 | open | all campaign commits are pushed |  |
| M20 | open | worktree and process state are clean except intentional detached services |  |
