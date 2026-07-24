# Hawking Motherload Completion Status

endpoint: `IN_PROGRESS`  
gates closed: 0/20  
ODYSSEY_LAUNCH_AUTHORIZED: `false`  
updated: 2026-07-24T20:14:37Z

| gate | state | condition | note |
|---|---|---|---|
| M01 | running | GLM traversal is 282/282 | GLM source traversal live: 37/282 verified, window W001/20, launchd com.hawking.glm52.source-fetch pid 72608, 0 faults |
| M02 | open | complete local GLM .gravity artifact exists |  |
| M03 | open | lowest broad parity rate is sealed, sub-bit preferred and H15 maximum |  |
| M04 | running | full GLM token executes from .gravity | GLM adapter (MLA+DSA+IndexShare+noaux_tc router+routed/shared experts) matches the numpy oracle at 3.8e-6 with argmax, top5 and DSA top-k exact on a tiny-GLM .gravity fixture; flagship artifact still traversing |
| M05 | running | measured base TPS and prefill exist | BASE_TRUE_TPS measured on the Llama .gravity instrument: decode 105.8/68.8/29.2/13.3 tok/s and prefill 116.5/92.8/48.8/19.1 tok/s at ctx 128/512/2048/8192, 1 command buffer and 210 dispatches per token; GLM numbers gated on the flagship artifact |
| M06 | open | acceleration stack is terminal and measured separately |  |
| M07 | open | GLM runs end to end inside HIDE |  |
| M08 | open | Prometheus S0 and source decision are sealed |  |
| M09 | open | Prometheus architecture and profiles are implemented |  |
| M10 | open | equal-budget Claim A is sealed |  |
| M11 | open | General and Math artifacts are selected and verified |  |
| M12 | running | Forge, continuity, sovereignty, and Limit Registry are sealed | sovereignty spine + 13 invariants green; false_refusal/boundary_error gated on a served model |
| M13 | open | Odyssey substrate and training bundle are complete |  |
| M14 | open | sandbox, roles, Ledger, verifiers, Tribunal, and retrieval are scaffolded |  |
| M15 | open | Lean/Mathlib and evidence environment are pinned |  |
| M16 | open | Odyssey dry-run validation passes |  |
| M17 | open | ODYSSEY_LAUNCH_AUTHORIZED remains false |  |
| M18 | running | rollback/source lifecycle is green | rollback seal replays the sealed receipt; 37/282 verified, 0 hash mismatches |
| M19 | open | all campaign commits are pushed |  |
| M20 | open | worktree and process state are clean except intentional detached services |  |
