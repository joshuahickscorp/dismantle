# HAWKING PARALLEL CONTINUATION STATUS

Generated from live evidence by `tools/campaign/parallel_status.py`. Do not hand-edit.
Ownership and the DAG are hand-authored architecture and are read, never written, by this tool.

    at:       2026-07-27T01:58:19Z
    endpoint: RAMANUJAN_SANDBOX_READY (not reached)
    width:    1/6 lanes running
    fences:   ODYSSEY_LAUNCH_AUTHORIZED=False RAMANUJAN_RESEARCH_AUTHORIZED=False HIDE_KERNEL_TURN=False

## Running

- `L01` hide-archaeology-v2 (grok) -- outputs missing: HIDE_ARCHAEOLOGY_V2.md, HIDE_ARCHAEOLOGY_V2.json

## Finished, awaiting controller review

- `L02` hide-memory-six (grok) -- DECLARED RUNNING BUT EVIDENCE FINISHED_AWAITING_REVIEW
- `L03` odyssey-t0 (grok) -- outputs missing: ODYSSEY_T0_RECEIPT.json, ODYSSEY_CONTRACT_CLOSURE.json, ODYSSEY_FEASIBILITY.json -- DECLARED RUNNING BUT EVIDENCE FINISHED_OUTPUTS_MISSING
- `L04` fabric-software (grok) -- DECLARED RUNNING BUT EVIDENCE FINISHED_AWAITING_REVIEW
- `L05` bridge-events-adapters (grok) -- DECLARED RUNNING BUT EVIDENCE FINISHED_AWAITING_REVIEW
- `L06` consolidation-inventory (grok) -- DECLARED RUNNING BUT EVIDENCE FINISHED_AWAITING_REVIEW

## Queued

- `L07` speculation-safety (grok)
- `L09` ramanujan-migration-prep (claude)

## Blocked (real data dependencies, see the DAG)

- `L08` hide-os-wiring (grok)
- `L10` odyssey-t1-t7 (unassigned)
- `L11` math-frozen (unassigned)
- `L12` consolidation-execute (unassigned)
- `L13` ramanujan-migrate (unassigned)
- `L14` ramanujan-cognition (unassigned)
- `L15` local-forge-training (unassigned)
- `L16` search-governance (unassigned)
- `L17` q0-q6-qualification (unassigned)

## Hard walls on the critical path

- **L10**: training a 92 GB artifact on a 96 GB machine that is already at load ~29 of 28 cores because a separate campaign (MOP) holds most of them
- **L10**: ODYSSEY_LAUNCH_AUTHORIZED is false and only the controller may flip it, after G1-G4 of ODYSSEY_PROMOTION_GATE.md
- **L15**: F0-F9 local training requires the frozen Math-Frozen Director, which is downstream of the L10 wall

## Next action

    review L02 (hide-memory-six) -- a report is a claim; re-derive before merging

```bash
cat /Users/scammermike/.claude-grok/tasks/hide-memory-six-20260726-213344/grok-report.md
```
