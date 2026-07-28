# Implement the bounded GLM-5.2 real-activation basis pilot

## Goal

In an isolated Grok worktree based on current HEAD
`299208e6e2ed22d6b74c6bbd391dc631b4e29ec5`, implement and execute the smallest
decisive real-capsule comparison needed before another full GLM traversal:

1. existing centered residual SVD,
2. uncentered SVD,
3. explicit normalized mean direction plus a centered residual basis orthogonal
   to it.

Return a bounded, deterministic pilot and measured receipt. Do not change
production defaults merely because an arm wins a pilot.

## Evidence and inputs

Read:

- `HAWKING_RESUME_CHECKPOINT.md`
- `GLM52_GENERATION_B_CAPABILITY_VERDICT.json`
- `GLM52_BYTE_ATTRIBUTION.json`
- `GLM52_REBUILD_PILOT_RECEIPT.json`
- `GLM52_REAL_ACTIVATION_SWEEP.json`
- `tools/condense/glm52_activation_aware_pack.py`
- `tools/condense/tests/test_glm52_activation_aware_pack.py`

Use real retained capsules only:

`/Users/scammermike/Library/Application Support/hawking/GLM52Gravity/source_fetch/teacher/capsules`

Resolve resident immutable GLM source tensors from live manifests and exact hashes.
If source bodies needed for the pilot are absent, do not silently substitute
Gaussian data or a small fixture. Prefer tensors whose exact source shard is
already resident; otherwise produce an explicit data-blocked receipt naming the
minimum immutable shard(s) needed and a safe official rehydration command.

## Scientific contract

- Use identical fit/holdout row indices for all three variants.
- Use real contextual held-out activations for every promotion metric.
- Compare equal total encoded bytes, not nominal equal residual rank. The explicit
  mean vector/direction and all coefficients/metadata count physically.
- Evaluate representative critical classes when reachable: routed expert
  gate/up/down, dense/shared MLP, attention projections, router/control, and
  global tensors. Explain any missing class.
- Ranks must cover the calibrated region, including 16/64 as negative controls
  and higher feasible ranks such as 128/256/512 where capsule row count permits.
- Report per tensor, per organ, per layer, minimum, median, worst case, exact
  bytes/BPW, constant-mean null, and output/token-relevant diagnostics available
  in bounded scope.
- `beats_null` and reconstruction error are diagnostics only.
- No Gaussian proxy may select a representation.
- No full 282-shard traversal or whole-model pack.

## Implementation requirements

- Prefer a separate pilot module or explicit opt-in basis-mode argument so the
  existing production path cannot change accidentally.
- Add deterministic unit tests for orthogonality, mean-direction inclusion,
  fit/holdout identity, byte accounting, and regression of the existing centered
  behavior.
- Add a test that would fail if the explicit mean arm gets one extra direction
  for free.
- Emit machine-readable JSON and a concise Markdown receipt with hashes of code,
  capsule inputs, tensor source inputs, preregistered thresholds, measured
  results, verdict, and remaining uncertainty.
- Commit all intended files on the Grok branch. Do not include `.serena` files.

## Forbidden

- Do not touch MOP, its processes, caches, files, or launch agents.
- Do not modify/delete teacher capsules, prior negative controls, model bodies,
  or any current artifact.
- Do not alter live launchd jobs or kill any process.
- Do not flip `ODYSSEY_LAUNCH_AUTHORIZED`,
  `RAMANUJAN_RESEARCH_AUTHORIZED`, or `HIDE_KERNEL_TURN`.
- Do not claim a bounded tensor pilot proves whole-model capability.

## Required report

Return independent diagnosis, competing explanations, files changed, exact tests
run, measured result, distinguishing verdict, remaining uncertainty, and the
next safe action. A skipped test is not evidence.
