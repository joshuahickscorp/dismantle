# Temporal Gravity Numeric Parity V2.1 near-tie fallback

Implement the missing frozen/versioned near-tie canonical fallback in an
isolated worktree. This is a bounded correctness prerequisite for later kernel
promotion, not a flagship benchmark or permission to weaken parity.

## Authority and scope

Read:

- `NUMERIC_PARITY_V2.md`;
- `crates/hawking-core/src/numeric_parity.rs`;
- `crates/hawking-core/src/gravity_glm.rs`;
- `crates/hawking-core/src/gravity_glm_resident.rs`;
- corresponding GLM parity and fixture tests;
- current runtime receipts for router, DSA/top-k, final head, and ICB paths.

Inspect current source before choosing insertion points. Preserve exact artifact
semantics, same-backend determinism, and existing default-off optimization
flags. Do not read real weights, run heavy model work, touch MOP, or publish a
performance claim.

## Required behavior

Define one versioned `NearTiePolicy` with:

- explicit schema/policy version;
- decision domain and numeric dtype/backend;
- frozen finite nonnegative absolute and relative guard parameters;
- canonical comparison/order rule, including stable index tie-break;
- exact trigger formula using the relevant top decision margin;
- explicit fallback implementation identity;
- counters for comparisons, guard hits, canonical fallbacks, and decisions;
- semantic seal/config identity suitable for runtime receipts.

The policy must cover every load-bearing discrete decision that the current GLM
runtime exposes through bounded fixture execution, including router/top-k and
final token/top-k selection. Extend to DSA or other exact discrete decisions
when those paths already share the same decision helper. Do not claim coverage
for an unwired path.

When the guard does not trigger, the existing fast decision remains
byte-for-byte and decision-for-decision unchanged. When it triggers, recompute
the smallest decisive slice through one deterministic canonical path and
require exact stable decisions. NaN, infinity, duplicate indices, invalid
top-k, empty inputs, and unsupported dtype/backend refuse.

The policy must be configurable and fully receipted. Default selection must
follow the existing Numeric Parity authority; if authority does not specify a
production default or calibrated thresholds, leave promotion default-off and
report the missing calibration instead of inventing one.

## Tests

Add deterministic source-body-free tests for:

- exact equality and stable index tie-break;
- values just below, exactly at, and just above each guard;
- positive/negative values, zeros, subnormals when supported, NaN/infinity;
- same-backend repeated determinism;
- condition-aware cross-backend perturbations around the guard;
- router/top-k and token/top-k exact decisions;
- fast-path identity outside the guard;
- counter reconciliation and policy seal stability;
- invalid/missing/stale policy versions and receipt bindings;
- no fallback when disabled and explicit proof it is disabled.

Add a bounded microbenchmark that reports fast-path overhead and guard-hit
overhead separately. It must not claim whole-model TPS.

## Authorized files

Change only the smallest necessary subset of:

- `crates/hawking-core/src/numeric_parity.rs`;
- `crates/hawking-core/src/gravity_glm.rs`;
- `crates/hawking-core/src/gravity_glm_resident.rs`;
- directly corresponding tests/benchmarks.

No status, artifact, capability, launch, or authorization file may change.

## Acceptance

Run targeted and affected Rust tests, formatting/lints available in the
repository, the bounded microbenchmark, and `git diff --check`. Report exact
files/hashes, policy semantics, paths actually wired, measured microbenchmark
cost, paths still unwired, and calibration still required.

State explicitly that no capable provider, TPS/TG milestone, real model
measurement, or MOP action occurred and all fences remain false.
