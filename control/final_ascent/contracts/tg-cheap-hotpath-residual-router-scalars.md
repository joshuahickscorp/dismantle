# Temporal Gravity cheap hot-path closure

Implement three bounded, default-off hot-path improvements in one isolated
worktree. They must preserve Numeric Parity V2.1 and existing default behavior.
Use only source-body-free fixtures and microbenchmarks. Do not read real model
bodies, run a flagship benchmark, touch MOP, change production defaults, or
claim a TG/TPS promotion.

## Authority

Read current source and receipts for:

- the ordinary three-batch GLM MLP path;
- resident residual state;
- device router and expert table hit/miss handling;
- expert-wave negative sealed evidence;
- final norm/device lm-head graph;
- persistent selected-route resources and scalar buffers;
- Numeric Parity V2.1 and its harness.

Live code overrides stale `TEMPORAL_GRAVITY.md`. Keep the rejected expert-wave
and ICB default-on results rejected.

## C1. Device residual on the ordinary three-batch MLP

The existing residual-closed device path must not require expert-wave.

- Preserve the ordinary gate/up/down three-batch execution and its exact
  combine order.
- After the final batch, perform the residual add on device using the same
  value/order semantics as the host reference.
- Keep the residual device-resident for the next layer when resident state is
  enabled.
- Remove only the host residual copy/combine barrier made redundant by this
  path.
- Do not enable, depend on, or silently route through expert-wave.
- Count physical command buffers, host/device residual bytes, waits, and
  dispatches before/after.
- Refuse unsupported dtype/layout/codec rather than substituting a different
  arithmetic path.

The optimization is accepted on fixtures only if the residual stream is
V2.1-valid and all expert, DSA, router, top-k, and token decisions remain exact.

## C2. Device router independent of expert-wave

Permit the already-qualified device router to eliminate full router-logit
download without requiring expert-wave.

- `DEVICE_ROUTER=1` with `EXPERT_WAVE=0` is a first-class tested matrix entry.
- Preserve exact noaux_tc/group selection, expert IDs, weights, order, and
  stable tie behavior.
- Keep router bias and score buffers distinct; include an aliasing regression.
- When the device expert table has a valid hit, consume selected IDs/weights
  entirely on device.
- Download only the smallest selected-ID evidence needed for a verified host
  admission on a real table miss.
- Miss, stale address generation, unsupported codec, or inconsistent table
  identity fails closed before residual mutation and follows a separately
  receipted fallback.
- Do not claim zero host traffic from a fixture that never executes a miss.

Report exact bytes downloaded on hit and miss, physical waits, and decision
parity.

## C3. Persistent final-norm weights and head scalars

Remove per-token host writes that do not change:

- bind final RMSNorm weight once per resident session/address generation;
- retain a stable GPU scalar arena for position, token, head/sample controls,
  and other already-supported head scalars;
- update only the values that actually change;
- invalidate and rebuild on buffer/address generation change;
- preserve the existing final norm, lm-head, argmax/top-k, and token decision
  semantics;
- do not enable `LM_HEAD_ICB`, full-vocab readback, or any rejected ICB default.

Measure norm/scalar host-write bytes per token, binding/rebuild counts, and
fast-path overhead. A helper-only microbenchmark is insufficient; exercise the
real fixture insertion points.

## Tests

Add deterministic source-body-free tests for:

- C1 residual device versus host reference across multiple layers and tokens;
- C1 with expert-wave explicitly off and the ordinary three-batch path proved;
- C1 invalid layout/codec, injected command failure, and rollback;
- C2 device-router on with wave off, exact experts/weights/order;
- C2 table hit, verified miss, stale generation, unsupported codec, and
  bias/score alias trap;
- C2 byte/wait counters that distinguish hit and miss;
- C3 one initial bind, zero unchanged norm uploads on warm tokens, scalar-only
  updates, address-generation rebuild, and error recovery;
- complete flag matrix with all new behavior off matching current default
  bytes/decisions;
- combined C1+C2+C3 fixture execution with V2.1 continuous metrics and exact
  router/DSA/expert/top-k/token decisions.

Run disabled baseline, each feature alone, and the combined path. Report
physical topology and micro-latency separately; never call fixture timing
`BASE_TRUE_TPS`.

## Authorized files

Change only the smallest necessary subset of:

- `crates/hawking-core/src/gravity_glm_resident.rs`;
- `crates/hawking-core/src/gravity_glm.rs` for flag/dependency plumbing only;
- an existing Metal residual/scalar kernel file only if the required exact
  operation is not already available;
- directly corresponding tests and bounded fixture benchmarks.

Do not modify numeric-parity policy code, serving/HIDE seams, artifacts,
receipts representing real measurements, launch/status files, or fences.

## Acceptance

Run targeted and affected Rust tests, Numeric Parity V2.1 fixture gates,
formatting/lints available in the repository, bounded insertion-point
microbenchmarks, deterministic replay, and `git diff --check`.

Report exact files/hashes, flags and defaults, before/after physical counters,
microbenchmark context, parity results, hit/miss evidence, remaining flagship
gate, and integration overlap with other TG worktrees.

State explicitly that no real model, product TPS, TG milestone, HIDE
production promotion, or MOP action occurred and all fences remain false.
