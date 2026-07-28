# Temporal Gravity cheap hot-path closure — Revision 1 physics/no-regression

Revise the frozen candidate in place. Preserve every earlier default-off,
Numeric Parity V2.1, non-wave, no-ICB, no-real-model, no-MOP, and false-fence
requirement.

Frozen candidate:

- `gravity_glm.rs` blob `2d651c1bac3ff5cc334520bc033e5cec5da55188`;
- `gravity_glm_resident.rs` blob `10aec3b1fc140856199a283122412fc232c94678`;
- fixture test blob `51efc9228d210f1e6d7da1afde0afae28d66fa89`.

An unsandboxed live Metal run of the candidate produced:

- baseline: about 55,480 us, 39 waits;
- C1: 51 waits;
- C2: 102 waits;
- C3: 102 waits;
- combined: about 62,980 us, 114 waits;
- `V2.1 continuous pass=false`, yet the test passed because it asserted only
  discrete fields.

This candidate is rejected as an acceleration. Fix the causes or remove/disable
the non-improving feature implementation. A default-off flag is not a waiver
for an adverse promoted path.

## 1. Numeric Parity is an unmaskable gate

For baseline, C1, C2, C3, and combined, run the complete frozen V2.1 continuous
and discrete contract at every required insertion/output surface. Assert the
top-level `score_pair.pass` plus the individual continuous/discrete fields.
Printing `pass=false`, checking only greedy/top-k, or treating the baseline f32
output as FP64 authority is forbidden.

Use source-body-free authoritative f64 fixture computation from original typed
inputs. Cover logits, residual, router scores/weights, expert IDs/execution
slots, DSA order, final top-k, and token decision. Near-tie-dependent C2 remains
qualification-refused until the versioned true-FP64 policy lands.

Any parity failure makes the test and candidate fail. Do not loosen V2.1
bounds.

## 2. Zero added steady-state synchronization

C1/C2/C3 exist to remove hot-path overhead. In steady warm execution, no
candidate may add a per-layer/per-token CPU-visible `commit_and_wait`, shared
buffer readback wait, or command-buffer boundary relative to the exact baseline
for the work it replaces.

- C1 residual must encode into an already required ordinary non-wave command
  buffer or remain host-side. An extra wait per MLP layer is a kill.
- C2 must not commit/wait separately at every router stage. Preserve exact k
  IDs+weights readback, but batch device work and use the minimum unavoidable
  synchronization boundary. If physical topology cannot improve versus host
  routing on the fixture, kill C2 as an acceleration.
- C3 persistent norm/scalars must not inherit or activate extra full-head waits,
  replay, ICB, or full-logit readback solely for the test. Compare the same head
  mode and output contract on both sides.

The combined path must not accumulate candidate-local waits. Freeze independent
counts at the actual command submission, encoder dispatch, synchronization,
shared-read, shared-write, and blit APIs; logical/session counters are not
physical proof.

## 3. Apples-to-apples Metal benchmark

The baseline and each mode use identical:

- device/context and compiled executable;
- tiny source-body-free fixture bytes/codec/geometry;
- prompt/token sequence and token count;
- head mode/full-logit/token-only setting;
- ledger/timing instrumentation setting;
- cold/warm definition and cache state.

Run each mode in randomized/interleaved order over enough warm iterations.
Report wall and GPU p50/p95, not one sample. Timing runs have ledger off;
separate counter runs have ledger on.

Report exact physical:

- command buffers, waits, encoders, dispatches, timestamps;
- D2H/H2D/shared CPU-read/shared CPU-write/blit/set-bytes categories;
- allocations, norm uploads, scalar updates, selected-ID/weight bytes;
- hit/miss and cold/rebuild separately.

## 4. Promotion kill criteria

For each candidate and combined:

- complete V2.1 must pass;
- zero forbidden wave/table-wave/replay/ICB calls;
- exact three ordinary gate/up/down matvec batches remain;
- steady waits and command buffers are no greater than baseline for the
  replaced work;
- physical bytes are reduced in the intended category without an unexplained
  increase elsewhere;
- wall p50 and p95 do not regress beyond frozen noise tolerance;
- combined must show a statistically and physically explained improvement, or
  the non-improving candidate is removed from the combined path.

Do not label a slower/default-off feature “hot-path closure.” It may remain
experimental only under a clearly non-promotable diagnostic flag and must not
be integrated into the TG acceleration chain.

## 5. Transaction and call-path probes

Add actual entry probes at the functions/APIs, not comments or inferred flags.
With C1/C2/C3 on, expert-wave/ICB/table-hit off, assert zero calls to wave,
table-wave, wave scratch, expert replay, and final-head replay. Assert exact
ordinary gate/up/down batch count and the actual k-ID+k-weight readback bytes.

Inject validation, encode, commit, wait, readback, binding, and rebuild failures.
Before mutation, state is byte-identical; after device/KV mutation, poison
refuses all later encoding until a verified reset.

Scalar-ring delayed completion must prove no overwrite using actual in-flight
fence/generation, not a counter increment after `commit_and_wait`.

## Required gates

Run on a real local Metal device:

- baseline, C1, C2, C3, combined V2.1 matrix;
- independent topology/counter run;
- randomized/interleaved micro-latency p50/p95 run;
- edge cases: signed zero, subnormal/FTZ, finite extremes, NaN/infinity refusal,
  alias, unsupported size/layout/codec, delayed completion, and injected
  failures;
- `git diff --check`, formatting, and targeted unit/integration suites.

Report exact files/hashes plus before/after physics. No product TPS or TG
milestone may be inferred from the tiny fixture.

No capable provider, real model body, TG/HIDE promotion, MOP action, or
authorization-fence transition is permitted.
