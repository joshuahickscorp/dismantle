# TG device-resident ordinary three-batch MLP — Revision 1

Revise the existing candidate in place. Preserve every base-contract
default-off, non-wave, source-body-free, no-real-model/no-MOP, live-Metal-kill,
and false-fence requirement.

Freeze the implementation base as
`ba2ca65b1765e833ec381b454ee1d68b48534656` plus this contract-only
revision. Do not silently rebase onto moving source.

## 1. Arithmetic authority is executable

Define bit-level expected behavior for every supported kernel/codec:

- gate/up/down multiply-add contraction policy;
- routed accumulation (host `r += out * weight`) versus Metal non-contracted
  multiply then add;
- exact `exp`/SiLU implementation and approximation allowance;
- f32 narrowing points;
- signed zero;
- subnormal/FTZ;
- finite overflow/underflow;
- NaN/infinity refusal.

Do not use Metal `fma` when the authority requires separate multiply/add.
Compile shader functions with the required contraction/fast-math policy and
bind that policy in the receipt.

Add frozen bit-pattern vectors for signed zero, smallest/largest subnormal,
normal boundaries, finite extremes, overflow/underflow, and cancellation.
V2.1 continuous scores do not replace bit assertions where signed zero or FTZ
is load-bearing.

Freeze separate authority construction/bounds for residuals, per-expert
outputs, route-weighted/shared combine, DSA/router decisions, and logits/token
decisions. Reject nonfinite candidates before any metric comparison; NaN may
not pass via false comparisons.

Near-tie canonical decision qualification remains blocked until the true FP64
authority lane lands. The MLP fixture may prove continuous math only; it cannot
qualify router/DSA/head near ties by proxy.

## 2. Prepared concrete GPU plan

Add the candidate device-output API on concrete `GpuWeightCache`/Metal
structures, not the platform-neutral `WeightAccess::matvec_batch` interface
that necessarily returns host vectors.

The prepared plan binds every gate/up/down tensor before token mutation:

- artifact/tensor/payload hash;
- codec, dtype, shape, range and alignment;
- GPU address and address/storage generation;
- expert/row/call order and route weight;
- destination arena range and alias proof;
- pipeline identity and arithmetic policy;
- cloned resource lease and command/fence generation.

Preflight the complete token's supported MLP triplets and layouts before the
first layer/KV mutation. A route-dependent unsupported expert discovered after
earlier mutation cannot take an “exact pre-token fallback.” It poisons the
token/session unless the entire token was metadata-preflighted before mutation.

Initially qualify only explicitly proved codecs, preferably direct-u8 PQ and
native BF16. Activation-aware remains typed refusal until its latent-buffer
lifetime, two-dispatch projection, and authority math are independently covered.

## 3. Abortable command ownership

Current `TokenCommandBuffer` auto-commits partially encoded work on `Drop`.
That behavior is incompatible with fail-before-mutation. Add a versioned
abortable owner or explicit state machine:

- `Prepared` before encode;
- `Encoding`;
- explicit `Committed`;
- explicit `CompletedSuccess` or `CompletedFailure`;
- explicit `AbortedBeforeCommit`.

Dropping `Prepared`/`Encoding` must discard/refuse rather than auto-commit.
No partial encoded work may execute after a validation/receipt error.
Production legacy callers retain existing behavior only through their old
type/path; the candidate uses the abortable API exclusively.

After commit, inspect command-buffer completion/error status. A fence-owned
lease retains every buffer/pipeline/resource until confirmed success/failure.
Arena slots cannot be overwritten until the exact fence generation completes.

Inject drop/abort/commit/completion failure at every stage and prove no hidden
auto-commit. Test delayed completion and out-of-order fence observation.

## 4. Canonical state versus scratch

Canonical state includes residual, KV/DSA, sequence length, route/expert/token
decisions, traces, persistent bindings/generations, and receipt state.
Ephemeral candidate scratch need not be byte-restored after failure; instead
invalidate it with a new generation before reuse.

Before canonical mutation, abort leaves canonical state exact. After any
canonical or unrollbackable device mutation, poison the session. Poison is
checked before every later encode/copy/mutation. Verified reset:

- waits for/invalidates outstanding fences;
- discards/rebuilds candidate arena and address generations;
- clears or rebuilds canonical KV/DSA/sequence state as defined;
- issues a new session/token generation;
- remains poisoned if any step fails.

Add explicit poison/reset state to `ResidentSession`; the current reset of only
DSA/seq_len/waits is insufficient.

## 5. Physical boundary hooks

Instrument actual APIs, not caller declarations:

- command-buffer create/commit/complete/fail;
- waits/synchronization;
- encoders and dispatches by stage/buffer;
- timestamps observed/missing;
- mapped shared CPU reads/writes;
- D2H/H2D/blit/set-bytes;
- allocations/deallocations;
- arena/address generations and rebuilds.

Authorize `crates/hawking-core/src/cost_ledger.rs` if needed for versioned
physical receipt fields. Physical evidence binds instrumentation source blob,
executable, device, session/token, and ordered raw events. Caller-authored
`record_transfer` alone is logical evidence.

Run production topology with `HAWKING_TCB_TRACE=off`; GPU trace mode that splits
dispatches into extra command buffers is forbidden for topology/timing
qualification.

## 6. Chosen dependency topology

Choose and document one:

- one ordered command buffer with device finite-status propagation and guarded
  residual mutation; or
- two dependency boundaries (gate/up+activation, then down+combine+residual)
  with only a tiny explicitly accounted status transfer.

No host `Vec`, `read_f32`, mapped intermediate, or per-expert wait occurs on a
supported hit. Accumulate routed outputs using serialized non-contracted
operations in ascending expert ID and shared last at `1.0`.

The candidate must beat or match ordinary baseline physical waits/CBs. Fewer
logical waits without wall improvement is not acceptance.

## 7. Frozen live-Metal protocol

Freeze in source/schema:

- exact fixture blobs/build/source/device/OS identity;
- modes and paired randomized/interleaved order;
- deterministic seed;
- at least 20 warmups and 200 measured iterations per mode unless a stronger
  statistically justified fixed count is frozen;
- cold/warm boundaries;
- timestamp coverage requirement;
- thermal/power guard or typed unavailable limitation;
- nearest-rank p50/p95 convention;
- paired relative wall/GPU noise tolerance no greater than 2% unless calibrated
  in advance.

Timing has ledger off; counter run has ledger on. Report individual and combined
raw samples/hashes. Any parity failure, timestamp insufficiency, topology
regression, or p50/p95 regression kills the candidate.

## Authorized surfaces

Smallest necessary subset:

- `crates/hawking-core/src/gravity_glm.rs`;
- `crates/hawking-core/src/gravity_glm_resident.rs`;
- `crates/hawking-core/src/metal/mod.rs`;
- `crates/hawking-core/src/cost_ledger.rs`;
- `crates/hawking-core/shaders/gravity_pq.metal`;
- `crates/hawking-core/src/numeric_parity.rs` only for generic nonfinite
  rejection or explicitly coordinated authority integration;
- directly corresponding source-body-free tests/fixtures.

Do not touch artifacts, HIDE, MOP, receipts representing real runs, production
defaults, or unrelated kernels.

Report exact files/hashes, command-owner API, supported/refused codecs,
arithmetic vectors, poison/reset proof, physical event schema, live Metal
parity/topology/p50/p95, negative results, and unchanged false fences.
