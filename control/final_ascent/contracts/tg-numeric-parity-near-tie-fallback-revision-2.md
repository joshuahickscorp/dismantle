# Temporal Gravity Numeric Parity near-tie fallback — Revision 2

This is a controlling addendum to
`tg-numeric-parity-near-tie-fallback-revision-1.md`. Revise the existing
candidate in place. Keep every earlier default-off, source-body-free, no-MOP,
no-heavy-run, and false-fence requirement.

The implementation is not acceptable if it widens f32 scores, independently
repairs rounded router stages, writes only part of a decision, journals after
publication, or derives authority/build identities from caller strings.

## 1. Per-domain FP64 authority lease

Introduce an explicit run-scoped authority-input provider/lease. Existing
decoded f32 `dense`, `row`, `matvec`, downloaded score, group-sum, DSA-score,
or logit interfaces are not sufficient canonical inputs unless the governing
typed model format explicitly freezes that f32 value as the authority root.

The immutable lease binds for every input:

- artifact/tensor content hash and immutable buffer/address generation;
- original payload bytes or a stable byte range;
- codec/version, dtype, endian, shape, strides/offset;
- quantization scales/zero points/codebooks and exact f64 decode semantics;
- run, request, token, layer, domain, subgroup/boundary, and candidate-set
  identity.

Native, PQ, activation-aware, or other codecs without defined f64 authority
decode refuse. Retain and revalidate the lease through recomputation, durable
prepare, canonical replacement, downstream commit, and durable commit to
prevent TOCTOU.

## 2. Freeze each authority DAG

The FP64 fallback recomputes the complete load-bearing decision from the
authority roots, including:

- router: input activation, original gate payload, bias, codec/shape, fixed
  accumulation order, sigmoid, correction, coherent group strength and mask,
  final expert ordering, normalization, and routing scale;
- DSA: authoritative query/key inputs, scales, ReLU, head weighting, causal
  mask, and fixed reduction/order;
- head/token: final residual, RMSNorm input/weight/epsilon/order, original head
  payload, complete logit contraction, argmax and requested ordered top-k.

The router is one coherent transaction: canonical within-group choices produce
canonical group strengths; those produce canonical group top-k and mask;
canonical corrected scores produce final expert IDs; canonical uncorrected
scores produce the committed normalized/scaled weights. Never feed a rounded
fast-stage result into a later “FP64” repair.

Freeze multiply/add versus FMA contraction, transcendental implementation,
epsilon and normalization order, narrowing, masks, stable ascending-index ties,
signed zero, subnormal/FTZ behavior, and intended causal negative infinity.
Unintended NaN/infinity refuses.

## 3. Sound ordered-boundary coverage

For ordered top-k, evaluate:

- every internal adjacent pair `(rank 0,1)` through
  `(rank k-2,rank k-1)`; and
- the admission pair `(rank k-1,rank k)` whenever `n > k`.

Trigger if any required pair is guarded. Apply this independently to router
within-group top-2, router group top-k, final expert top-k, DSA, requested token
top-k, and argmax `0/1`. Bind the ordered pair IDs, scores, margins,
thresholds, and trigger results in the journal.

Exercise `k=1`, `k=n`, singleton groups, exact ties, duplicates, signed zero,
subnormals, intended masks, and unintended nonfinite values. Complete candidate
scores are required unless a sealed conservative error bound proves omitted
candidates cannot enter. Token-only or diagnostic head top-k without the
complete required boundary refuses.

Guard calibration/error bounds bind domain, backend, codec, and device. A
possible authority inversion must not bypass fallback. Missing sound
calibration remains `calibration_missing` and cannot qualify coverage.

## 4. Explicit run/session and two-phase durable journal

Remove process-global policy, counters, environment-driven authority, and
shared receipt vectors from the production path. Use an explicitly owned
run/session handle containing policy, authority provider, counters, sequence,
sink, and poison state.

Decision IDs contain run, request, token, layer, domain, subgroup/boundary
ordinal, candidate-set hash, and monotonic sequence. Reusing a run ID or
destination fails closed.

Journal every enabled decision, including no-hit, as:

1. length/checksum/predecessor-framed `prepared` record plus file `fsync`;
2. canonical downstream replacement and device commit;
3. framed `committed` record plus file `fsync`;
4. `aborted` record on any failure where journaling remains available.

Use parent-directory `fsync` when creation/rename durability requires it.
Recovery validates frames and predecessor hashes, accepts only complete
committed transactions, and deterministically truncates or refuses a corrupt or
prepared-only tail. A prepared-only crash record is never counted as committed.

Terminal reconciliation binds ordered committed/aborted IDs, all counters and
domains, final predecessor hash, journal SHA-256, artifact/source/executable
identity, and final output/model-state identity.

Source identity is a canonical manifest of exact source/generated-source blobs
and relevant build inputs. Executable identity covers actual executable and
loaded library/Metal bytes, compiler/linker/SDK versions, target/features,
`Cargo.lock`, generated outputs, and build flags. Caller strings are not
identity evidence.

Two barrier-interleaved concurrent runs with different artifacts, policies,
and sinks must have disjoint IDs, counters, journal writes, policies, and
terminal records. Same-destination ownership refuses.

## 5. Atomic downstream transaction

Before any validation, expert dispatch, attention, trace publication, token
publication, stop handling, persistence, or output escape:

- router replaces expert IDs, normalized/scaled weights, execution slots,
  route trace, device/shared selection buffers, and descriptor/table lookup;
- DSA replaces every host/device/shared ordering used by attention;
- head replaces sample token, requested ordered top-k indices/values, trace,
  next-token state, and publication.

Inject failures at authority acquisition, recomputation, prepare-fsync,
replacement, device upload/commit, commit-fsync, and terminal seal. Restore the
complete pre-token state or poison the session. The rollback/poison boundary
includes residual, KV/DSA caches, sequence length, router/head buffers,
host/device traces, table/address generations, receipt state, and any already
mutated token state. Poison is checked before every later encode/mutation.

## 6. Real insertion-point evidence

Source-body-free tests must pass through the real host and resident insertion
points. Include a constructed case where original-input f64 recomputation
reverses the promoted-f32 decision.

Matrix: disabled, enabled/no-hit, forced-hit, refusal, injected receipt failure,
crash recovery, and threaded concurrency across host/device router, host/device
DSA, full-input head, and token-only head refusal.

Forced-hit timing includes authority acquisition, real device
synchronization/readback, full f64 recomputation, both journal fsyncs,
canonical replacement/re-upload, barriers, and downstream continuation. Report
component and end-to-end p50/p95, cold/warm journal and authority-cache phases,
randomized/interleaved mode order, warmups, iterations, D2H/H2D bytes,
synchronizations, allocations, fsync count/bytes, and recomputation work.

All artifacts bind machine, OS, device, exact source/build manifest, policy
seal, fixture hashes, mode order, and terminal journal hash. Results are
bounded insertion-point costs only, never model TPS or TG promotion.

No capable provider, TG milestone, real model run, HIDE promotion, MOP action,
or authorization-fence transition is permitted.
