# Near-tie fallback Revision 1: real FP64 authority and durable evidence

The first implementation was frozen as immutable Git blobs:

- `numeric_parity.rs` blob `63f938c4d584a14bbd903ea1f9063e5a156311e2`
  (SHA-256 `a5a55e98c1062ba7fd13014eac41f98497d6f88c4b0c6304db1b552a159c3a52`);
- `gravity_glm.rs` blob `063258368e3cf759526d6b62e12ef98e7bcda090`
  (SHA-256 `59b614737ddac0da9c1777966304ed8118da5b4722708aa59d14e1969f3bcf9e`);
- `gravity_glm_resident.rs` blob
  `3c190823875db8842b8ec99b74163808c89338c4`
  (SHA-256 `f81aafcde0d403b65513c6f04b257667cbf70625a142542526b9af89c885e420`).

Its 21 near-tie, 3 top-k, and 12 preexisting numeric-parity tests pass.
Promotion nevertheless remains refused. Apply this cumulative correction in
the same isolated worktree. Do not read real weights, touch MOP, change
defaults/fences, or claim TG/TPS.

## 1. Canonical fallback recomputes the V2.1 authority

`promote_f32_to_f64_exact` is not the FP64 authority for a decision whose f32
scores were produced by rounded f32/Metal reductions. On a guard hit,
recompute the smallest decisive router, DSA, or head score slice from the
original typed inputs under the exact `NUMERIC_PARITY_V2_1.md` FP64 authority:

- original activations/queries;
- original weights/bias/keys/logit inputs and their immutable hashes;
- exact dtype widening;
- frozen left-to-right or otherwise authority-defined accumulation;
- exact sigmoid/correction/normalization where applicable;
- stable descending score and ascending index tie-break.

Bind the input slice identity, operation, accumulation rule, and authority
output hash into the decision receipt. Widening already-produced f32 scores may
remain a diagnostic but cannot be named or accepted as canonical fallback.

If original authority inputs are unavailable at a wired boundary, an enabled
policy must refuse that path before any decision or downstream mutation.

## 2. Guard every decision-changing boundary

For ordered top-k, inspect:

- every adjacency inside the committed ordered top-k;
- the k/k+1 admission boundary;
- router within-group top-2;
- router group top-k;
- final expert top-k;
- DSA k/k+1 and ordered ranks;
- token argmax and every requested token top-k adjacency.

The fast path must expose all required candidates. A single margin between
`s[k-1]` and `s[k]` cannot qualify ordered top-k. Bind all observed margins and
derived thresholds.

Token-only head without the full candidates/original FP64 authority inputs is
not covered. When the policy is enabled, either obtain the complete decisive
head inputs/candidates and recompute the authority or refuse token-only head
before publication. A diagnostic partial top-k is not qualification.

## 3. Durable per-run receipt carrier

An in-memory process-global `Mutex<Vec<NearTieDecisionReceipt>>` is not durable
evidence. Replace it as the authority with an explicit per-run/session receipt
sink that:

- is created with immutable run ID, artifact/index hash, build identity,
  policy seal, backend/device, and output destination;
- appends canonical duplicate-key-free records through an atomic, fsynced,
  predecessor-sealed journal or an equivalently durable caller-owned receipt
  transaction;
- fails before decision commit when the sink is absent, unwritable, stale,
  poisoned, duplicated, or cross-run;
- publishes a sealed terminal reconciliation record only after all decisions
  and counters reconcile;
- recovers or refuses deterministically after a crash/short write;
- keeps diagnostics/test collectors explicitly non-authoritative.

The runtime measurement receipt binds the exact journal/file hash, semantic
seal, terminal record, expected decision IDs, and counts. Missing or
nonterminal evidence refuses qualification.

Process-global policy/counters are insufficient for concurrent requests.
Scope policy, sequence, counters, receipts, and reset lifecycle to one explicit
run/session. Prove two concurrent runs cannot mix IDs, counters, receipts,
policies, or sinks.

## 4. Real source and build identity

`implementation_source_hash` must bind the exact immutable source blobs or
source-tree manifest used to build the running implementation. It may not hash
a hand-written label naming functions.

`executable_build_hash` binds the exact executable/library bytes plus compiler,
target, features, and build inputs. A synthesized constant or package-version
string is insufficient.

At runtime, compare receipt identities to the loaded executable/source
manifest. Missing, placeholder, stale, all-zero, or attacker-supplied identities
refuse.

## 5. Downstream transaction and fail-closed coverage

The committed canonical decision must replace every expert ID, expert weight,
execution slot, DSA rank, head top-k, sample token, trace, table lookup, and
downstream consumer before dispatch/attention/sampling/state mutation.

Define the transaction boundary. Receipt failure, authority-input failure,
counter failure, or downstream replacement failure leaves pre-decision model
state unchanged or poisons/resets the session before reuse. No fast decision
may leak into state when canonical commit refuses.

If any wired load-bearing domain lacks complete authority recomputation,
durable receipt support, or downstream replacement, an enabled policy refuses
the whole qualification run. Do not report partial wired-domain coverage as
promotion-ready.

## 6. Tests and insertion-point benchmark

Retain all first-round tests and add:

- a constructed case where promoted f32 scores choose a different result than
  recomputation from original FP64 inputs;
- every internal ordered-top-k adjacency and k/k+1 guard;
- missing/or altered original input, dtype, weight hash, accumulation rule, and
  authority output;
- no-authoritative-sink, unwritable sink, short write, crash, stale/cross-run
  sink, duplicate decision ID, and nonterminal journal;
- two interleaved concurrent runs with different policies/artifacts;
- actual source-tree and executable identity mismatch;
- receipt emission failure before downstream mutation;
- every downstream consumer receiving the canonical rather than fast decision;
- token-only head refusal when complete authority inputs are unavailable;
- exact default-off identity and no authoritative receipt claim when disabled.

Benchmark disabled, enabled/no-hit, and forced-hit at the real router, DSA, and
head insertion points. Include actual device readback/synchronization and FP64
recomputation costs. Bind machine/device/build/source/policy identities,
warmups, iterations, p50/p95, and machine-readable receipt. A helper-only host
vector benchmark remains diagnostic.

## Acceptance

Change only the originally authorized source/tests/benchmarks. Run targeted and
affected Rust tests, formatting on changed files, `git diff --check`, crash and
concurrency tests, and bounded insertion-point benchmarks.

Report exact file hashes/blobs, true authority inputs/order, durable journal
schema/seal, source/build binding, covered/refused domains, downstream
transaction proof, measured bounded costs, and remaining calibration.

No production enablement, capable provider, real model measurement, TG
milestone, MOP action, or fence change is authorized.
