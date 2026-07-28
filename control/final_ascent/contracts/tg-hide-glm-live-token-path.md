# Temporal Gravity GLM live-token path into HIDE

Repair the serving seam that prevents fast GLM token decisions from reaching
HIDE and forces every GLM request out of continuous batching. Work in an
isolated worktree with source-body-free fixtures. Do not read real weights,
run a heavy benchmark, touch MOP, flip `HIDE_KERNEL_TURN`, or claim a TG/HIDE
latency promotion.

## Current evidence to verify

Trace the current path:

`HIDE SubmitTurn -> hawking serve -> /v1/hawking/generate -> BatchDriver ->
single-stream fallback -> GravityEngine::generate -> GravityGlmGpu`.

Verify before editing:

- `GravityModel::prefill_slot` is intentionally unimplemented, so batch-1 GLM
  serve falls back to `Engine::generate`;
- the GPU lm-head token-only path returns `GlmTrace::sample_token`;
- the direct TPS runner consumes that token;
- `GravityEngine` discards the trace and invokes `Sampler` on returned logits;
- an empty logit slice can deterministically produce token zero;
- `healthz.fallback_present=false` describes fallback-model presence, not
  serving-path fallback.

If live source contradicts any premise, report the contradiction and implement
the smallest fix that satisfies the behavior below.

## 1. One exact token-output contract

Define one explicit output contract between `GravityGlmGpu` and
`GravityEngine`:

- full-logit mode returns complete logits and the engine samples them once;
- token-only mode returns no fake logits and a typed, validated committed token
  decision plus its trace;
- empty logits without a valid typed token decision is an error, never token
  zero;
- if both logits and a trace token are present for a parity/debug path, require
  greedy/full-logit argmax to equal the trace token before committing;
- the committed token is used consistently for output, next-token state, stop
  handling, SSE publication, accounting, and persistence;
- no downstream layer resamples or substitutes a token.

Preserve Numeric Parity V2.1. Bind the exact artifact, runtime flags, decision
mode, token, and trace identity. Token-only output remains unavailable when its
parity prerequisites are not met.

Add a hard invariant at the `Sampler` boundary: an empty candidate/logit input
cannot silently yield any token.

## 2. Real batch-1 Gravity prefill/decode path

Implement the smallest correct `GravityModel` batch-1 support required by
serve's default batch size:

- `prefill_slot` consumes the exact rendered/tokenized prompt and establishes
  the same model/KV/state as the direct engine path;
- the next decode operation returns the same greedy token as the direct
  `GravityEngine` path under identical artifact, flags, prompt, seed, and
  context;
- slot reset/cancel/error paths release or invalidate state deterministically;
- batch size greater than one remains explicitly unsupported unless fully
  implemented and tested;
- unsupported conditions refuse or take a separately receipted fallback before
  token publication.

The ordinary batch-1 serve path must no longer use the generic single-stream
fallback merely because `prefill_slot` is absent.

## 3. Honest serve/HIDE evidence

Expose machine-readable per-request evidence:

- exact artifact/index hash and runtime build identity;
- complete resolved GLM flags;
- `engine_path` with a canonical value for the batch-1 path;
- separate fallback-model and serving-path fallback counters/reasons;
- token decision mode (`full_logits` or `gpu_token_trace`);
- trace/policy seal and committed token trace hash;
- queue, prefill, TTFT, and inter-token decode durations separately;
- prompt/tokenizer/seed identity and output-token hash.

Do not call time before prefill finishes "decode-only." HIDE must receive and
persist the run/request evidence needed to prove which path produced the turn.
This task may wire the data path and use fixture tests; it cannot produce a
capable-model HIDE receipt without an approved artifact.

## Tests

Add source-body-free deterministic tests for:

- empty logits plus no trace refuses instead of returning token zero;
- full logits commit one greedy token;
- token-only trace commits that exact token;
- dual evidence agrees; disagreement refuses;
- committed token reaches stop handling, SSE/output, next-token state, and
  persistence unchanged;
- direct engine and batch-1 `prefill_slot`/decode parity for prefill plus
  several steps;
- exact prompt/history preservation;
- slot reset, cancellation, injected error, and repeated request isolation;
- `engine_path`, flags, hashes, timing phases, decision mode, and both fallback
  counters are present and internally consistent;
- the real local HTTP/SSE route uses batch-1 rather than the serving fallback;
- HIDE's model-provider fixture consumes and persists the returned evidence.

Tests must assert they exercised the intended path. A fake server, skipped
model test, or healthz fallback Boolean alone is not evidence.

## Authorized files

Change only the smallest necessary subset of:

- `crates/hawking-core/src/model/gravity_engine.rs`;
- directly corresponding hawking-core tests;
- `crates/hawking-serve/src/lib.rs`;
- `crates/hawking-serve/src/http.rs`;
- directly corresponding serve tests;
- HIDE model-provider/host evidence plumbing and directly corresponding tests,
  only if needed to carry the already-produced receipt.

Do not modify GLM kernels, model artifacts, capability/status/launch receipts,
authorization fences, or unrelated HIDE surfaces.

## Acceptance

Run targeted and affected Rust tests, formatting/lints available in the
repository, deterministic replay, and `git diff --check`. Report exact
files/hashes, token-contract semantics, engine paths exercised, fallback
counters, fixture limitations, and the remaining capable-artifact live gate.

State explicitly that no real model, TG milestone, HIDE production promotion,
or MOP action occurred and all fences remain false.
