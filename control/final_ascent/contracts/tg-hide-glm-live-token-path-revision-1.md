# Temporal Gravity GLM live-token path into HIDE — Revision 1

This is a controlling addendum to `tg-hide-glm-live-token-path.md`. Revise the
existing candidate in place. Preserve every earlier false fence and prohibition
on real weights, heavy benchmarks, MOP, or TG/HIDE promotion.

A typed enum alone is insufficient. Acceptance requires closed decision
identity, exact policy eligibility, pre-mutation validation, versioned ordered
transport, raw engine evidence, and durable lifecycle on both serve and HIDE.

## 1. Closed token-decision algebra

Use one versioned, non-optional production decision enum at every prefill and
decode boundary:

- `FullLogits` carries exactly the resolved vocabulary width, permitted finite
  values, and the complete post-policy row;
- `CommittedTraceToken` carries a range-checked token and a sealed GLM trace.

Empty vectors, `Option<u32>`, parallel raw-`u32` success paths, wrong-width
rows, all-invalid rows, and NaN-dependent decisions refuse. Freeze
deterministic argmax tie-breaking and invalid-float handling.

Every trace binds schema version, run/request ID, slot ID and slot epoch, phase,
absolute token position, previous KV/state seal, artifact/build/complete flag
seal, resolved policy seal, decision mode, and monotonic decision sequence.
Freeze canonical serialization and hashing. Prior-request, released-slot,
prior-epoch, wrong-position, or wrong-config traces refuse.

When both forms coexist, compare the trace token to deterministic argmax of the
exact post-policy logits before mutation. The validated decision commits once
and is the sole token for tokenizer decode, EOS/stop, next-token/KV state,
accounting, SSE, output hashing, and persistence.

## 2. State, slot epochs, and poison boundary

Batch-1 prefill consumes the exact rendered/tokenized prompt and complete
resolved policy, creates a fresh slot epoch, and returns its first decision
through the same typed contract as decode. `max_tokens=0` publishes no token.

Validate shapes, vocab bounds, request/slot/epoch/position, trace, tokenizer
output, evidence, and terminal prerequisites before mutating scheduler-visible
state where possible. If model/KV state has advanced and any later validation,
decode, scheduler, SSE, cancellation, disconnect, or receipt operation fails,
poison the slot epoch. Do not “roll back” a token while retaining advanced KV.

A poisoned slot is removed from ready/prefix/copy/reuse indexes and refuses all
later encode/mutation until an explicit engine wipe/reset succeeds and issues a
new epoch. Reset failure stays poisoned. Batch size greater than one refuses
unless separately implemented and qualified.

Direct and batch-1 paths must agree on prefill plus multiple decode decisions
under identical artifact, complete flags, prompt/template/tokenizer, policy,
seed, and context; tests assert the actual engine path.

## 3. Exact eligibility and fallback refusal

Freeze one predicate over the complete resolved request/runtime policy. At
minimum, batch trace-token mode requires finite greedy temperature, repetition
penalty exactly `1.0` (never `<=1.0`), canonical greedy `top_k`/`top_p`,
no grammar/JSON/logit constraint, no logprobs or other logit processor, and no
speculative/draft decision. Bind the resolved values and seed.

Non-eligible requests use exact typed full-logit semantics or refuse before
prefill/token publication. Immediate and deferred admission preserve `Err`
versus `Ok(None)`; refusals never enter the wait queue.

No production prefill/decode error invokes `Engine::generate`. Diagnostic
fallback requires operator authorization and explicit per-request opt-in,
occurs before the first token, never mixes paths, has a separate
receipt/counter/reason, and terminally disqualifies qualification.

## 4. Versioned ordered channel/SSE state machine

Replace every `Result<String,()>`, optional token/stats, string terminal, and
bare `[DONE]` success path with one versioned envelope containing run/request
IDs, monotonic sequence, predecessor hash, event hash, and exactly one of:

- `Token { id, text, decision_evidence }`;
- `Done { typed_reason, generation_receipt }`;
- `Error { typed_reason, partial_evidence }`.

Exactly one terminal is accepted. Tokens are unique and ordered; no event,
token, count, stats, or ordinary done marker follows terminal. Missing,
duplicate, reordered, malformed, truncated, unrecognized, or hash-invalid
frames refuse.

Distinguish EOS, stop string, max tokens, cancellation, refusal, execution
error, tokenizer error, and disconnect. None can be normalized to a zero-stat
success. Freeze EOS publication and stop-string byte semantics, including
cross-token matches. Token-ID hash, emitted-byte hash, HIDE buffer, counts, and
terminal reason reconcile.

Tokenizer failure or send/disconnect failure reaches the engine lifecycle,
records an abort even if the client is gone, and poisons state whenever KV may
have advanced.

## 5. Raw engine evidence

Create one raw per-request execution record at the engine boundary and seal the
corresponding evidence before Token/Done publication. It binds:

- exact artifact index and loaded shard-manifest hashes;
- runtime binary/build, tokenizer, and chat-template hashes;
- complete resolved GLM flags and engine path;
- request, prompt, policy, seed, slot epoch, and ordered decision identities;
- output-token and emitted-text hashes;
- separate fallback-model and serving-path fallback reasons/counters.

Requested settings, reconstructed environment, health fields, global counters,
and prior-request evidence are non-authoritative. HIDE validates and persists
the exact serve terminal-receipt hash; it does not reconstruct evidence.

## 6. Durable two-sided lifecycle

Serve durably appends request start before execution and exactly one linked
success/refusal/abort terminal receipt before terminal SSE. HIDE durably
appends generation start before dispatch and exactly one validated terminal
receipt before a successful assistant message/turn is persisted.

Both hash chains bind run ID, serve request ID, predecessor, artifact, build,
request/prompt, ordered output, raw execution, and terminal hashes. Appends are
atomic/idempotent; duplicate terminal attempts refuse. Receipt persistence
failure is terminal abort/refusal—success is never emitted first and made
durable later.

Crash/restart/replay leaves open requests/generations visibly incomplete or
adds a linked abort/refusal. Partial text is never persisted as a successful
turn. Test crashes before/after start, decision, token publication, terminal
persistence, terminal SSE, and HIDE message persistence.

## 7. Monotonic timing

Use one documented server monotonic clock, units, precision, and request/slot
epoch. Freeze acceptance; queue enter/leave; prefill start/end; first committed
decision TTFT; each engine decode; each SSE enqueue/publish; terminal
persistence; and total wall time. HIDE transport uses a separately labelled
clock domain.

Declare overlap/disjointness and reconcile samples plus explicitly named gaps
to total wall time. Invalid arithmetic refuses qualification. Deferred requests
retain the original identity and acceptance timestamp.

## Required source-body-free tests

In addition to the base contract, cover:

- wrong vocab width, out-of-range trace token, NaN/infinity, deterministic
  ties, stale slot epoch/position/state/config trace;
- exact `repetition_penalty == 1.0` boundary and every policy disqualifier;
- failures before and after KV advance, poison enforcement, wipe/reset failure;
- fragmented/malformed/duplicate/reordered/unknown-version SSE frames and
  predecessor-hash substitution;
- cancellation/disconnect at admission, queue, prefill, post-decision,
  post-Token, and terminal-persistence boundaries;
- EOS first, `max_tokens=0`, multi-token stop string, tokenizer failure after
  KV advance, receipt-write failure, and no later token/success;
- serve and HIDE crash recovery/replay with exact terminal-before-success
  ordering.

Report exact changed files/hashes, engine paths exercised, fallback counters,
durability fixtures, and capable-artifact limitations. No capable provider,
real model run, TG/HIDE milestone, MOP action, or fence transition is
authorized.
