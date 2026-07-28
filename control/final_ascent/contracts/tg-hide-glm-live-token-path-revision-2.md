# Temporal Gravity GLM live-token path into HIDE — Revision 2

Revise the existing candidate in place. Preserve every Revision 0/1 token
algebra, fail-closed, exact eligibility, ordered SSE, raw evidence, timing,
durability, no-real-model/no-MOP, and false-fence requirement.

The Revision 1 candidate is rejected. Its report admits:

- serve/HIDE “durable” logs are in-memory primitives;
- not every production turn path is bound;
- scheduler poison does not cover every prefix/copy/reuse index;
- full timing-gap reconciliation is not a qualification gate.

It also modified the governing
`control/final_ascent/contracts/tg-hide-glm-live-token-path.md`. The authority
contract is immutable and must be restored byte-exactly to Git blob
`b9d82a840a582b8729356fc1dd04f3846027ff6c` (SHA-256
`5e11f2f78b9a327955fde2b33bf3c9041063a8dfb0530522f42dbd0b42905427`).

## 1. Real durable serve lifecycle

Replace in-memory success authority with an on-disk, versioned, hash-chained
serve journal under an explicit configured root. Every frame has canonical
closed schema, length/checksum, monotonic sequence, predecessor hash,
run/request/slot epoch, and operation identity.

For a new request:

1. exclusively create/own the journal;
2. append canonical `START`, short-write-safe;
3. `fsync` journal and parent directory before engine execution;
4. append Token decision/publication evidence in exact order as needed;
5. append exactly one `SUCCESS`, `REFUSAL`, or `ABORT` terminal;
6. `fsync` journal before terminal SSE;
7. terminal SSE carries the exact persisted terminal frame/journal hash.

Use immutable per-frame/no-replace publication or framed recovery so a torn
tail cannot erase the last valid prefix. Duplicate/nonidentical terminal,
cross-run append, stale predecessor, missing fsync, or write failure refuses.
Disconnect still appends and fsyncs abort even if the client cannot receive it.

Startup/restart scans journals strictly. Open/torn/corrupt requests remain
incomplete or receive a linked durable abort; none becomes success.

## 2. Real durable HIDE lifecycle and host integration

Implement the equivalent on-disk HIDE generation journal:

1. durable `GENERATION_START` before provider dispatch;
2. validate ordered serve SSE and exact persisted serve terminal hash;
3. durable HIDE success/refusal/abort terminal;
4. only after HIDE terminal `fsync` may `run_turn_core` persist a successful
   assistant message/turn.

Integrate this ordering into the actual canonical HIDE host/turn path, not an
unused helper. Partial text, missing/malformed terminal, provider error,
disconnect, journal failure, or restart never persists as success.

HIDE does not reconstruct or zero-fill serve evidence. It persists and binds
the exact serve journal/terminal hash, run/request IDs, artifact/build,
prompt/policy, ordered token and emitted-text hashes.

Authorize the smallest necessary changes to
`crates/hide-backend/src/host.rs` and directly corresponding tests for this
actual integration.

## 3. Complete slot-epoch poison enforcement

One failed epoch is removed from every scheduler/driver structure that can
admit, ready, prefix-cache, copy, defer, retry, or reuse it. Bind poison to slot
ID plus epoch; a later epoch cannot inherit or clear it accidentally.

If KV/model state may have advanced and tokenizer, decision validation,
scheduler, SSE, cancellation/disconnect, journal, or terminal persistence
fails:

- poison the exact epoch before any later scheduling;
- refuse every later engine encode/mutation/copy;
- remove it from ready/deferred/prefix/copy/reuse indexes;
- require an explicit verified engine wipe/reset;
- issue a new epoch only after successful reset;
- reset failure stays poisoned.

Test each index/path and concurrent neighboring slots. One poison cannot
downgrade/corrupt the cohort.

## 4. Monotonic timing qualification

Raw execution records use one documented server monotonic clock and frozen
units/precision. Bind acceptance, queue enter/leave, prefill start/end, first
committed decision, each decode, each SSE enqueue/publish, journal terminal
fsync, and total wall time.

Define every interval as disjoint/overlapping and recompute:

`total_wall = named_disjoint_phases + named_gaps/overhead`

within a frozen tolerance. TTFT and inter-token decode samples reconcile.
HIDE transport/observation is separate clock-domain evidence. Missing,
negative, nonmonotonic, cross-request, or arithmetically inconsistent timing
refuses qualification; never substitute zero.

## 5. Full decision/transport closure

Keep and independently test:

- exact vocabulary width and finite full-logit row;
- range/epoch/position/state/config-bound trace token;
- deterministic lowest-index ties;
- exact `repetition_penalty == 1.0` and every policy disqualifier;
- exactly one terminal hash-chained SSE envelope;
- malformed/fragmented/duplicate/reordered/unknown/post-terminal refusal;
- EOS, stop bytes, max tokens, cancellation, refusal, error, tokenizer failure,
  and disconnect remain distinct.

No bare `[DONE]`, optional token/stat, zero-filled stats, or string terminal is
success authority.

## 6. File scope and cleanup

Do not modify authority contract files. Remove `.serena` from deliverables.
Revert unrelated `hawking-orch` and test-only initializer edits by providing
backward-compatible/default evidence plumbing where safe; if a change is
semantically indispensable, list and justify it explicitly.

`Cargo.toml`/`Cargo.lock` may change only if no existing workspace hashing/
journal dependency can satisfy the implementation. Prefer existing workspace
dependencies and report the necessity.

Expected substantive surfaces are the core engine/batch types, serve
HTTP/driver/scheduler, HIDE protocol/model-provider/host, and directly
corresponding tests.

## Required fresh-process tests

Use source-body-free fixtures and real subprocess restarts over persistent temp
directories:

- serve/HIDE crash before/after start file fsync, directory fsync, decision,
  Token, terminal append/fsync, terminal SSE, HIDE terminal fsync, and message
  persistence;
- torn/short/duplicate/cross-run/cross-epoch journal frames;
- receipt disk-full/permission/write/fsync failures;
- poison after every post-KV failure and every scheduler index;
- disconnect at admission/queue/prefill/post-decision/post-Token/terminal;
- exact timing boundary and arithmetic mutations;
- direct vs batch-1 multi-step fixture parity and all policy eligibility edges;
- actual local HTTP/SSE plus actual HIDE host turn path.

Tests assert exact paths were exercised. A pure helper, fake parser without the
real host, in-memory log, ignored model test, or health Boolean is not
qualification evidence.

Run targeted and affected Rust suites, fresh-process matrices, format/lint, and
`git diff --check`. Report exact files/hashes, journal schemas/paths, fsync and
recovery evidence, poison index coverage, timing reconciliation, fixture
limits, and unchanged false fences.

No capable provider, real model run, TG/HIDE milestone, MOP action, or
authorization transition is permitted.
