# Temporal Gravity → HIDE live-token path — Revision 3

Revise the Revision-2 candidate in place. Preserve every earlier typed-decision,
durable-terminal, poison/reset, monotonic-timing, source-body-free, default-off,
no-MOP, false-fence, and no-promotion requirement.

The only authorized predecessor has base
`35cbd3046aea21abae2a4e8f4046c3b823ec6ff5`. For the canonical manifest
formed by every changed/untracked non-`.serena` path as
`path<TAB>git-blob-id<LF>`, sorted under `LC_ALL=C`, the SHA-256 must be:

`1033ff18386bf2e35c77335851314c4c917b011188bf7ec120b8058c8d367fc1`.

Refuse before editing if it differs. Remove `.serena`.

The governing authority file is the current main authority:

- path `control/final_ascent/contracts/tg-hide-glm-live-token-path.md`;
- Git blob `13f18f0afb6278d0783d410b5d11514355a4da99`;
- SHA-256
  `5e11f2f78b9a327955fde2b33bf3c9041063a8dfb0530522f42dbd0b42905427`.

This resolves the inconsistent blob label in the prior addendum. The successor
must keep that file byte-exact.

Revision 2 is `REVISE`. Existing green targeted tests do not close the
executable counterexamples below.

## 1. Closed chained journals and strict replay

Serve and HIDE journals use closed, versioned schemas and frames binding:

- frame magic/version/kind/length;
- run/request/slot/epoch/model/executable/provider identities;
- sequence number and predecessor frame hash;
- canonical payload hash and frame hash;
- exactly one ordered terminal;
- final stream/output/message identity.

Replay validates every field, predecessor, sequence, identity, state
transition, payload/frame hash, and terminal uniqueness. It rejects:

- unknown schema/version or extra/missing fields;
- nonmonotonic sequence;
- cross-run/request/slot/epoch identities;
- incorrect predecessor or frame hash;
- duplicate or post-terminal tokens/frames;
- terminal without start;
- success after refusal/abort;
- torn/corrupt middle or tail.

CRC alone is not an identity/predecessor chain. Recovery may truncate only a
torn tail after validating the complete prefix, then must file-fsync and
directory-fsync the truncation before use. Semantic corruption refuses.

Close these exact reproduced cases:

- serve recovery accepted `schema=999`, nonmonotonic sequence,
  cross-identity, and post-terminal stream as success;
- HIDE recovery accepted bad predecessor/frame hashes and still allowed a
  message.

## 2. Stable durable ownership across fresh processes

Production journal roots must be explicit, stable, durable configuration bound
to the run/session. Per-process or per-call temporary roots containing PID,
counter, or current time cannot support recovery and refuse in production.

Use exclusive run/request ownership durable across processes. A fresh process
must discover the same incomplete journal, reconcile it to refusal/abort, and
must never start a second success chain at a different implicit root.

Every create/rename/truncate/terminal requires checked file fsync and parent
directory fsync. Never ignore an fsync error.

## 3. Serve terminal before transport success

The engine path, deferred stream task, scheduler, and HTTP/SSE layer share one
terminal state machine. A deferred engine error or channel close without a
durable success terminal becomes durable refusal/abort. It cannot emit bare
`[DONE]`, success stats, or normal channel completion.

SSE token frames bind the serve journal predecessor chain. The durable serve
terminal and its hash exist before the terminal SSE frame. No token or stats
frame follows a terminal.

Serve recovery may authorize success only from a fully valid terminal chain,
never from record-kind presence or internally plausible payloads.

## 4. HIDE terminal before `agent.message`

The real `run_turn_core` production path requires and validates the live serve
terminal hash/chain. It then durably writes the corresponding HIDE terminal,
fsyncs it, and only afterward may persist `agent.message`.

Remove `success:model_free_stub`, `model_free_local:*`, zero-fill, fabricated
completion hashes, and any path that persists a successful message without
serve evidence. Model-free unit stubs may test refusal, but may not create a
production-shaped success terminal or message.

HIDE parsing rejects forged `done`, post-terminal tokens, legacy success stats,
missing terminal hashes, and reconstructed receipts. Only the typed,
chain-bound serve terminal can authorize the HIDE terminal.

## 5. Complete slot-epoch poison

Poisoned `(slot_id, epoch)` state must be removed or marked unusable in every
scheduler/cache/reuse surface, including:

- ready, prefill, retry, deferred, copy-reuse, prefix-reuse indexes;
- slot maps and free/active queues;
- prefix cache and `SystemPromptKvBank`;
- KV/cache handles, sequence state, pending HTTP/SSE senders;
- generation/run ownership and receipt state.

The reproduced case where a poisoned slot remains reusable through
`SystemPromptKvBank` must refuse. Neighbor epochs/slots remain healthy. A
verified wipe/reset with a new epoch is required before reuse.

## 6. Monotonic timing and exact arithmetic

Use one monotonic clock domain and bind raw phase timestamps plus derived
durations. Enforce all causal relationships, including:

`enqueue <= publish <= durable transport terminal <= HIDE terminal <= message`.

Require nonnegative finite/bounded durations and exact
`total_wall = named_disjoint + gaps` within the frozen tolerance. Validate
queue, prefill, first decision, every decode sample, SSE enqueue/publish,
terminal fsync, and persistence phases.

The reproduced impossible fixture with `publish < enqueue` and total wall
`1e12` must refuse. Missing/zero phases never synthesize qualification.

## 7. Fresh-process executable matrix

Run real subprocess cases over explicit shared journal roots for:

- crash/write/fsync cut before/after every serve and HIDE state;
- deferred engine error, closed channel, client disconnect, scheduler error;
- duplicate/cross-identity/post-terminal frames;
- forged done/legacy stats/missing terminal;
- model-free/no-evidence refusal;
- message persistence only after both durable terminals;
- poison removal from every index/cache and neighbor isolation;
- impossible/missing/nonmonotonic timing.

Each canonical case must invoke the public production path and assert exact
terminal/message state. Source-string checks or private helper-only calls are
not evidence.

## 8. Exit

Return the exact changed-file manifest, Git blobs, SHA-256 identities, complete
test counts, and fresh-process matrix. Keep real weights, heavy benchmarks,
MOP, `HIDE_KERNEL_TURN`, TG/HIDE promotion, capable-provider status, and every
authorization transition false/default-off.
