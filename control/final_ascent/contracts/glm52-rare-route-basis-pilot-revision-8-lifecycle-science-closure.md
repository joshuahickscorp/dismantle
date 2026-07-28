# Revision 8: close lifecycle recovery and streamed-reader evidence

Revision 7 was frozen as immutable Git blobs and independently reproduced:

- pilot blob `c134c58c7f322c4668fe5da9929898832c559307`
  (SHA-256 `d712d20e07d193da2bbdaa96baa2d38328463ec9e4e95560800b2bf407bf9c1b`);
- tests blob `33d874165084f3c55fefa0ab085d0d647404ac95`
  (SHA-256 `5c5e95255eec6664b0955d61215b8be3c76a05abb633e9d77b880262ed8e05bf`);
- preflight JSON blob `265dcdeab0eb75534899fe9a1e46e7a94bb15858`
  (SHA-256 `6bc252ef75bf10e6b32c9c654a142f89b7c2e8b59aee578deda43918f0c261ac`);
- preflight Markdown blob `41d067c3bba9ac3fb3e50bbf05d06fa5650944ce`
  (SHA-256 `8f36cdb5dbe3f79c806df47cf9ea896d4c6c5e0a9cdf608beddae9e581510986`).

The exact Revision 7 suite passes 116 tests, its selftest passes, and two
source-body-free preflight regenerations are byte-identical with semantic seal
`5423fba61cdf6dc80a891037c8aff3562c86b6330e6d67a1fd016dab710301e7`.
The three Revision 5 metadata false positives now refuse. Those results do not
promote the pilot.

Frozen Revision 6/7 inspection still finds load-bearing gaps:

- a sparse COMPLETE receipt plus any shard-level `RELEASE_OK` row can classify
  as fully complete without the eight-state release chain;
- `_release_locked()` catches a corrupt release chain and resets it to an empty
  chain;
- acquisition writes a journal but has no strict loader, classifier, or
  restart path, so a crash after publication cannot resume;
- acquisition appends without validating the existing chain;
- Hugging Face snapshot and blob deletions are unjournaled destructive
  transitions;
- the streamed peer-Z producer can use unfiltered peer route rows before panel
  and counterfactual exclusions;
- reader, tensor, capsule, and streaming evidence are serialized but are not
  mandatory and independently recomputed at partial and aggregate validation;
- passing tests still contain conditional skips and masking expressions.

Apply this cumulative closure after Revision 7. Revisions 1–7 remain
authoritative. Do not read a real source body, run acquire/measure/release/
aggregate against real data, touch a real model or MOP, start a parent
traversal, or change any authorization fence while implementing this contract.

## 1. One strict parser and one authoritative chain model

Every acquisition, cache-cleanup, release, intent, receipt, and ledger parser
must:

- open through trusted non-symlink directory components;
- accept only a regular file at the exact trusted-root-relative basename;
- reject duplicate JSON keys at every depth, non-finite numbers, trailing
  bytes, truncated records, unknown schema/kind/status, and unknown fields;
- verify every record seal before using any field;
- require one nonempty operation ID, exact shard, exact ordered state, and exact
  predecessor seal;
- reject missing, skipped, duplicated, reordered, extra, or alternate-path
  states;
- distinguish an absent file from an unreadable, malformed, or non-regular
  file;
- fail closed on every contradiction instead of ignoring a later state or
  reconstructing authority from filesystem guesses.

Append functions first load and validate the complete existing chain. The next
state must be exactly the one legal successor. An existing state is idempotent
only when its complete canonical bytes and all evidence are identical. No
append may begin a second operation over a nonterminal or corrupt chain.

## 2. Acquisition has complete read-side recovery

Keep the ordered acquisition chain:

1. `STAGING_CREATED`;
2. `DOWNLOAD_COMPLETE`;
3. `BODY_VERIFIED`;
4. `PUBLICATION_AUTHORIZED`;
5. `FINAL_PUBLISHED`;
6. `INVARIANT_VERIFIED`;
7. `RECEIPT_PUBLISHED`;
8. `LEDGER_COMMITTED`.

Add strict production read-side entrypoints equivalent to
`_load_acquire_journal`, `_classify_acquire_durable_state`, and
`_resume_acquire`. `acquire()` must invoke them under the lifecycle lock before
checking for or creating a new staging body.

Every state carries the complete operation identity and the evidence needed to
resume from that exact prefix: trusted basenames; device, inode, mode, size,
and full hash; expected final identity; receipt identity; exact ledger-event
identity; and predecessor seal. Recompute pathname and open-descriptor
identities before every transition.

Recovery semantics are mandatory:

- after `STAGING_CREATED`, resume only through the same controlled inode;
- after `DOWNLOAD_COMPLETE`, recompute size and full hash before advancing;
- after `BODY_VERIFIED`, changed bytes, inode replacement, or a second complete
  pathname refuses;
- after `PUBLICATION_AUTHORIZED`, use atomic no-replace publication; an already
  present final is accepted only when the chain proves it is the exact
  already-published inode;
- after `FINAL_PUBLISHED`, require staging absence, exact final identity, and
  no complete cache, hardlink, or symlink duplicate;
- after `INVARIANT_VERIFIED`, publish only the deterministic intended receipt;
- after `RECEIPT_PUBLISHED`, append exactly one fully bound `ACQUIRE_OK` event;
- after `LEDGER_COMMITTED`, return the original receipt without download,
  publication, receipt rewrite, or duplicate ledger event.

A resident final body with a valid nonterminal chain is recovery input, not
automatically "another resident source body." A resident body with no matching
valid chain refuses. Recovery never quarantines the sole verified body merely
because a later durability step crashed.

Inject a fresh-process restart immediately before and after every state append,
file fsync, publication rename, directory fsync, receipt publication, ledger
write, and ledger fsync. Repeat each restart twice and prove byte-identical
terminal output and one ledger event.

## 3. Hugging Face cache cleanup is its own durable operation

Snapshot-reference and cache-blob deletion are not incidental cleanup. Add a
sealed cleanup chain, bound to the acquisition operation, with at least:

1. `CACHE_IDENTITY_BOUND`;
2. `STAGING_COPY_DURABLE`;
3. `SNAPSHOT_UNLINK_AUTHORIZED`;
4. `SNAPSHOT_UNLINKED`;
5. `BLOB_UNLINK_AUTHORIZED`;
6. `BLOB_UNLINKED`;
7. `CACHE_ABSENCE_VERIFIED`.

Each record binds exact cache-root-relative snapshot and blob basenames,
device, inode, mode, size, full hash, staging identity, acquisition operation
ID, and predecessor seal. Prove the staging copy durable and independently
verified before either unlink authorization.

Before blob unlink, prove no other snapshot reference, hardlink, process, or
out-of-scope owner requires it. Treat snapshot and blob unlink as separate
authorized transitions. Never recursively delete, delete by glob, follow a
symlinked component, cross the isolated pilot cache root, or infer safe deletion
from absence alone.

Restart before and after every state and both unlinks. Cover a missing snapshot
with present blob, dangling or retargeted snapshot, missing blob with present
snapshot, changed blob identity, second snapshot/hardlink race, busy or failed
unlink, and corrupt cleanup journal. Every failure preserves the verified
staging body and unrelated cache entries. `FINAL_PUBLISHED` cannot advance
until cleanup is terminal and the exact complete-body scan proves one pathname.
No unlink, fsync, parser, or identity error may be swallowed or converted into
an absent-cache success.

## 4. Release completion requires the complete chain

Keep the ordered release chain:

1. `PREPARED_FINAL`;
2. `PENDING_RENAMED`;
3. `PROBES_COMPLETE`;
4. `UNLINK_AUTHORIZED`;
5. `UNLINKED`;
6. `COMPLETE_INTENT`;
7. `COMPLETE_RECEIPT`;
8. `LEDGER_COMMITTED`.

Replace the legacy loose classifier with classification derived from the strict
chain loader plus current disk evidence. `fully_complete` is possible only
when all eight records validate and cross-bind:

- one operation ID and exact predecessor chain;
- source device, inode, mode, size, full hash, final and pending basenames;
- the original pre-release preservation inventory and its proof;
- exact partial bytes, file hash, semantic seal, and authority seal;
- exact confirmation and complete release gate;
- two independently successful process-probe receipts;
- COMPLETE intent file bytes and seal;
- COMPLETE receipt file bytes, file hash, and semantic seal;
- exactly one matching fully bound `RELEASE_OK` ledger event;
- absent final/pending body and an exact-path duplicate scan proving no complete
  body exists under any name.

A receipt status and shard-level ledger row are never sufficient. A stale
receipt or ledger row cannot outrank a resident body or missing chain state.
Ledger parsing errors refuse; they do not mean "no event."

Remove every catch that converts `_load_release_state_chain()` failure to
`[]`. Corrupt or conflicting chain evidence propagates as a refusal without
state publication or deletion.

Recovery resumes only from the longest valid exact prefix:

- after `UNLINKED`, never unlink again;
- after `COMPLETE_INTENT`, publish only the matching receipt and suffix states;
- after `COMPLETE_RECEIPT`, append only the exact missing ledger event and
  `LEDGER_COMMITTED`;
- a receipt without the matching chain and intent refuses;
- an absent body before durable `UNLINKED` refuses as unexplained deletion;
- a nonterminal chain with a pending body reopens it with `O_NOFOLLOW` and
  revalidates full identity, not inode alone.

Carry the original inventory from `PREPARED_FINAL`; never create a new
"before" inventory during recovery. Every state-specific evidence block is
mandatory. Existing intent, receipt, state, or ledger objects with nonidentical
bytes refuse rather than being replaced.

Inject restarts before and after every state publication, fsync, rename,
unlink, receipt publication, partial ledger write, and ledger fsync. Delete or
alter each chain record and each evidence field independently. Assert no new
state, overwrite, aggregate, verdict, or deletion beyond an already durable
`UNLINKED`.

## 5. Streamed peer-Z uses the same filtered science path

Create one shared `filtered_peer_rows` operation for streamed and non-streamed
execution. Before any peer Z computation, derive the exact target-route union,
target fit/holdout splits, sealed counterfactual reserve, and
`forbidden = target_routes union counterfactual_rows`.

For every authority-listed peer:

1. derive its route rows;
2. subtract `forbidden`;
3. apply the deterministic cap;
4. validate a typed, ordered, unique row witness;
5. only then compute `swiglu_intermediate` and Z.

Filtering occurs before capping and before any peer contribution. Too few
remaining rows produce authority-defined nonconstructible evidence; excluded
rows may not be restored.

Every streamed contribution binds shard, layer, peer expert, ordered tensor
names and hashes, capsule/member seal, target-route-union seal,
counterfactual-reserve seal, filtered-row witness, cap, Z shape, and Z byte
hash. A precomputed block suppresses the ordinary path only after all these
fields independently validate. Missing, malformed, duplicate, extra,
reordered, or authority-unlisted peer blocks refuse.

Streamed and non-streamed execution over identical inputs must produce
byte-identical typed peer witnesses, bases, deployed hashes, and scores.
Fixtures must contain at least one panel/peer overlap and at least one
counterfactual/peer overlap, assert that both overlaps exist, and prove both are
excluded. Reinserting either row and resealing attacker-controlled evidence
must refuse in partial and aggregate validation.

## 6. Reader evidence is canonical, mandatory, and independently aggregated

Use one canonical `reader_evidence` object. Legacy top-level aliases cannot
fill a missing field. It must contain:

- complete `reader_audit`;
- exact ordered `tensor_witnesses`;
- exact ordered `capsule_member_witnesses`;
- exact ordered streamed peer contribution witnesses;
- a typed retain/release event trace;
- derived peak live bytes, ceiling, streaming-complete result, and
  peer-tensors-streamed result.

The immutable authority carries the exact target/peer tensor inventory,
capsule inventory, sidecar/member identities, and order. Partial validation
requires exact equality and independently recomputes:

- each tensor name, projection, shard, layer, expert, dtype, shape, offsets,
  length, payload hash, finite result, bounds, and non-overlap;
- each capsule layer/hash, member name, raw member hash, normalized hash, dtype,
  original/normalized shape, row count, sidecar seal, and derived equality;
- each retain/release event, live-byte total, peak, ceiling, and final zero;
- that no two peer triplets coexist and each peer triplet is released before
  the next is retained;
- the derived streaming Booleans.

Unknown object IDs, double retain/release, negative sizes, missing release,
arithmetic inconsistency, missing/extra/reordered calls, or a serialized true
Boolean with an invalid trace refuses.

Aggregate implements a separate checker and repeats every computation from the
immutable authority and serialized evidence. It may not rely on an empty
partial error list, status, summary Boolean, or copied semantic seal. Bind the
canonical reader-evidence seal through partial, release intent, COMPLETE
receipt, and `LEDGER_COMMITTED`. All checks occur before any score, floor, or
promotion count.

The aggregate reader checker must consume raw canonical `reader_evidence` and
immutable authority directly. It may not call, wrap, delegate to, or consume
the result of the partial reader-evidence checker. Aggregate must execute its
independent checker even when `_validate_partial` has already found an error;
validation collects both error sets without short-circuiting and emits stable,
field-specific `aggregate_independent_*` evidence.

The same independence requirement applies to Revision 7 sidecar and source
identity evidence. Aggregate directly requires and recomputes every
`sidecar_seal_verified`, `sidecar_seal_sha256`, capsule/member binding, and
source-before/source-after device value from raw partial fields and immutable
authority. It may not rely on `_validate_partial` to reject a missing sidecar
seal or device.

Mutation tests delete, alter, duplicate, add, and reorder every call, tensor
field, capsule field, peer contribution, and retain/release event. They mutate
peak/current/ceiling and both streaming Booleans, retain two peers
simultaneously, and supply only legacy aliases. Each resealed mutation must
refuse independently at partial and aggregate entrypoints. Test each aggregate
checker directly, then through `aggregate()`. Also run with the partial checker
stubbed to return both empty and nonempty unrelated errors; the independent
aggregate checker must still execute and return the same field-specific
reader, sidecar, or device refusal.

## 7. Test and preflight evidence is unmaskable

Retain all prior regression and mutation cases. The rare-route test file must
contain zero `pytest.skip`, `skipif`, `xfail`, filtered parameter cases, dead
`if False` branches, unconditional `or True`, or equivalent assertion masking.
Source-body-free fixtures and immutable inputs are test prerequisites; their
absence fails the gate rather than skipping it.

Every mutation:

1. deep-copies one complete producer-shaped baseline;
2. asserts the intended field or condition changed;
3. recomputes only attacker-controlled seals;
4. calls the actual production partial and aggregate entrypoints;
5. requires stable field-specific refusal from both, including direct
   aggregate-local refusal rather than only propagated partial errors;
6. restores and revalidates the pristine baseline.

Do not mock validation success, rebuild authority from the mutation, silently
restore missing fields, or collapse multiple mutations behind one helper
assertion. Maintain an explicit canonical case registry with exact expected and
executed case IDs. Zero, missing, duplicated, or unexpected case IDs refuse
preflight publication.

The mandatory matrix includes:

- every prefix and before/after durability cut of both eight-state chains;
- every prefix and unlink cut of the seven-state cache-cleanup chain;
- every missing/corrupt/reordered/cross-operation state and evidence field;
- resident/staging/pending/receipt/ledger contradictions;
- the complete Revision 7 science/provenance mutation registry;
- peer-Z panel and counterfactual overlap mutations;
- every canonical reader/tensor/capsule/streaming mutation in Section 6;
- coupled aggregate JSON/Markdown crash recovery and nonidentical-output
  refusal.

Preflight JSON and Markdown report exact per-group expected, executed, passed,
and failed case counts and IDs. They bind the exact cumulative contracts,
pilot, tests, code inputs, immutable authority, and predecessor preflight.
Publication refuses unless all cases execute with zero skip, xfail, mask, or
failure.

## 8. Acceptance and authorized files

Only these deliverables may change:

- `tools/condense/glm52_rare_route_basis_pilot.py`;
- `tools/condense/tests/test_glm52_rare_route_basis_pilot.py`;
- `GLM52_RARE_ROUTE_PILOT_PREFLIGHT.json`;
- `GLM52_RARE_ROUTE_PILOT_PREFLIGHT.md`.

Required gates:

- strict AST parse with no duplicate top-level definition;
- duplicate-key-rejecting parse of every JSON artifact;
- full rare-route suite with the exact expected count, zero skip/xfail/mask;
- fake-only selftest;
- two byte-identical, seal-verified preflight regenerations;
- existing pack-v2 and route-census suites plus selftests;
- `py_compile`;
- `git diff --check`;
- independent science, lifecycle, and test audits of immutable Git blobs.

Implementation acceptance remains fake-only until all three immutable audits
accept. Even acceptance does not authorize a real source command. Real
acquire/measure/release/aggregate requires a separate controller promotion
receipt after integration.

The completion report must include exact file hashes, Git blobs, preflight
seal, every executed gate and case count, independent-audit disposition,
remaining uncertainty, and unchanged fences:

- `RAMANUJAN_RESEARCH_AUTHORIZED=false`;
- `HIDE_KERNEL_TURN=false`;
- `ODYSSEY_LAUNCH_AUTHORIZED=false`;
- full parent traversal false;
- MOP untouched.

This file is the cumulative Revision 8 contract. Implement it later in an
isolated worktree and report integration separately from real lifecycle or
science authorization.
