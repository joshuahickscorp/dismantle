# Revision 6: close remaining lifecycle, recovery, and replay false positives

Revision 4B was independently audited from immutable Git blobs:

- pilot blob `bc2c567c70ff6a40810b0dc10caf87fde53e0748`
  (SHA-256 `320b26b4f3ee0255ad3684a5c857858eb8cd4b726fa2b610dbc639ac19db449d`);
- tests blob `09064608313ca41bbaf84323327eac3f60f4f2cf`
  (SHA-256 `e53cc251c7a321aafd2ad02dbb3dadcf9f377765d6e4b70ae7147a99d68573c3`);
- preflight JSON blob `2b4de8ffc5abd7b7fb17f9b0f1ab066b9f5282fc`
  (SHA-256 `91c9048a9d3e81737a575187d81e2c0bd7f03c9cd929b476daacae121c2922ea`);
- preflight Markdown blob `e703698d18d6c77f3a75770626748643f352239d`
  (SHA-256 `1ef8f3dbb9ccd79fd6b603ed3a76b188c4f55d96774d548a45a119da1fd722a1`).

The frozen suite passed 74 rare-route tests, 53 prior v2/census tests,
selftests, compilation, and deterministic preflight checks. The lifecycle is
nevertheless rejected: production-shaped mutations still pass with failed
process probes, unbounded peer retention, conflated capsule identities,
missing peers, unsafe acquisition publication, incomplete release evidence,
sparse aggregate inputs, and nonidentical Markdown replay.

Apply this lifecycle-only revision after Revision 5. Revision 5 remains the
science and provenance authority and is cumulative with this contract. Do not
weaken its authority bundle, evidence, rank, witness, scalar, or mutation
requirements. Do not run a real source command, read a real model body, touch a
real MOP, or claim real lifecycle readiness while implementing or testing this
revision.

## 1. Process probes require successful execution

Both the `lsof` and argv/`ps` probes must independently satisfy all of:

- the exact intended executable and arguments were invoked;
- the process completed before the frozen timeout;
- return code is exactly zero;
- stdout has the expected parse structure and complete required fields;
- stderr and parser diagnostics do not indicate truncation or failure;
- the parsed result is clean under the frozen exclusion rules.

Return code 1 is not an alternate clean result. Missing executable, any
nonzero code, timeout, signal, exception, malformed output, partial row, or
unknown process identity refuses release. A foreign process is never excluded
merely because its command line contains this pilot's name.

Tests must invoke the real default probe adapter while injecting subprocess
results. Independently test return codes 1 and 2, timeout, exception, empty and
malformed output, missing columns, a foreign command containing the tool name,
and a genuinely clean return-code-zero result. Supplying an already classified
`process_scan` object does not test this requirement.

## 2. The production reader is genuinely bounded

The safetensors reader must retain no collection of all eligible-peer arrays.
It may retain the target triplet, one peer gate/up/down triplet, fixed-size
accumulators, and the final bounded basis/statistics required by the frozen
algorithm. After each peer contributes, its arrays must become unreachable
before the next peer is decoded.

The production path must:

- derive the exact target and eligible-peer panel from the immutable authority;
- request every required tensor exactly once and no unlisted tensor;
- stream peer triplets in the authority-defined order;
- hash and validate each tensor before it contributes;
- reject a missing, duplicate, extra, reordered, or incomplete triplet;
- report measured peak live array bytes and the frozen memory ceiling;
- set any streaming-complete field only from reader instrumentation that proves
  the bound, never from reaching the end of a loop.

`derive_exact_panel()` failure, an empty peer set when peers are required,
missing authority, or any production derivation exception is fatal. It may not
fall back to an empty list, a fixture panel, receipt-defined peers, or reduced
evidence.

The gate must include a sufficiently large multi-peer fixture whose
instrumented live-byte high-water mark is bounded by the target triplet plus
one peer triplet and fixed accumulators. A mutation that retains every peer,
swallows panel derivation failure, omits one peer, or reports a false streaming
boolean must fail.

## 3. Safetensors and capsule parsing preserve typed identities

Parse the safetensors header with a duplicate-key-rejecting JSON decoder.
Duplicate keys at any object depth, including duplicate tensor names or
metadata keys, are malformed. Validate the full header and exact nonoverlapping
payload ranges before decoding the first tensor. Ordinary `json.loads` with
last-key-wins behavior is forbidden.

Revision 5's capsule identities are mandatory in the reader and lifecycle
binder. For each allowed `.npy` member, independently compute:

- SHA-256 of the exact extracted archive-member bytes, including the `.npy`
  header;
- SHA-256 of the decoded, dtype-validated, normalized C-contiguous array bytes.

These are distinct typed values. The census/sidecar digest documented as an
array-body digest binds only the normalized array bytes; it cannot be copied
into `member_bytes_sha256`. The capsule file hash, exact member name, raw
member hash, normalized hash, dtype, original and normalized shape, and row
count must all agree with the Revision 5 authority bundle.

Tests must include duplicate safetensors JSON keys, overlapping/aliased ranges,
a valid member whose raw and normalized digests differ, raw-only corruption,
normalized-value corruption, a copied digest in both fields, missing sidecar,
and a synthesized-member-hash fallback. Every mutation must fail before
measurement evidence is published.

## 4. Acquisition is a durable, no-replace state machine

Acquisition must use an explicit, sealed, replay-only journal with monotonic
states for staging created, download complete, body verified, publication
authorized, final published, invariant verified, receipt published, and ledger
committed. Every state binds its predecessor seal, exact trusted-root-relative
basenames, device/inode/mode/size/full hash, expected final hash, and intended
receipt/event identity. Fsync the state file and containing directory at every
durability boundary.

Create a unique same-filesystem staging regular file with
`O_CREAT|O_EXCL|O_NOFOLLOW` through a trusted directory file descriptor. Keep
that inode linked and controlled throughout download. A downloader adapter may
write through the open descriptor; if it must use a pathname, verify after
every call that the pathname still names the original inode. Never unlink the
exclusive placeholder before copying or moving downloaded content into place.

Publication must be one atomic no-replace rename operation, such as an
available platform `RENAME_NOREPLACE`/`RENAME_EXCL` primitive, followed by a
directory fsync. Existence-check-then-`os.replace`, or hardlink-then-unlink with
a durable interval containing two pathnames, is not publication. Fail closed
when a genuine no-replace primitive is unavailable.

For a normal Hugging Face snapshot/cache result, either stream into the
controlled staging inode or journal and recover every exact cache-reference,
blob, staging, and final transition. Moving a cache blob before durably
handling its snapshot reference is forbidden. A crash or retry may never leave
a dangling snapshot reference, silently consume the only cache body, or create
a second complete pathname.

After publication, scan the complete pilot root without following directory
symlinks. Use `lstat` on every entry regardless of suffix or filename. Count as
a duplicate every regular file with the sealed complete content, same-inode
hardlink, or symlink resolving to such a body, including arbitrary-name
symlinks. Record exact relative paths in the proof. Dangling symlinks and
unclassifiable entries in lifecycle locations refuse.

On a post-publication invariant failure, journal and atomically roll a verified
final body back to its controlled staging name. Quarantine only the exact
staging inode whose bytes mismatch the sealed source. Never move a verified
final body into mismatch quarantine. Recovery must deterministically resume or
roll back from every journal state without recursive cleanup.

Attack tests must cut execution after every write, file fsync, rename, and
directory fsync, then retry from a fresh process. They must cover arbitrary-name
symlink and hardlink duplicates, root/intermediate/lock symlinks,
cross-filesystem staging, publication collision, competing publisher,
short/interrupted download, HF snapshot crash points, post-publication
invariant failure, quarantine scope, and poisoned retry. At every cut, assert
the exact number and paths of complete bodies and verify true peak allocation.

## 5. Release records every irreversible transition

Release must use a sealed, replay-only state chain with at least:

1. `PREPARED_FINAL`;
2. `PENDING_RENAMED`;
3. `PROBES_COMPLETE`;
4. `UNLINK_AUTHORIZED`;
5. `UNLINKED`;
6. `COMPLETE_INTENT`;
7. `COMPLETE_RECEIPT`;
8. `LEDGER_COMMITTED`.

The transition names are semantic, not optional labels. Each record binds its
schema, kind, status, predecessor seal, operation ID, and the complete evidence
available at that transition. Publish and fsync `UNLINK_AUTHORIZED` before
unlink; fsync the directory after unlink; then publish and fsync `UNLINKED`.
Absence of a body under `PREPARED_FINAL`, `PENDING_RENAMED`, or
`PROBES_COMPLETE` is unexplained deletion and refuses. The durable
`PROBES_COMPLETE` record may not be represented only by an in-memory pass.

`PREPARED_FINAL` must bind:

- source device, inode, mode, size, and full hash;
- exact safe final and pending basenames under a trusted root `dir_fd`;
- the operator confirmation and complete release gate;
- full Revision 5 partial path, file hash, semantic seal, and authority seal;
- intended COMPLETE intent, receipt, and ledger-event identities;
- an inventory of every preexisting non-source path.

The preservation inventory records exact relative path and type; device, inode,
mode, and size as applicable; a full hash for every regular file; exact link
text for every symlink; and exact child membership for directories, excluding
only declared lifecycle artifacts. It is captured before the original release,
carried through every state, and compared after recovery. Reconstructing a new
"before" inventory at recovery entry, checking only inode/mode, or using a
hardcoded preservation boolean is forbidden.

Hold the verified source descriptor through normal rename, probes, hash
revalidation, and unlink. Address entries only by trusted `dir_fd` and exact
basename. Before every transition, `lstat` the entry and require all bound
device/inode/mode/size/hash values. Recovery with a pending body must reopen it
without following symlinks and repeat the full checks; inode equality alone is
insufficient.

The two successful process-scan receipts are included in
`PROBES_COMPLETE` and every descendant. `COMPLETE_INTENT` additionally binds
the directory-fsynced absent final/pending state and preservation proof.
`COMPLETE_RECEIPT` binds every prior seal and all source, partial, confirmation,
gate, scan, basename, inventory, and disk-state evidence.
`LEDGER_COMMITTED` binds the exact COMPLETE receipt file hash and semantic seal,
not merely an operation or shard name.

Recovery starts from the longest single internally consistent predecessor
chain. A stale COMPLETE receipt or ledger event can never outrank a resident
body, a missing COMPLETE intent, a different operation ID, or an unbound
receipt seal. Conflicting chains, missing predecessors, body/intent
contradictions, unknown files, or ambiguous disk state refuse without deletion.
Recovery may republish only byte-identical intended artifacts and append only
the one fully bound missing ledger event.

Failure-injection tests must restart after every state publication, fsync,
rename, unlink, receipt publication, partial ledger write, and ledger fsync.
Mutations must cover a sparse COMPLETE receipt, stale ledger event, absent
COMPLETE intent, body still resident, missing scans, missing gate or
confirmation, wrong pending basename, wrong partial file hash/seal, changed
pending bytes with the same inode, changed preserved-file contents, missing
inventory path, and two conflicting state chains. All must refuse or recover
only along the one fully bound chain.

## 6. Aggregate requires the complete release chain

For every shard, production aggregate requires exactly one authority-bound:

- Revision 5-valid partial whose file hash and semantic seal both match;
- complete release state chain from `PREPARED_FINAL` through
  `LEDGER_COMMITTED`;
- COMPLETE intent, COMPLETE receipt, and ledger event with exact mutual
  bindings;
- pair of successful process-scan receipts;
- confirmation, release gate, pending basename, preservation proof, and final
  absent disk-state proof.

None of these fields may be absent, `None`, inferred from status alone, or
accepted because another boolean is true. Recompute every semantic seal,
re-hash every referenced file, follow every predecessor edge, require exactly
one operation ID, and independently verify the Revision 5 authority bundle.
Use `lstat` to prove the final and pending entries absent and the exact-path
duplicate scan to prove that no complete body exists under any other name.
Incomplete recovery is never aggregate-ready.

Existing aggregate JSON and Markdown are a coupled replay unit. Before writing
either, compute both intended byte strings and inspect both destinations
without following symlinks:

- if both exist and are byte-identical, return an idempotent replay;
- if either exists with nonidentical bytes, refuse without changing either;
- if neither exists, publish through a durable journal and atomic no-replace
  writes;
- if exactly one identical output exists after a recorded crash, recover only
  the other intended output from the same sealed journal.

The Markdown nonidentical branch may not fall through to replacement. Test all
four existence combinations, a nonidentical JSON, a nonidentical Markdown,
symlink destinations, a competing writer, and a crash at every coupled-output
transition. Assert that no preexisting byte changes on refusal.

## 7. Every immutable write is race-safe and replay-only

The common immutable writer must open a trusted parent directory, create a
unique `O_EXCL|O_NOFOLLOW` temporary regular file, complete a short-write-safe
write loop, fsync and close it, and publish with an atomic no-replace primitive
followed by directory fsync. On destination collision, open without following
symlinks and accept only byte-identical content.

An existence/read check followed by `os.replace` is a time-of-check/time-of-use
overwrite and is forbidden even when callers pass `allow_replace=False`.
There is no production option that replaces nonidentical intents, receipts,
partials, aggregate outputs, or journals. Ledger append must use sealed
operation/event identities, handle short and interrupted writes, fsync before
acknowledgement, recognize an exact existing event as replay, and refuse
malformed tails or same-ID/different-content events.

Race tests must replace the destination between check and publish, use a
destination symlink, inject short writes and interrupted fsyncs, and race two
different payloads. Exactly one payload may win; the loser must neither
overwrite nor report success.

## 8. Source-body-free traps cover all real escape paths

Tests must exercise interception through the APIs the production code actually
calls, including:

- builtins, `io`, `os`, and `pathlib` open/read variants;
- `socket.connect`, `connect_ex`, `send`, `sendall`, `sendto`, `sendmsg`, and
  preconnected TCP and UDP sockets;
- top-level and submodule Hugging Face download APIs, including current imports
  captured before trap installation and imports performed afterward;
- `subprocess.run`, `Popen`, shell execution, executable aliases, absolute
  executable paths, and downloader/network tools.

Prefer a deny-by-default subprocess boundary with the smallest explicit local
allowlist. Matching only selected executable basenames is insufficient.
Interception tests may install fake modules when an optional dependency is not
present; `ModuleNotFoundError` may not turn a required trap test into a pass or
skip. Each prohibited path must demonstrate that the trap—not an explicit
pre-call guard—stopped the attempted operation.

## Required adversarial gate

Start from complete production-shaped fake fixtures and call the actual
production validators, default process adapter, readers, recovery entrypoints,
aggregate, writer, and trap context. Helper-generated sparse receipts or
preclassified clean scans are not valid positive fixtures.

At minimum, retain all Revision 1–5 gates and independently mutate and reseal
fixtures to exercise every refusal in Sections 1–8. Tests must explicitly prove:

- `lsof` return code 1 refuses;
- multiple peers never coexist in the reader and panel derivation fails closed;
- raw member and normalized array hashes cannot substitute for one another;
- duplicate safetensors JSON keys refuse;
- arbitrary-name symlink/hardlink duplicates are found;
- acquisition recovers at every durable cut without two complete pathnames;
- a stale receipt/ledger cannot bypass a resident body or missing intent;
- every release field and predecessor binding is mandatory;
- aggregate rejects missing partial-file and COMPLETE-intent bindings;
- nonidentical Markdown replay cannot overwrite;
- every network, file, Hugging Face, and subprocess escape path is trapped;
- a competing immutable writer cannot replace the winner.

No skipped test counts as evidence. Run the full fake-only suite with zero
skips, selftest, two deterministic reload-verified preflights, `py_compile`,
the existing v2 and census suites/selftests, and `git diff --check`. Compare
the two preflight outputs byte-for-byte and independently verify their seals.

Change only the four pilot/preflight implementation deliverables when applying
this contract; this contract file itself is the specification, not an
additional implementation deliverable. Report integration acceptance
separately from real lifecycle authorization. This revision cannot authorize a
real source command, real acquisition, real measurement, real release, or MOP.
