# GLM-5 rare-route pilot — Revision 10 adversarial ledger/identity closure

This controlling addendum follows Revision 9. Revise the existing candidate in
place. Preserve every prior exact-four-file, fake-only, source-body-free,
no-real-source, no-MOP, false-fence, atomic publication, lifecycle epoch,
authorized-unlink, independent-science, and immutable replay requirement.

Revision 8's exact blobs passed their 128 shipped tests but failed an immutable
adversarial audit. Every counterexample below must fail at both direct
classifier/helper and public acquire/release/cleanup entrypoints.

## 1. Terminal acquisition proves live state

A complete journal prefix is not sufficient. `fully_complete` and idempotent
resume require all of:

- exact current lifecycle epoch and operation ID;
- sealed acquisition terminal linked to the current prior terminal;
- exact final regular-file basename/device/inode/mode/size/SHA-256;
- exact sealed receipt identity;
- exactly one sealed `ACQUIRE_OK` ledger event with matching operation/epoch,
  receipt, source identity, and predecessor;
- no later release/cleanup terminal consuming that acquisition.

Mandatory regression: a sealed eight-state acquisition chain plus an
attacker-minimal receipt, but no final body and no matching ledger, must refuse.
It may never classify `fully_complete` or return the receipt.

## 2. Inode/generation continuity

Every acquisition transition binds the controlled staging and final inode
generation. Recovery from `PUBLICATION_AUTHORIZED` may not accept an unrelated
same-byte final file.

Mandatory regression: chain evidence binds inode `1`; an independently created
same-size/same-hash final inode must refuse rather than append
`FINAL_PUBLISHED` through `LEDGER_COMMITTED`.

Equivalent identity continuity applies to release pending/final bodies and HF
snapshot/blob entries. After `SNAPSHOT_UNLINK_AUTHORIZED`, replacing the
snapshot path with an unrelated entry must refuse; it must never unlink the
replacement.

## 3. Closed sealed lifecycle/ledger schemas

All lifecycle state, receipt, intent, and ledger objects have exact closed
versioned schemas. Missing `schema`/`kind`, unknown fields, aliases, wrong
types, duplicate keys, missing record terminator/frame, malformed seal,
unrecognized version/state/event, or noncanonical bytes refuse.

Every `ACQUIRE_OK`, `CACHE_CLEANUP_OK`, and `RELEASE_OK` is sealed and binds:

- lifecycle epoch and operation ID;
- exact event kind and monotonic sequence;
- predecessor state/ledger seal;
- artifact/source/body identity;
- receipt/intent/terminal seal;
- run/root identity.

Null/missing operation ID, unsealed row, wrong operation/epoch, stale
predecessor, or receipt mismatch never counts. Exactly one matching event is
required; duplicate attempts refuse or return the exact existing sealed row.

Mandatory regressions:

- unsealed `ACQUIRE_OK` with `operation_id=null` must not match an expected
  operation;
- an eight-state release chain with empty evidence plus an unsealed
  `RELEASE_OK` from another operation must not classify complete;
- malformed ledger parsing propagates refusal and never becomes an empty list.

## 4. Exact replay, including same state

Idempotent same-state replay compares the entire canonical sealed record:
schema/kind/state/epoch/operation/predecessor and every evidence field. Changed,
missing, extra, reordered, aliased, or cross-operation evidence refuses.

Mandatory regressions: release and cache-cleanup same-state replays with one
changed evidence field must refuse without overwriting the original record.

## 5. Acquisition restart closure

`STAGING_CREATED` and the new pre-staging intent are real recoverable states,
not explicit “not implemented” refusals. Recovery in a fresh process twice
either resumes the exact controlled inode safely or performs a sealed,
identity-bound quarantine/abort and starts a linked new operation. It never
silently downloads over, guesses, or leaves ambiguous orphan state.

Test all state cuts with real process death before and after file/dir `fsync`
and publication.

## 6. HF reference and replacement safety

Hold the cache lock across discovery through final absence verification.
Enumerate and revalidate every snapshot/reference/hardlink to the blob under
trusted `dir_fd` traversal. Cleanup proceeds only if the target ownership set is
exact and isolated.

Mandatory regressions:

- a second snapshot symlink/reference to the blob prevents deletion; cleanup
  must not report complete and must not leave a dangling reference;
- replacing the snapshot entry after unlink authorization refuses and preserves
  the replacement;
- a new reference introduced at every probe/unlink boundary is serialized or
  refuses;
- a hardlink count/reference mismatch refuses before any unlink.

## 7. Preflight matrix must execute attacks

Do not map many nominal case IDs to one constants/callability check. Every
canonical case ID invokes a distinct adversarial fixture and asserts the
specific refusal/state. The registry binds case ID, mutation/cut operation,
expected error/state, executed count, and pass result.

The Markdown and JSON both contain the complete Revision 10 matrix,
case/cut/mutation counts, exact code/test/input/predecessor hashes, immutable
candidate commit/tree, and identical semantic disposition.

The preflight must be path-independent. It may not seal absolute active
worktree paths. Materializing the exact candidate blobs at a different root and
running two fresh processes must produce byte-identical JSON/Markdown and the
same seal.

## Mandatory immutable audit gates

From an exact candidate commit/tree:

- execute the nine counterexamples above at helper and public entrypoints;
- execute every valid/malformed lifecycle prefix in fresh processes twice;
- verify exact collection/execution/pass totals and zero masks/skips;
- generate preflight at two unrelated absolute roots and compare bytes;
- run three independent resealed science mutations;
- run fake-only selftest, pack-v2 28, route-census 25, strict JSON/AST gates,
  `py_compile`, and `git diff --check`.

Report exact four blob/SHA identities, candidate commit/tree, adversarial case
IDs, and independent audit disposition.

The pilot remains unpromoted and fake-only. No real acquire/measure/release,
parent traversal, model-body access, MOP action, or authorization-fence
transition is permitted.
