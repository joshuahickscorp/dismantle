# GLM-5 rare-route pilot — Revision 11 science/witness closure

This controlling addendum follows Revisions 9 and 10. Revise the existing
candidate in place. Preserve all earlier exact-four-file, fake-only, atomic
lifecycle, immutable replay, no-real-source/no-MOP, and false-fence rules.

Revision 8 passed 128 shipped tests but accepted the production-shaped,
resealed scientific attacks below. Each is a mandatory direct partial,
direct aggregate-local, and public `aggregate()` regression.

## 1. Closed reader-evidence algebra

`reader_evidence` is one exact versioned closed object. It binds:

- shard/layer/run/operation and authority-bundle seal;
- exact ordered tensor inventory;
- exact ordered reader call inventory;
- tensor, capsule-member, streamed-peer, and precomputed-peer witness
  inventories;
- retain/release event sequence and per-event object/bytes/identity;
- recomputed live-byte trace, peak, final live bytes, and completion;
- predecessor/receipt seal plus its own canonical `reader_evidence_sha256`.

Partial and aggregate independently recompute the canonical reader-evidence
seal and every derived counter. Missing, false, extra, alias, reordered,
duplicated, unknown, or cross-operation fields/events refuse.

Final live bytes must be zero and every retain has exactly one matching release
in legal order. Residual live bytes are never accepted merely because an object
appears in an inventory.

Mandatory regressions:

- delete the sole release event while claiming final live bytes zero;
- delete or replace `reader_evidence_sha256`;
- replace `reader_audit` with `{}`;
- add an authority-unlisted tensor witness;
- empty capsule-member witnesses;
- garbage/unknown streamed-peer contribution objects.

Resealing the outer partial must not make any attack valid.

## 2. Exact authority inventories

Construct expected tensor, reader-call, capsule-member, peer, source-device,
and sidecar inventories exclusively from immutable authority inputs.

Validation requires exact set and canonical order equality. Every
authority-listed peer emits exactly one contribution record, including
zero-row, missing-weight, too-small, or nonconstructible disposition. Missing,
extra, duplicated, or reordered peers refuse; a bare `continue` is forbidden.

Mandatory regressions:

- authority peers `[2,3]` with only peer 2 evidence must refuse;
- precomputed contributions `[3,2]` against authority order `[2,3]` must refuse
  or be canonicalized before sealing and then prove exact canonical order;
- duplicate/unknown/zero-evidence peers must produce the defined disposition
  rather than disappear.

## 3. Precomputed Z derivation proof

Every precomputed Z contribution has mandatory, non-optional:

- shard, layer, peer/expert, capsule, member, and producer-operation identity;
- exact filtered-row witness, target-route-union seal, counterfactual seal, cap,
  and authority order;
- source tensor names/hashes/dtypes/shapes/offsets;
- Z dtype/shape/byte hash and canonical derivation seal;
- capsule/member and reader-evidence seals.

Recompute the filtered rows and Z relation from authority inputs at partial
validation and independently at aggregate. Fields are never accepted “when
present”; omission refuses. An arbitrary Z with only `expert_id`,
`fit_indices`, and bytes/hash must fail.

Ordinary streamed and precomputed paths serialize the same complete
contribution schema and use the same canonical row-result object, while
partial/aggregate recomputation remains independently implemented.

## 4. Independent sidecar and device authority

Aggregate-local sidecar validation independently checks, for every authority
member:

- capsule file/member identity;
- dtype, raw/normalized shapes, row count, offsets and byte ranges;
- raw bytes, normalized array, member, and sidecar seals;
- raw-vs-normalized equality/difference claim;
- tensor/capsule/reader cross-bindings.

Mandatory regressions: change dtype to `float16`, shapes/row count, capsule
identity, or `raw_and_normalized_equal`; partial and aggregate-local errors must
be distinct and public aggregate must contain both.

Source device identity is authority-bound, not merely before==after. Bind
device/volume/filesystem identity in the authority bundle and both observations.
Changing before and after together to attacker-selected `987654` must refuse at
partial and aggregate.

## 5. Public aggregate no-short-circuit proof

Tests invoke the actual public `aggregate()` with isolated authority/partial
fixtures. Replace partial validation with:

- `[]`; and
- unrelated nonempty errors.

In both cases, aggregate-local reader, sidecar, device, filtered-row, peer-Z,
capsule, and mutation checks all run, collect stable
`aggregate_independent_*` errors, and only then refuse. Calling helpers manually
is not coverage of public call ordering.

No score, constructibility summary, release gate/intent/receipt/ledger event,
or deletion occurs after either layer fails.

## 6. Adversarial overlap and path-independent preflight

Overlap tests must feed a resealed forbidden-row reinsertion into both partial
and public aggregate validators. Merely constructing an unused `attack` list or
calling `filtered_peer_rows` once is not a test.

Streamed/nonstreamed equality compares independently produced canonical row
objects, bases, Z hashes, contributions, scores, and final verdicts.

Preflight never seals absolute active-worktree/support/cache paths. Bind
root-independent logical authority IDs plus content hashes. Exact candidate
blobs materialized under two unrelated absolute roots must produce
byte-identical JSON/Markdown and the same semantic seal.

## 7. Mechanical coverage

Derive mutation IDs from exact reader, tensor, capsule, peer-contribution,
sidecar, device, and aggregate schemas. For every required field/call/event,
execute delete/wrong type/wrong value/duplicate/extra/reorder/alias/
cross-operation substitutions where meaningful.

Registry expected IDs exactly equal collected, executed, passed, and preflight
reported IDs. No group of nominal IDs may call only one constants/callability
check.

## Required immutable gates

From a fresh candidate commit/tree:

- run all executable attacks above at partial, aggregate-local, and public
  aggregate;
- run three independently resealed route/basis/science mutations;
- run exact collection/AST/mask gates and fresh-process lifecycle matrices;
- regenerate preflight under two unrelated roots and compare bytes/seals;
- run full rare-route suite, selftest, pack-v2 28, route-census 25,
  `py_compile`, strict JSON, and `git diff --check`.

Report exact four file/blob/SHA identities and separate lifecycle,
science/aggregate, and test-audit dispositions.

The pilot remains fake-only and unpromoted. No real source/model body, parent
traversal, MOP, or authorization-fence transition is permitted.
