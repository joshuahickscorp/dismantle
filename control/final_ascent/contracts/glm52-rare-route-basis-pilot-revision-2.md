# Revision 2: close deployed-representation, provenance, and recovery gaps

Revision 1 remains mandatory. The implementation/revision task still must not
fetch, read, release, or modify any real source body. Reconcile every item below
before integration or real-source authorization.

## 1. Freeze the low-count split

The split is:

- `n_routes == 0`: fit and holdout are both empty;
- `1 <= n_routes < 32`: fit is empty and every routed row is holdout;
- `n_routes >= 32`: `n_hold = max(1, ceil(n_routes / 5))`, with the remaining
  rows in fit.

Fit and holdout are always disjoint. Never call a target-local constructor for
counts 0 or 1. The rank-32 and rank-128 constructibility boundaries remain 40
and 160 total routed rows respectively. Add exact boundary and disjointness
tests.

## 2. Freeze duplicate-row identity and deterministic basis orientation

Use a typed row identity containing capsule/member seal, layer, and flattened
C-order row index.

- shared hidden `H` fits deduplicate by layer-scoped contextual-row identity;
- shared down `Z` fits retain distinct `(peer_expert_id, row_identity)` pairs;
- peer order and input enumeration order cannot affect row witnesses or bases;
- canonicalize every basis-column sign by making its largest-magnitude element
  positive, with the lowest coordinate as the deterministic magnitude-tie
  break.

Preflight must freeze the numeric seed, split rule/order, peer order and cap,
counterfactual reserve size and selection, deduplication rule, SVD sign rule,
and little-endian float16 C-order deployment layout. Add permutation,
sign-perturbation, and repeated-run tests.

## 3. Score the deployed representation

All scientific verdicts use the actual serialized little-endian float16 basis
and coefficient matrices decoded back to float32. Do not score ideal float32
factors.

Add a fixture where the ideal float32 reconstruction clears a floor but the
serialized float16 round trip does not; the verdict must fail.

## 4. Define the dual-residual arm

For arm 4:

- build the rank-64 shared primary first;
- project target-local fit rows through that deployed primary;
- build the rank-32 target-local residual from the remaining error;
- orthogonalize every residual column against the shared primary and require
  actual residual numerical rank 32;
- apply this construction separately to hidden `H` and down `Z`;
- deploy primary and residual as two physical coefficient matrices and two
  physical basis identities, with their own tensor headers under the current
  v2 ABI.

Test primary-only versus primary-plus-residual reconstruction,
primary/residual orthogonality, rank deficiency, and the exact basis,
coefficient, and header charges.

## 5. Make basis identity collisions fatal

A repeated basis identity is a reference only when all of these match:

- kind, layer, and scope;
- width and rank;
- stored dtype and layout;
- SHA-256 of the deployed serialized basis bytes.

Any mismatch is a fatal collision; silent refcounting is forbidden.

Independently assert these whole-model values:

- coefficient bytes: `51447595008`;
- tensor headers: `15040256`;
- native bytes: `29816121344`;
- exact ceiling: `92282917708`;
- F1 basis bytes: `10887724800`;
- F1 total: `92166481408`;
- F1 unique basis identities: `19839`;
- F3 basis bytes: `722062080`;
- F3 total: `82000818688`;
- F3 unique basis identities: `459`;
- 150 shared-hidden identities fit and 151 fail;
- 4,977 shared hidden/down pairs fit and 4,978 fail.

All targets in the same layer must reference byte-identical shared hidden/down
objects. The real panel spans four layer identities `{18, 19, 31, 76}`.

## 6. Bind provenance and canonical typed hashes

Preflight and measurement must validate and bind:

- the route-census `receipt_sha256`;
- the rehydration receipt `seal_sha256`;
- the complete official tensor-index file SHA-256;
- source shard size/hash and the exact target-to-shard mapping;
- exactly 15 unique targets and complete gate/up/down triplets;
- exact tensor dtype, shape, offsets, finite values, row count, and C-order
  flattening;
- the sealed member hashes for both and only `pre_router_hidden` and
  `topk_indices`;
- measured top-k hashes/counts equal the census.

Reject stale seals, missing/duplicate/cross-shard tensors, wrong shapes,
out-of-bounds offsets, or reads of unrelated capsule members. A tracking ZIP
test must prove only the two named members are loaded.

Canonical JSON is sorted compact UTF-8 with `allow_nan=False` and the seal field
omitted. Keep typed hashes distinct:

- member hash: sealed raw C-contiguous bytes;
- row-set hash: dtype, shape, layer namespace, and ordered row indices;
- basis hash: deployed little-endian float16 bytes plus shape, rank, and kind.

Aggregation recomputes verdicts from score rows. A freshly resealed but
inconsistent `floor_pass=true` field cannot override a failing score.

## 7. Freeze negative-control and unobserved semantics

- zero-route diagnostics cannot affect any floor, median, candidate flag, or
  denominator, even when their score is 1.0;
- missing zero diagnostics fail completeness/integrity but never become
  promotion evidence;
- centered, output-side, all-row, and Gaussian controls cannot select an arm;
- only arm 3 may survive, and no other arm/control may rescue it;
- layer 78 `route_count=None` remains
  `UNOBSERVED_NOT_TESTED` and is never converted to zero;
- even perfect bounded-panel scores can set only the two bounded-pilot flags;
  every traversal, capability, research, HIDE, Odyssey, and MOP fence remains
  false.

Add mutation tests for every rule.

## 8. Close lifecycle recovery gaps

Acquisition must create exactly one complete copy by content and inode:
download to same-filesystem staging, verify, then atomic-rename into the final
regular path. A complete hash-named cache blob is a duplicate source body and
must be refused; metadata and incomplete files may remain.

Measurement:

- opens the shard read-only with `O_NOFOLLOW`;
- proves pre/post size, full hash, inode, and mode are unchanged;
- atomically publishes a partial and refuses overwrite unless byte-identical;
- leaves an integrity-valid scientific failure releasable.

Release:

- validates against an open file descriptor and inode;
- durably writes a sealed `PREPARED` intent before deletion;
- atomically renames the body to a unique release-pending name;
- repeats both successful-and-clean `lsof` and argv/process probes;
- unlinks only the validated inode using a trusted directory descriptor;
- durably publishes the sealed `COMPLETE` receipt afterward.

Missing, malformed, nonzero, or timed-out probes refuse release. Inject a
failure at every transition and prove deterministic safe recovery without
stranding an unaccounted body.

Aggregation binds exactly one partial and one complete release receipt to each
shard, rejects missing/duplicate/cross-shard/tampered/inconsistent artifacts,
and proves all three named bodies plus any complete cache duplicate are absent.

Preflight and selftest must trap network calls, subprocess downloaders, and any
open of `*.safetensors`, proving they are source-body-free.

## Required rerun

After revision, run the complete Revision 1 matrix plus:

- serialized-float16 scientific mutation tests;
- exact ledger and identity-collision tests;
- provenance/member-access mutation tests;
- negative-control/unobserved mutation tests;
- lifecycle duplicate-copy and injected-recovery tests.

Commit only the intended pilot/preflight deliverables. Exclude `.serena`, real
source files, lifecycle artifacts, caches, and temporary files.
