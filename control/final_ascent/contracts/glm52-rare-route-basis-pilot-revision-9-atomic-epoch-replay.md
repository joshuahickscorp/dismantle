# GLM-5 2-shard rare-route pilot — Revision 9 atomic epoch/replay closure

Revise the frozen Revision 8 candidate in place. Preserve every Revision 8
scientific, lifecycle, exact-four-file, fake-only, no-real-source, no-MOP, and
false-fence requirement. The Revision 8 blobs are:

- pilot `75f9eb5f2521f9e463ed51d6e2c34080be1c3c35`;
- tests `357cdd51c4749e6c20a34abdaec4b1f4f15c01de`;
- preflight JSON `3ec6b2a5f37887fbdfe94d20821dc22c29a4dae6`;
- preflight Markdown `25838acc145223c4f19fa6b8b14cf33e7b73741d`.

Revision 8 is not promotable. Its report admits incomplete double-restart and
HF race coverage. The additions below are controlling.

## 1. Crash-atomic state publication

Do not append unframed JSONL for a load-bearing lifecycle state. A short write
or torn append cannot both be strictly rejected and automatically recovered.
Publish each state as an immutable regular `100644` record using trusted
`dir_fd` traversal:

1. create an exclusive temporary file under the trusted directory;
2. write the complete canonical bytes with a short-write loop;
3. `fsync` the file;
4. no-replace publish to the operation/epoch/state destination;
5. `fsync` the directory;
6. reopen through `dir_fd` with `O_NOFOLLOW`, verify bytes/seal/type/mode;
7. only then allow the next external mutation.

Identical replay may reuse the existing exact record. Nonidentical replay,
symlink, nonregular file, wrong mode, extra state, alternate operation, torn
temp, or destination collision refuses without overwrite. The valid prefix
remains readable after every injected short write/process death.

## 2. Lifecycle epochs and terminal linkage

Namespace acquisition, HF cleanup, release, and later reacquisition by a sealed
lifecycle epoch and operation ID. Link terminal seals:

`acquire(epoch N) -> optional cleanup(epoch N) -> release(epoch N) ->
acquire(epoch N+1)`.

The classifier selects exactly one latest valid epoch/prefix. Alternate
operation IDs, singleton legacy state names, stale terminals, orphan records,
or multiple candidate latest epochs refuse. An old terminal acquire receipt
must never be returned after its body was durably released. Reacquire creates a
new epoch with a new operation ID and predecessor equal to the release
terminal.

All three state machines share one trusted-root, cross-process lock whose
ownership and stale-owner recovery are validated. Test lock symlinks,
replacement, wrong owner, crash, and acquire/cleanup/release interleavings.

## 3. Acquisition intent before staging

Close the crash window before `STAGING_CREATED`. Durably publish an
identity-bound acquisition intent before creating the staging inode, or define
an equivalent atomic first state that creates and binds the inode without an
unjournaled window.

Recovery may remove/quarantine an orphan staging inode only when its exact
epoch, operation, basename, device, inode, mode, size/hash state, and durable
predecessor authorize that action. It never guesses from a filename.

Every valid acquisition prefix, including the pre-staging intent and all eight
existing states, is recovered in a fresh process twice. Test crash before and
after file/dir fsync and no-replace publication. Clean, partial, and terminal
replay must be deterministic.

## 4. Authorized-unlink recovery

Resolve the release unlink crash window explicitly:

- before durable `UNLINK_AUTHORIZED`, an absent body refuses;
- after a valid `UNLINK_AUTHORIZED`, the exact authorized source may be either
  present or absent because unlink may have completed;
- if present, revalidate exact identity then unlink once;
- if absent, revalidate the sealed authorization, pending/final path inventory,
  source identity, and absence of alternate complete bodies, `fsync` the
  directory, then publish `UNLINKED` without unlinking again.

Apply the equivalent rule to HF snapshot entry and blob cleanup. Absence alone
never proves success.

Hold the isolated HF cache lock continuously across snapshot-reference scan,
link-count/process checks, identity revalidation, unlink authorization,
unlink, directory `fsync`, and absence verification. A newly introduced
reference/hardlink between any two checks must refuse or be serialized.

All traversal/open/unlink operations use trusted `dir_fd` plus `O_NOFOLLOW`;
pathname `resolve()` or precheck-then-open is not authority.

Run fresh-process recovery twice for every release state and all seven HF
cleanup states, including before/after the authorized unlink, file `fsync`,
directory `fsync`, receipt, ledger, and terminal link.

## 5. Canonical filtered peer-Z closure

Freeze one exact row algorithm:

1. validate typed integer route rows, bounds, order, and uniqueness;
2. construct the complete target-route union;
3. subtract target-route union and counterfactual reserve;
4. preserve authority-defined row order;
5. cap only after filtering.

No post-cap `unique`, sorting, or dedup may change membership/order. Emit a
contribution record for every authority-listed peer, including zero-row or
too-small/nonconstructible peers; a bare `continue` is forbidden.

Ordinary and precomputed paths consume the same sealed
`filtered_peer_rows` result object. A precomputed block additionally binds and
proves its exact row witness, tensor triplet identities, capsule/member
identity, Z dtype/shape/bytes, producer operation, and recomputed Z relation.
Hashing attacker-supplied Z is not derivation proof.

Partial independently recomputes filtered rows and Z while inputs are
available. Aggregate independently reconstructs row/inventory/cap/Z bindings
from immutable authority evidence. Resealed panel-row or counterfactual-row
reinsertion yields distinct partial-local and aggregate-local errors.

## 6. Truly independent aggregate validation

Partial and aggregate semantic validators are separate implementations. They
may share strict parsing and canonical hashing only—not semantic helpers,
results, error lists, serialized pass Booleans, or call summaries.

Public `aggregate()` always constructs authority from immutable inputs, runs
the complete aggregate-local reader, sidecar, device, row, peer-Z, capsule,
route, and mutation checks even when partial validation already failed, collects
both complete error sets, and only then refuses.

Direct aggregate-checker and public-aggregate tests must produce the same
stable `aggregate_independent_*` errors when partial validation is replaced by
either an empty result or unrelated errors. No score, floor, constructibility
count, release intent/receipt/ledger event, or deletion occurs before all
required validation layers finish.

Reader evidence is a closed exact schema: missing, extra, alias, reordered,
duplicated, unrecognized, or cross-operation objects refuse independently at
both entrypoints.

## 7. Exact immutable four-file replay

The candidate diff path set is exactly:

- `tools/condense/glm52_rare_route_basis_pilot.py`;
- `tools/condense/tests/test_glm52_rare_route_basis_pilot.py`;
- `GLM52_RARE_ROUTE_PILOT_PREFLIGHT.json`;
- `GLM52_RARE_ROUTE_PILOT_PREFLIGHT.md`.

Require set equality and regular `100644` mode. No `.serena`, cache, rename,
symlink, submodule, or mode-only artifact is deliverable.

All tests/audits run on Git blobs materialized into fresh detached trees, never
mutable worktree bytes. Generate JSON/Markdown twice in separate fresh
processes with frozen Python/dependency/locale/timezone/`PYTHONHASHSEED` and
verify exact byte identity plus embedded code/test/input/predecessor hashes.

Preflight JSON and Markdown publication is one coupled no-replace transaction:
identical replay succeeds; a sealed intent recovers one missing side; any
nonidentical existing side or mismatched pair refuses without overwrite.

## 8. Mechanical mutation and restart registry

Derive expected lifecycle, schema-field, reader-call, aggregate-check, and
mutation IDs mechanically from the canonical schemas/state/evidence/call
inventories. Test deletion, wrong type/value, duplicate, extra, reorder, alias,
cross-operation substitution, malformed seal, and disk contradiction where
meaningful. Expected IDs exactly equal collected, executed, and passed IDs.

AST/static gates cover skip/skipif/xfail/importorskip, empty or filtered
parametrization, deselection, dead constant branches, unconditional truth
masks, swallowed exceptions, and early return/continue from mutation cases.
Require exact collected/executed totals and zero skip/xfail/xpass/deselect.

Fresh-process lifecycle cuts are real subprocess restarts over persistent
filesystem state, repeated twice, not module-global reset simulations.

## Acceptance and report

Run the complete rare-route suite, all fresh-process matrices, three independent
resealed scientific mutations, fake-only selftest, two-process preflight
regeneration, pack-v2 28, route-census 25, duplicate-key parsing, AST/static
gates, `py_compile`, and `git diff --check`.

Report exact four SHA-256/blob identities, candidate tree/commit, complete
case/cut/mutation counts, independent-audit disposition, and limitations.

The pilot remains fake-only and unpromoted. Real acquire/measure/release,
parent traversal, model bodies, MOP, and every authorization-fence transition
remain forbidden.
