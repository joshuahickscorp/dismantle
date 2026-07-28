# GLM-5.2 rare-route basis pilot — Revision 12 immutable closure

## Controlling scope

This is a cumulative source-body-free correction of the unfinished Revision-9
candidate. Authority is the base contract followed by Revisions 1 through 11.
Apply Revision 4A science before Revision 4B lifecycle; filename lexical order
is not semantic order. The stale `revision-bundle.md` is not complete
authority.

The only authorized Revision-9 predecessor is:

- base `HEAD`:
  `83bb2dcc2c30a0840afe19ff33ef253357e1b428`;
- pilot SHA-256:
  `98323fb2c5cad14a7a1c6155c9dc71de5768816dcad330c2a7d7f90cf7e21304`;
- test SHA-256:
  `f8d3b7d9a699719771c636c9e5e2fb2b2db7e81a13c134faeb5f3991bc67ba3e`;
- preflight JSON SHA-256:
  `ee5515b9c1f76dd815a804dce260b97cc06a4e9332b7d524e280b93e088fd1ae`;
- preflight Markdown SHA-256:
  `612a9381b181c2666d59604850e6991645b5a79594e043d5f648674e1c53dd0d`.

The starting index also contains two forbidden intent-to-add paths,
`.serena/.gitignore` and `.serena/project.yml`. Remove them. Refuse before
editing if the base or any of the four authorized starting hashes differs.

The result is not a real-source authorization. Do not acquire, map, inspect, or
rehydrate a model shard. Current disk admission is known to fail:
`free_bytes - sealed_shard_size >= 75,000,000,000` is false. Any generated
`next_supervising_command` must refuse real acquisition without mutation.

Revise only the existing four deliverables:

- `tools/condense/glm52_rare_route_basis_pilot.py`;
- `tools/condense/tests/test_glm52_rare_route_basis_pilot.py`;
- `GLM52_RARE_ROUTE_PILOT_PREFLIGHT.json`;
- `GLM52_RARE_ROUTE_PILOT_PREFLIGHT.md`.

Remove `.serena` additions. No other file is deliverable. Preserve fake-only,
no-MOP, false-fence, no-full-traversal, and exact three-shard/fifteen-target
scope.

## 1. Immutable and path-independent candidate

The current Revision-9 draft is refused because it is uncommitted, its detached
tree tests fail, and its receipts embed absolute mutable-worktree paths.

Produce an exact-four clean diff that can be committed by the supervisor and
audited from a fresh detached tree. Preflight generation and all identities
must be independent of absolute checkout root, temporary path, branch name,
ambient mutable worktree, and untracked files. Use canonical repository-relative
paths plus content/blob identities.

Do not claim an immutable pass from a mutable worktree. Return exact four blob
and SHA-256 identities, but label the candidate `AWAITING_IMMUTABLE_AUDIT`
until the supervisor creates a commit/tree and reruns every inherited gate from
fresh detached roots. Two unrelated-root regenerations must be byte-identical.

All supporting authority inputs must be read from the candidate tree or sealed
content identities, never from main by absolute path.

## 2. Real distinct lifecycle fixtures

Revision 9's 22/22 registry is not evidence: several case IDs alias the same
smoke helper, labels are unused, and a mutation mismatch executes `pass`.
Replace every nominal case with a distinct executed mutation/cut and expected
refusal or exact successor state. The registry must mechanically bind case ID,
input mutation or crash cut, invoked public/direct/aggregate entry point,
expected disposition/state, observed disposition/state, and evidence hash.

Implement all Revision-9/10 lifecycle attacks, including:

- valid pre-staging intent followed by a sealed eight-state acquire chain with
  no live body and/or no live ledger;
- fake terminal acquire with no body/ledger;
- unrelated same-byte final inode substitution;
- missing `STAGING_CREATED` resume;
- null or unsealed acquire operation;
- open-schema and extra-field records;
- same-state changed evidence;
- hardlink insertion between authorization and unlink;
- snapshot replacement after authorization;
- present and already-absent authorized-unlink recovery;
- restart at every acquire and release publication cut, then a second restart;
- release terminal reuse, stale terminal replay, and reacquire epoch confusion;
- lock owner death, replacement, reuse, and cross-epoch contamination.

Use one immutable per-state file, atomic trusted-`dir_fd` publication, exact
closed schemas, predecessor hashes, epoch identity, operation identity, and
live-object/ledger identity. Revision 9 authorized-unlink recovery overrides
Revision 8 blanket absence refusal. A terminal state is never proof of the
current filesystem: revalidate the live body, inode/generation, link count,
ledger entry, and authorization immediately before each mutation.

Do not interpret Revision 10's prose “nine counterexamples” as exactly nine;
execute every enumerated canonical case as its own fixture.

## 3. Closed science witness authority

Every reader-evidence, tensor/capsule/peer inventory, precomputed-Z record,
aggregate sidecar, and device receipt uses a closed schema with no extras. Bind
run, operation, shard, tensor/capsule/member/peer order, authority source,
dtype, shape, offsets, derivation algorithm/version, raw and normalized
relations, producer identity, own evidence SHA/seal, and exact current
live-object accounting.

Reader evidence must prove and recompute:

- every acquisition and release, with distinct operation IDs;
- `final_live_objects == 0` and `final_live_bytes == 0`;
- no deleted release event;
- no residual peer-Z/capsule/tensor object;
- exact reader evidence hash bound by partial and aggregate receipts.

Do not synthesize capsule aliases or accept generic smoke-schema records.
Deleting a release, omitting `final_live_bytes`, adding an extra field, emptying
reader/capsule witnesses, or deleting the top-level reader seal must refuse at
direct, aggregate-local, and public entry points.

Precomputed Z is accepted only with the full producer/source/derivation
witness. Proof fields are mandatory, never optional. Recompute exact filtered
rows, authority-ordered peers, counterfactual reserve, route union, caps,
member contributions, raw Z, normalized Z, shapes, and byte hashes. Arbitrary
Z, reordered/missing peers, missing weights, silently substituted empty
reserve, or retained peer Z must refuse.

Aggregate sidecars must bind exact dtype, shapes, offsets, member ordering,
raw/normalized relation, partial identities, and cross-seals. Device evidence
must bind the actual authority device; attacker-selected equal before/after
device IDs refuse.

The public `aggregate()` path must execute all independent partial, reader,
sidecar, device, filtered-row, peer-Z, capsule, and mutation checkers without
short-circuiting. Tests that manually call helpers, use vacuous
`errors == [] or isinstance(errors, list)` assertions, or merely inspect source
strings are invalid.

At minimum, actual candidate validators must refuse each of these already
executed counterexamples at every applicable entry point:

- deleted release while claiming zero live;
- extra reader field;
- missing `final_live_bytes`;
- missing top-level `reader_evidence_sha256`;
- empty reader audit;
- empty capsule-member witness;
- garbage streamed-peer contribution;
- arbitrary minimal precomputed Z with omitted proof fields;
- attacker-selected before/after device `987654`;
- sidecar member dtype changed to `float16`.

## 4. Frozen real scope and non-claims

The exact possible later real scope remains three shards, fifteen targets,
twelve scored routed targets, and three zero-route diagnostics. Revision 1
excludes panel routed rows; Revision 2 assigns all `n < 32` targets to holdout.
Even a valid aggregate authorizes only a wider bounded pilot. It does not close
missing globals, attention output, dense layers, layer 78, expert population,
capability, or full traversal.

Remain false:

- `RAMANUJAN_RESEARCH_AUTHORIZED`;
- `HIDE_KERNEL_TURN`;
- `ODYSSEY_LAUNCH_AUTHORIZED`;
- `full_parent_traversal_started`;
- `full_traversal_authorized`;
- `capable_artifact_claimed`;
- `MOP_touched`.

## 5. Required local gates

Before handoff:

- full rare-route suite with zero skips/xfails;
- every R1–R12 case registry with distinct executed evidence;
- fake-only selftest and preflight;
- pack-v2 and route-census suites;
- duplicate-key and closed-schema refusals;
- two fresh-process restarts for every lifecycle cut;
- two unrelated-root byte-identical preflight regenerations;
- `py_compile`, AST/static no-skip gate, and `git diff --check`;
- exact-four status with no `.serena`.

These are implementer gates only. Independent lifecycle, science, and test
audits from a supervisor-created immutable commit/tree are still mandatory.
