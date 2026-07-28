# Revision 3: eliminate false-positive science and lifecycle gates

The stable Revision 1+2 implementation was independently audited at:

- pilot SHA-256
  `a9d056da143d0b5c0541c9e590cd0680c2d878006bd16022562aea4aa1068104`;
- tests SHA-256
  `1b08eda6f7c2520a2f8391a1dff1e2420c0f32b93feb477dbb462dfda154631a`;
- preflight JSON SHA-256
  `43ecc3f4653d5d72b9ab3b2add240586d7d0673b810dcc6222998fc626579e07`;
- preflight Markdown SHA-256
  `6c8f9201a836ca5f901c7ae2d41f64d6ec3aaf5799cd4ca72a5ef4859661da74`.

That snapshot is rejected. Its 41 passing tests are false-positive evidence for
the blockers below. Read the original contract plus Revisions 1 and 2
completely, then implement and test every requirement here.

This revision remains source-body-free. Do not run real acquire, measure,
release, or aggregate; do not read/fetch/delete a real source body; do not
touch MOP.

## 1. Fix canonical artifacts first

The generated preflight is invalid after a JSON round trip:

- recorded seal:
  `c8b82573e5847fb30063c515669ee7f4286103e3b512d39e3e76f45a9ae84258`;
- reload/recomputed seal:
  `d153546e0e6c00c5ee104328913928191c9963ed3f1315767c71ba8cb1dcf49f`;
- `verify_seal(reloaded_json) == false`.

Integer map keys become strings during JSON serialization. Canonicalize the
entire payload to JSON-compatible key/value types before sealing, or prohibit
non-string object keys. The in-memory object, encoded bytes, and reloaded
object must have the same seal.

Add tests that:

- write, reload, and verify the preflight;
- regenerate JSON and Markdown twice byte-identically;
- independently recompute the receipt from the encoded JSON;
- mutate every bound hash/seal and fail closed.

## 2. Score both deployed float16 payloads

For every input-side tensor, compute the ideal coefficient matrix
`L = W @ B`, then independently serialize both `B` and `L` as little-endian
float16 C-order, decode both to float32, and score
`W_hat = L_deployed @ B_deployed.T`.

Never stamp `scored_on_deployed_float16=true` unless both deployed byte
payloads were used. Publish separate typed basis and coefficient hashes,
shapes, dtype, layout, and physical identities.

For arm 4, serialize/decode primary and residual bases and coefficient matrices
as four separate physical payloads. Reconstruct by summing the two deployed
products. Recheck residual numerical rank after orthogonalization and after
float16 deployment. Missing shared peer-Z is `NOT_CONSTRUCTIBLE`; never
substitute a target-local primary.

Tests must include an ideal-float32 pass whose basis-only float16 version still
passes but whose basis-plus-coefficient float16 deployment fails.

## 3. Measure independent rank-64 and rank-128 arm-3 treatments

Build distinct shared hidden/down bases, identities, coefficient payloads, and
score rows at rank 64 and rank 128. Do not copy one score list into both gates.

The aggregate requires exactly:

- nine unique finite rank-64 promotion scores;
- nine unique finite rank-128 promotion scores;
- 27 unique finite rank-128 between/below scores;
- explicit requested/emitted ranks and non-capped evidence for every row.

Missing rank evidence fails. Rank-64 failure cannot be rescued by rank-128,
and vice versa. Zeros and all controls remain excluded.

Add rank-specific mutation tests, including a rank-64 failure with rank-128
success and the inverse.

## 4. Use typed row identities everywhere

The existing shared-fit path ignores `member_seal` and hashes bare integers.
Use `(capsule/member seal, layer, flattened C-order row index)` for hidden rows
and `(peer expert, typed row identity)` for down samples. Fit witnesses,
exclusion witnesses, counterfactual reserves, partials, and aggregate checks
must bind these typed identities. No bare-index witness may establish
provenance.

Use an exact largest-magnitude/lowest-index tie break for basis signs; no
tolerance-based near-tie.

## 5. Validate every provenance seal

Fail if the census receipt or rehydration receipt seal does not verify
canonically. Bind and verify the official tensor-index hash. Validate exact
source shard mapping, 15 unique targets, roles, route counts, complete triplets,
tensor offsets/shapes/dtypes, capsule/member seals, row counts, finiteness, and
top-k equality with the census.

Selectively read only the two required members from a normal multi-member
capsule. Unrelated archive members may exist but must never be opened; their
mere presence is not an error. Use a tracking reader test that fails if an
unrelated member is accessed.

## 6. Implement the real measurement path without executing it

`measure_shard` may remain disabled unless explicitly authorized, but the
authorized path itself must be implemented and fake-tested. It must:

- hold the lifecycle lock and rerun the complete source/partial state gate;
- component-`lstat` the trusted root;
- open the exact shard with `O_NOFOLLOW`;
- bind inode, mode, size, and full hash before and after measurement;
- parse only the named safetensors tensors for target and eligible peer
  triplets, validating offsets, dtype, shape, and finiteness;
- load only the two required capsule members and validate their seals;
- perform the two-rank deployment-honest measurement;
- atomically publish the exact sealed partial;
- refuse a nonidentical existing partial while accepting a byte-identical
  replay;
- prove the source inode/mode/size/hash was unchanged.

Fake tests must inject small tensor/capsule readers and exercise this exact
path without opening a real `*.safetensors` body.

## 7. Make acquisition leave exactly one pathname and one body

The normal Hugging Face cache result is often a symlink and always leaves a
complete cache body. Do not publish from a completed retained cache blob.
Download/stream directly to a unique same-filesystem staging regular file, or
move the verified cache body so no complete cache pathname remains.

After publication, scan all pilot-root descendants, including
`*.safetensors`, staging, hash-named blobs, hardlinks, and arbitrary names.
Exactly one complete pathname with the sealed size/hash may exist: the final
named body. A hardlink at another pathname is a duplicate even when its inode
matches.

Requirements:

- component-`lstat` existing parents before any `mkdir`;
- create the lock with `O_NOFOLLOW|O_EXCL` semantics or validate/open it
  without following a symlink;
- never recursively delete stale staging or incomplete files;
- prove the staging and final files are on the same filesystem;
- atomic rename only;
- rollback/quarantine safely if any post-publication invariant fails;
- account the true peak additional allocation, including any transient second
  copy;
- preserve every incomplete/cache sentinel.

Add same-inode hardlink, cross-filesystem, normal HF-symlink, staging duplicate,
lock symlink, root/intermediate symlink, publish-race, and rollback tests.

## 8. Make release a recoverable state machine

Both process probes must return zero, parse successfully, and be clean. Do not
exclude an arbitrary foreign process merely because its argv contains this
tool's name.

Keep the validated source file descriptor open through pending rename and
unlink, and compare its inode to the gate inode and the `dir_fd` entry at every
transition. Revalidate the exact partial immediately before deletion.

The durable `PREPARED` intent must bind:

- source size/hash/inode/mode and final/pending basenames;
- confirmation phrase;
- full release-gate receipt;
- exact partial seal and file hash;
- expected COMPLETE receipt path.

Fsync the directory after rename and unlink. Implement idempotent recovery for
crashes after PREPARED, pending rename, second probes, unlink, COMPLETE receipt
publication, and intent completion. Never strand an unaccounted body or
deletion.

Publish COMPLETE intent before, or atomically consistently with, the COMPLETE
release receipt so aggregate cannot accept a completed receipt with an
incomplete intent. Refuse overwrite unless byte-identical. Do not hardcode a
preservation proof (`or True` is forbidden).

## 9. Make partial/release/aggregate validation semantic

A minimally self-sealed document must never pass.

Partial validation must recompute and bind schema/kind, shard/source,
preflight/code/test/input hashes, exact five unique targets and roles/counts,
capsule/member seals, tensor witnesses, typed row witnesses, basis and
coefficient identities/hashes, requested/emitted ranks at both ranks,
constructibility, leakage exclusions, controls, finite score completeness,
and every false fence.

Release validation must require `status == COMPLETE` exactly and bind the
matching COMPLETE intent, source, partial file hash/seal, gate, pending
basename, and deleted inode. No legacy `status=None`.

Aggregate must:

- reject dangling symlinks using `lstat`, pending files, incomplete intents,
  duplicate cache/source bodies, and model-named duplicates;
- recompute all science verdicts from exact score rows and deployed evidence;
- treat absent rank/evidence fields as failure;
- reject duplicate target or score rows;
- cross-bind one partial, intent, and release per shard.

## 10. Make source-body-free and writes real guarantees

During preflight/selftest, trap:

- `builtins.open`, `io.open`, `os.open`, and `Path.open/read_bytes` for
  `*.safetensors`;
- socket/network access;
- Hugging Face download entry points, including late imports;
- subprocess launch of download/network tools.

Exercise each trap with a mutation test. Calling an explicit guard helper is
not evidence.

Replace the fixed `.tmp` JSON writer and non-durable lifecycle append. All
partial/acquire/aggregate/intent/release writes use unique `O_EXCL` staging,
file fsync, atomic publication, directory fsync, no symlink-follow, and
byte-identical replay rules.

## Required gate

Run:

- the expanded fake-only rare-route tests with no skips used as evidence;
- rare-route selftest;
- source-body-free preflight twice and reload/verify its seal;
- `py_compile`;
- existing v2 tests/selftest;
- route-census tests/selftest;
- `git diff --check`.

Change only the four intended pilot/preflight deliverables. Exclude `.serena`,
source files, caches, lifecycle artifacts, partials, receipts, and temporary
files. Report honest remaining gaps; do not claim real-lifecycle readiness if
any item above remains unimplemented or untested.
