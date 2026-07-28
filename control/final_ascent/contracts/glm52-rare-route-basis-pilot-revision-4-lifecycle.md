# Revision 4B: implement the real reader and recoverable lifecycle

Apply this only after Revision 4A science/provenance is complete. This is a
focused lifecycle correction. Do not run a real source command or touch a real
source body or MOP.

## Authorized measurement must actually measure

The current authorized path is a fake-world wrapper: `tensor_reader` is never
called, capsule output is ignored, and every non-fake run raises. Replace it.

Implement an injected reader protocol that the production adapters and fake
tests both use:

- `tensor_reader(fd, tensor_name, expected_meta) -> ndarray + byte witness`;
- `capsule_reader(layer, allowed_member_names, expected_meta) -> arrays +
  member witnesses`.

The measurement engine must depend only on these protocols. A fake test passes
no preassembled `fake_world`; it supplies only the readers and must prove every
required tensor/member is requested exactly once, no unrelated item is read,
and wrong/missing data fails.

Implement the production safetensors adapter against the already-open
`O_NOFOLLOW` file descriptor. Parse the safetensors header and named byte
ranges without reopening an untrusted pathname. Validate header length, JSON,
unique names, dtype, shape, offset monotonicity/bounds, exact byte length, and
finite decoded values. Support the sealed GLM source dtype explicitly; fail on
unknown dtype. Stream peer triplets so memory stays bounded.

Implement the production capsule adapter from the sealed census inventory.
Read only the exact `pre_router_hidden.npy` and `topk_indices.npy` ZIP members;
unrelated members may exist but may not be opened. Validate capsule/member and
normalized-array hashes, dtype, shape, row count, C-order flattening, and
top-k/census equality.

Keep the source descriptor open throughout measurement. Compare mode, inode,
size, and full hash before and after. Publish a semantic Revision-4A partial
through byte-identical replay-only durable write.

## Acquisition must be possible and leave one complete pathname

Fix the normal Hugging Face snapshot case. Moving the blob while leaving its
snapshot symlink makes the symlink a duplicate and poisons future retries.

Use a downloader that writes directly into a unique same-filesystem
`O_EXCL|O_NOFOLLOW` staging regular file, or remove only the exact verified
snapshot reference as part of an atomic move protocol. After publication,
exactly one complete pathname with the sealed content may exist anywhere under
the pilot root: the named final body. Same-inode hardlinks and symlinks count as
duplicates.

Before mutation:

- component-`lstat` every existing parent before `mkdir`;
- create/open the lifecycle lock without following symlinks and verify its
  regular inode;
- do not recursively remove stale staging/incomplete/cache files.

Publication:

- prove staging/final `st_dev` match;
- use a no-replace atomic publication primitive or fail closed if unavailable;
- close the existence-check/rename race;
- roll back the final body on any post-publication invariant failure;
- quarantine only the exact mismatching staging body;
- make cross-filesystem or second-copy paths fail before creating another
  complete copy;
- record true peak allocation.

Add normal HF-symlink, lock/root/intermediate symlink, hardlink duplicate,
cross-filesystem, publish-race, post-publish rollback, and poisoned-retry tests.

## Process probes must succeed

Require return code zero, expected parse structure, and clean results from both
`lsof` and argv/`ps`. Missing, nonzero, malformed, timeout, or exception is a
refusal. Do not exclude a foreign process merely because its command contains
this tool's name.

## Release must recover at every durable state

Implement idempotent recovery at command entry for:

1. PREPARED with body still final;
2. PREPARED with body renamed pending;
3. probes complete with pending body;
4. body unlinked but COMPLETE intent absent;
5. COMPLETE intent present but receipt absent;
6. receipt present but final ledger event absent.

Every state is bound by source inode/mode/size/hash, exact final/pending
basename, confirmation, gate, partial file hash/seal, and intended receipt.
Unknown or inconsistent state refuses without deletion.

Hold the validated source FD through rename and unlink. Use a trusted root
`dir_fd`, compare the exact entry inode at every transition, and fsync the
directory after rename and unlink.

Publish the COMPLETE intent before the release receipt, then recover receipt
publication from the COMPLETE intent. All writes are byte-identical
replay-only; `allow_replace=True` for nonidentical intents is forbidden.
Inventory and prove preservation of every preexisting non-source path—no
hardcoded booleans.

Add failure injection/retry tests after every state transition.

## Release and aggregate binding

Release gate revalidates the full Revision-4A partial immediately before
rename. COMPLETE receipt binds schema/kind/status, source, deleted inode,
partial file hash/seal, PREPARED/COMPLETE intent seals, both process scans,
gate, confirmation, pending basename, and before/after disk state.

Aggregate requires and verifies the exact COMPLETE intent and release receipt
for each shard. Reject:

- `status=None`;
- missing/mismatched partial or intent hashes;
- wrong source/deleted inode;
- pending files or dangling symlinks (`lstat`, not `exists`);
- any complete duplicate under any name;
- incomplete recovery state.

Use durable replay-only writes for both aggregate JSON and Markdown.

## Source-body-free traps

Exercise real interception—not explicit guard calls—for:

- builtins/io/os/Path open variants;
- socket `connect`, `connect_ex`, and UDP send paths;
- current and late-import Hugging Face download APIs;
- `subprocess.run`, `Popen`, and downloader/network executables.

## Required gate

Run all prior tests plus the new reader, acquisition, probe, crash-recovery,
and aggregate mutations. No skipped test counts as evidence. Then run
selftest, two deterministic reload-verified preflights, `py_compile`, v2 and
census suites/selftests, and `git diff --check`.

Change only the four pilot/preflight deliverables. Do not claim real lifecycle
readiness unless every item above is implemented and fake-tested.
