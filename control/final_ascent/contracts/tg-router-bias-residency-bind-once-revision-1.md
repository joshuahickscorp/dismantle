# Temporal Gravity router-bias residency — Revision 1

Revise the bind-once candidate in place. Preserve default-off/versioned,
non-wave, no-ICB, no-table-wave, rejected-C2 isolation, source-body-free,
physical-evidence, no-MOP, false-fence, and no-TG/TPS-claim requirements.

The correct implementation base is
`4e891c4ab092a569bebc4a9800ab33dd61932a28`. The full hash printed in the base
contract was a typo.

The only authorized predecessor blobs are:

- `crates/hawking-core/src/gravity_glm.rs`
  `83fc2f3050ed918fb4c78f99eb85afa432a9c4b7`;
- `crates/hawking-core/src/gravity_glm_resident.rs`
  `ca2d74615d95cba41d95aea8032bff15a1a78b3a`;
- `crates/hawking-core/tests/gravity_glm_router_bias_residency_bind_once.rs`
  `bc9b853874450bce0d1a31bbb52eb87855fd2e1b`.

The resident predecessor already withdraws an old binding after successful
reserve and before encode, and its focused test correctly forces a destination
generation rebuild before exercising an encode-step failure. Preserve both.
Refuse before editing if any blob differs. Remove `.serena`.

This revision additionally authorizes narrowly required changes to:

- `crates/hawking-core/src/metal/mod.rs`;
- `crates/hawking-core/src/cost_ledger.rs`.

## 1. Close failed-rebuild stale reuse

Revision 0 admitted this executable sequence:

1. cold bind bias A and publish A;
2. attempt rebuild to B;
3. write B into the live destination;
4. fail at commit/completion;
5. leave A published;
6. request A and return `WarmHit` with zero upload while destination contains B.

The successor must never leave a published record for bytes whose successful
completion is unproven. Reserve failure before the rebuild begins may preserve
the old record. After reserve begins, withdraw the old record before any
possible destination mutation. Every later failure leaves the key unbound and
cannot return a warm hit.

Do not host-write the live destination before the GPU transaction. Create an
immutable staging buffer and encode staging→destination. Publish only after a
checked successful completion. If completion may have partially mutated a
destination formerly used by an observed token, poison the session; otherwise
the next request must perform a full cold upload.

Add the exact A→B failure→A counterexample for every failure step. Verify live
destination bits, binding count, outcome, upload count, and poison state.

## 2. Real source authority, not caller labels

Production currently synthesizes `source_address` from a tensor label and
hardcodes generation `1`. Equal-byte unrelated allocations can therefore reuse
when given the same caller metadata.

Introduce a provider-owned immutable `RouterBiasSourceLease` created from the
actual artifact/tensor load:

- exact artifact/index/tensor content hash;
- immutable byte range and payload hash;
- dtype/endian/shape/offset/layout;
- live source buffer/address or mapped-file generation;
- weight-cache/artifact generation;
- layer/router/tensor identity;
- session/provider identity.

The residency engine accepts the lease, not independent caller strings.
Revalidate the actual source bytes/range and generation at reserve, before
commit, after device completion, and before publish. Ephemeral `Vec` addresses,
pointer equality, label hashes, length, or a constant generation are not
authority. Same bytes in an unrelated allocation do not reuse unless the
authority provider proves it is the same immutable tensor generation.

## 3. Checked Metal completion and abortable ownership

`TokenCommandBuffer::commit_and_wait` currently waits but does not require
`MTLCommandBufferStatus::Completed` or surface the command error. Add a
candidate-safe checked/abortable path:

- explicit prepared/encoding/committed/completed-success/completed-failure/
  aborted-before-commit states;
- Drop before commit never silently commits;
- after wait, require completed-success and no command error;
- retain staging/destination/source leases through the completion fence;
- no publish from destructor or local counter.

Do not alter legacy default-path semantics beyond adding the checked API.

## 4. Physical evidence from real APIs

Local increments are diagnostics, not physical proof. Emit ordered raw physical
events at the actual buffer allocation, staging copy, command creation,
encode, commit, completion/failure, wait, generation transition, publish, and
warm-hit APIs. Bind run/request/token/layer/source lease/device/executable and
predecessor event hashes.

The test receipt derives command buffers, waits, allocations, H2D bytes, and
completion status from those events. Ledger-off timing and ledger-on topology
remain separate.

## 5. Production observation and reset

The real router path must call `note_token_state_observed` only after a
successfully bound bias participates in a committed router decision. Failures
after an observed binding or possible partial GPU write poison the owning
session. Verified reset waits fences, discards destination/binding generations,
bumps session and poison epochs, and refuses if any step fails.

## 6. Gate matrix

Run on real local Metal:

- cold A = one upload; 200 warm A = zero further uploads/allocations;
- A→B failure at reserve/encode/commit/completion/publish→A;
- same-length changed bytes and unrelated same-byte allocation;
- source, artifact, cache, destination, device, session, and poison generation
  changes;
- alias/undersize/misalignment;
- checked command failure and verified reset;
- exact fixture router IDs/weights/order, with near-tie fail-closed;
- wave/table/ICB/replay probes all zero;
- default-off path unchanged.

Run focused rustfmt, Clippy with no new warnings, lib tests, and the focused
integration suite. Any warm stale reuse, unchecked completion, caller-identity
reuse, parity failure, physical-counter fabrication, or wall/topology
regression rejects the candidate.

## 7. Exit

Return exact blobs/SHA-256 identities, complete test counts, real Metal
topology/timing evidence, and explicit non-claims. Production/combined remains
baseline-only; no C2 revival, real model, MOP, `BASE_TRUE_TPS`, TG milestone,
HIDE promotion, capable-provider claim, or authorization transition.
