# Temporal Gravity Numeric Parity near-tie fallback — Revision 3

Revise the frozen Revision-2 candidate in place. Preserve every earlier
default-off, source-body-free, no-MOP, no-heavy-run, false-fence, complete-DAG,
ordered-boundary, two-phase-journal, and no-promotion requirement.

The only authorized predecessor blobs are:

- `crates/hawking-core/src/numeric_parity.rs`
  `6460f57088f6a78c6567561ca2f3eefa79fd65ce`;
- `crates/hawking-core/src/gravity_glm.rs`
  `61777934fcd0abf347f9e80d32b1b51417b2fe17`;
- `crates/hawking-core/src/gravity_glm_resident.rs`
  `476bce23a0eb050e2f057b9812a359a982008935`.

Refuse before editing if any differs. Remove `.serena`; it is not deliverable.
Revision 2 is `REVISE` and nonfunctional in production.

## 1. Explicit run-owned production session

Thread an explicit `NearTieSession` and opaque production authority provider
through the actual host and resident GLM session/forward signatures. Every
host/device router, DSA, and complete-head decision uses that owned handle.
Remove process-global policy, `active_run_id`, installed-session lookup, and
environment authority from production decisions.

Score-only helpers must not be the enabled production path. The current host
and resident callsites all refuse because no authority is passed, while the
authority-aware device helper is unused. Add source-body-free real insertion
tests proving each load-bearing callsite reaches the owned session.

## 2. Authentic immutable authority leases

Replace the in-memory caller-hash echo provider with provider-owned immutable
leases over the actual typed payload byte ranges and live buffer/address
generations. The caller never receives a mutable owned clone of authority
inputs.

Bind and re-hash/revalidate at acquisition, recomputation, durable prepare,
replacement, device completion, durable commit, and terminal:

- exact artifact/tensor bytes or immutable mapped range;
- codec/version, dtype, endian, shape/strides/offset;
- quantization metadata and exact f64 decode;
- live buffer/address generation and device identity;
- run/request/token/layer/domain/subgroup/candidate set;
- every router activation/gate/bias root;
- every DSA query/key/head-weight/scale/mask root;
- every head residual/RMS weight/epsilon/head-payload root.

Generation `1`, caller-provided digest strings, empty leases, omitted bias/RMS
roots, or zero-input DSA leases refuse. Reproduce and close these exact
counterexamples:

```text
before=[1.0,1.0] after=[-100.0,1.0] revalidation=Ok(())
dsa_leased_inputs=0 dsa_revalidation=Ok(())
```

## 3. Bind the fast DAG and make router authority coherent

Fast scores must be proven outputs of the exact fast DAG from the same leased
roots. Guarding an unrelated caller vector is forbidden. Recompute authority
unconditionally unless a sealed domain/backend/codec/device error bound proves
no authority inversion can cross a required boundary.

One router transaction must produce and commit:

- canonical sigmoid and corrected within-group choices;
- group strengths from the exact production-defined corrected scores;
- canonical group top-k and mask;
- masked global final expert top-k of the requested cardinality;
- exact ordered expert IDs and execution slots;
- normalized/scaled weights from the required uncorrected scores.

Do not return an unmasked full corrected vector for every router stage. Freeze
and test the current executable counterexample:

```text
chosen_groups=[0]
corrected_strengths=[0.611855...,1.168941...]
dag.expert_ids=[0,1]
returned_top2=[2,3]
```

The corrected successor must choose the production-authority group and return
one self-consistent masked decision/witness.

Freeze the transcendental implementation exactly; the identifier
`libm_or_rust_f64_exp` is ambiguous and cannot qualify.

## 4. Real atomic replacement and rollback

`NearTieDownstreamReplacement` metadata is not a replacement. Use a
transactional callback/owner operating on the actual insertion-point state:

- router IDs, weights, slots, traces, dispatch/descriptor buffers;
- DSA host/device ordering and attention consumers;
- head sample token, ordered top-k, next-token state, and publication;
- residual, KV/DSA caches, sequence length, address/table generations, and
  receipt state.

Durable prepare precedes mutation. A real device upload/commit/fence precedes
durable commit. Capture a byte-exact pre-token snapshot and actually restore it
on failure, or poison the owning GLM generation session. Store-only
`last_replacement`, comments that restore is “required,” duplicated execution
slots, or partial field sets refuse.

Poison is checked by real host/resident encode paths before every later
mutation. Run injected failures at authority, recompute, prepare fsync,
replacement, upload/commit, completion fence, commit fsync, terminal, and
publication.

## 5. Transactionally sound durable journal

Prepared records seal the complete canonical payload. Committed/aborted records
reference that exact prepared seal and cannot change decision, authority input,
replacement, or identity fields after prepare.

Recovery must validate closed schemas, predecessor seals, run/destination
ownership, ordered phases, and exact prepared-to-terminal bindings. A corrupt
or prepared-only tail must refuse or be durably truncated with file and parent
directory fsync before use. It may not return qualification-success metadata.

Close these exact counterexamples:

- change `committed_decision` and `authority_input_seal` after prepare, then
  commit;
- `recover_journal_committed_only` returns `Ok` on a prepared-only tail;
- two simultaneous sessions with the same run ID and different sinks both
  open.

Persist durable run-ID and canonical destination ownership across processes.
The terminal binds its exact final journal hash, ordered committed/aborted IDs,
all counters/domains, exact output/model-state identity, and predecessor seal.
Caller string `output:unset_fixture` is never authority.

Every directory fsync error is fatal; do not discard it.

## 6. Complete boundary and calibration evidence

Journal actual candidate IDs, not rank ordinals, for every internal adjacency
and admission boundary. Intended negative infinity must be proven by a bound
causal/mask witness; generic `-inf` in every domain refuses.

Guard calibration is a typed closed map keyed by domain, backend, codec,
device, arithmetic policy, and artifact/build identity. A prose binding note
cannot qualify.

## 7. Honest source/build identity

Embed the exact build-time source/generated-source manifest and relevant build
inputs. Hash the actual executable plus loaded libraries/Metal bytes, lockfile,
toolchain/compiler/linker/SDK, target/features, generated outputs, and flags.
Reading mutable source paths under `CARGO_MANIFEST_DIR` at runtime is not the
identity of compiled bytes.

## 8. Real insertion evidence

Use the real source-body-free host/resident router, DSA, and head insertion
points. Record actual authority acquisition, Metal synchronization/readback,
full f64 recomputation, prepare fsync, replacement/upload, device fence,
commit fsync, downstream continuation, D2H/H2D bytes, allocations, and work.

Remove:

- CPU vector copies labeled device readback/sync;
- constant order vectors labeled randomized/interleaved;
- fsync/replacement timings derived as 35%/5% of another interval;
- estimated H2D/fsync counters labeled measured;
- `includes_* = true` without direct observation;
- one helper relabeled as three different production insertion points.

Run warmups and randomized/interleaved modes with measured component/e2e
p50/p95. Evidence remains bounded insertion cost only, never product TPS or TG.

## 9. Exit

Return exact blobs/SHA-256 identities and complete executed test counts. Keep
production disabled and `calibration_missing` until a sound calibration and
capable provider exist. No real model, MOP, TG/TPS/HIDE milestone, capable
artifact, or authorization-fence transition.
