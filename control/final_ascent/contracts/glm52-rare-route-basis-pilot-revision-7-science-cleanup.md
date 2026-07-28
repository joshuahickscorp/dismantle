# Revision 7: remove remaining science/provenance fail-open paths

Revision 5 implementation artifacts were independently audited as immutable
Git blobs:

- pilot `f4fbcd6ced387263b58f370808ab0a1e21bfc611`
  (SHA-256 `e1851e01fed3133c3aac64b48f1f0920ef5afcfce124f04ccf660c1fdd9d1d1d`);
- tests `13eea9b2fcde78a14a84985eb012a6b7e76e861b`
  (SHA-256 `eb67132c2330219f9bbd19fba7efd7226594a261c7c6d42a44e3741cd1b6e183`);
- preflight JSON `f0ec9f8c72fe766cf1c465f06d7034dd894a9277`
  (SHA-256 `980fdfecd0dd530cb2bbae0ac9a888bdfc13fdb29316c64f5b0de94c9d65b706`);
- preflight Markdown `cf1c58c45318ee7a1039646efceefa3cfd7e1638`
  (SHA-256 `792e2373102dd124e209fef7a162fcd9f9cbc648d626b701938e4a14dea091b7`).

Most prior mutations now refuse, but three independently reproduced,
production-shaped mutations still return `errors=[]`:

1. delete top-level `sidecar_seal_verified` and `sidecar_seal_sha256`;
2. delete `device` from both source bindings;
3. change a basis `bytes_sha256` and set
   `deployed_sha256_is_raw_payload=true`.

Apply this science/provenance cleanup after Revision 6. Revisions 1–6 remain
cumulative. This revision does not weaken Revision 6 lifecycle requirements
and cannot authorize a real source read, measurement, release, aggregate, model
change, or MOP action.

## 1. Authority fields are required before they are compared

For every authority-bearing object, validation must first require the exact
schema and every mandatory field, then compare or recompute every value. A
missing field, `None`, wrong type, alias, empty value, unknown field, or
unrecognized schema version refuses. Patterns equivalent to these are
forbidden for required evidence:

```text
if value is False: error
if value is not None and value != expected: error
value = nested.get(name) or receipt.get(name)
if receipt_boolean: skip_validation
```

Production receipts use one canonical representation. Top-level or legacy
aliases cannot fill a missing nested field. Redundant aliases must be removed;
if retained only for migration diagnostics, all supplied values must be
type-identical and equal to the canonical value, and the canonical value
remains mandatory.

The immutable in-process authority, not the receipt, decides which evidence is
required. No receipt field may make a required block optional.

## 2. Capsule sidecar and member identities are complete

For every authority panel layer, require and cross-bind:

- `sidecar_seal_verified == true`, derived by the validator;
- exact `sidecar_seal_sha256`;
- exact capsule SHA-256 and sidecar/census identities;
- both and only the authorized `pre_router_hidden` and `topk_indices` members;
- exact archive member name;
- independently computed raw archive-member byte SHA-256;
- independently computed normalized C-contiguous array-byte SHA-256;
- dtype, original shape, normalized shape, C-order row count, and array length;
- exact sidecar value, census inventory value, and measured value.

Absence of any field refuses. The validator must compare member names even when
the receipt omits them. It may not synthesize a member digest, reuse the
normalized digest as the raw archive digest, or accept a sidecar summary
Boolean instead of the sidecar seal.

`raw_and_normalized_equal` and `sidecar_equality`, if serialized, are derived
claims only. Independently hash and compare the typed byte streams first, then
require each claim to equal the derived result. A serialized
`independently_hashed_equal` assertion is not evidence and must be removed.

Production authority construction requires exact raw archives and the exact
preflight artifact. `require_archives=false`,
`require_preflight_file=false`, a default/live capsule-directory fallback, or
an unbound `member_bytes_sha256` is forbidden in every production validator,
aggregate, and preflight path.

## 3. Source identity includes device and immutable source revision

Both source-before and source-after bindings must contain and exactly match:

- device, inode, file type/mode, size, and full file SHA-256;
- authority shard name, size, and SHA-256;
- exact Hugging Face repository and immutable revision;
- the same verified source identity before and after measurement.

Missing `device`, `hf_repo`, or `hf_revision` refuses. No field is conditionally
compared only when supplied. The repository and revision come from the
in-process authority and cannot be receipt-defined.

## 4. Serialized validation escapes are forbidden

Remove `deployed_sha256_is_raw_payload`, `allow_payload_digest_only`,
`independently_hashed_equal`, and every serialized `allow_*`, `skip_*`,
`*_only`, raw-payload, compatibility, fake, reduced-evidence, or similar field
that changes whether mandatory evidence is parsed, recomputed, or compared.
Unknown former escape fields refuse rather than being ignored.

Every basis and coefficient witness must include its canonical nested rank,
shape, dtype, layout, physical identity, exact serialized-byte hash, and typed
deployed hash. Recompute the typed deployed hash unconditionally and require
exact equality. A raw-payload digest may be retained as an additional typed
digest, but it cannot replace or excuse a typed deployed digest.

Summary fields such as `constructible`, `constructibility_witnessed`,
`leakage_proof`, `zero_route_promoted`, `tensor_bindings_complete`,
`non_capped_evidence`, and `beats_null` may not control validation. When the
schema retains one, derive it after validating its complete typed evidence and
require the serialized value to be present, Boolean-typed, and exactly equal
to the derived result.

## 5. Required scientific evidence has no conditional skip

Validate every authority-required target, rank, projection, score, and witness
whether the receipt says it is constructible or not. A false or missing
`constructible` field cannot skip rank-block or score validation. Only an
authority-declared optional block may be absent, and Revision 5's sealed
rank-64/rank-128 promotion, between, and below rows are not optional.

Require typed, authority-bound witnesses for:

- target fit and holdout rows;
- shared-hidden fit rows;
- peer-down fit rows with exact peer expert identity;
- exclusions and all fit/holdout disjointness relationships;
- counterfactual reserves and the exact rows actually scored.

For each witness, recompute the ordered typed identities and seal; require
exact layer, member seal, peer where applicable, integer row values, bounds,
uniqueness, `n_rows`, and authority row count. A bare-index hash or positive
integrity Boolean cannot satisfy the requirement.

## 6. Tensor and preflight identities are authority-cross-bound

The production authority bundle must carry the exact preflight file hash and
semantic seal plus the complete target and eligible-peer tensor inventory.
For every gate/up/down tensor, compare the receipt to the authority for:

- tensor name, projection, shard, dtype, and exact shape;
- byte offsets, payload length, and exact serialized payload SHA-256;
- non-overlap, in-bounds, and finite-value results derived by validation;
- target or exact peer expert identity and authority-defined order.

Presence, shape plausibility, nonempty hashes, or receipt Booleans are
insufficient. Missing, duplicate, extra, reordered, renamed, or authority-
unlisted tensor records refuse.

Authority construction must verify the preflight’s schema, file hash, semantic
seal, embedded code/test/input hashes, and predecessor bindings. Merely hashing
an optional preflight pathname is insufficient.

## 7. Counts and score records use one canonical schema

Use one mandatory score-count field. Remove the fallback chain among
`n_holdout_rows`, `n_scored_rows`, and `sample_count`; if legacy aliases remain
for diagnostics, require all of them to equal one another and the count derived
from the typed scored-row witness.

Every rank block and score row must be present according to the authority
inventory and validated independently. Exact row uniqueness, scalar bounds,
null/surplus arithmetic, `beats_null`, basis/coefficients, corresponding-rank
physical distinctness, and typed sample identity are checked before any floor
is counted.

## 8. Aggregate independently repeats every science check

Aggregate must reject a partial that fails any Section 1–7 rule and must also
independently repeat:

- canonical schema and mandatory-field presence;
- exact authority and preflight binding;
- source device/repository/revision and before/after identity;
- capsule, sidecar, raw-member, normalized-member, and row metadata;
- complete target/peer tensor inventory;
- every typed fit, holdout, shared, peer, exclusion, and counterfactual
  witness;
- basis/coefficient metadata and typed deployed-hash recomputation;
- rank-block/score binding, count consistency, scalar validity, row
  uniqueness, and corresponding-rank distinctness.

Aggregate may not treat a partial validator's empty error list, a status, or a
summary Boolean as proof. Floors are recomputed only after all independent
checks pass.

## Required unmaskable mutation gate

Start from one complete production-shaped fake fixture that passes both the
production partial validator and aggregate with the immutable in-process
authority. For every mutation below:

1. deep-copy the passing baseline;
2. apply exactly one declared mutation and assert that it changed the intended
   field;
3. recompute only the receipt's attacker-controllable seal;
4. call the actual production partial and aggregate entrypoints;
5. assert both return at least one stable, field-specific error;
6. restore and revalidate the pristine baseline before the next mutation.

No helper may silently restore a deleted field, rebuild the expected authority
from the mutated receipt, preclassify the result, or omit a mutation because a
fixture field is absent. Every parametrized case must report its executed case
count. Zero collected cases, skip, conditional return, `xfail`, filtered
parameter, fixture fallback, or assertion against a helper instead of the
production entrypoint is a gate failure.

At minimum, independently mutate and reseal:

- delete `sidecar_seal_verified`;
- delete `sidecar_seal_sha256`;
- delete both together;
- delete `device` from source-before, source-after, and both;
- change a basis `bytes_sha256` while adding each former deployed-digest escape
  flag, alone and in combination;
- delete `typed_deployed_sha256` while adding
  `allow_payload_digest_only=true`;
- add every removed/unknown serialized validation escape field;
- delete or alter every capsule/member name, hash, dtype, shape, row-count,
  sidecar-equality, and raw/normalized-equality field;
- assert raw/normalized equality or independent hashing without matching
  independently computed digests;
- omit or alter exact preflight file hash/seal and attempt archive/default-path
  fallback;
- omit, rename, reorder, duplicate, or alter every target and peer tensor
  authority field;
- omit each typed shared, peer, exclusion, target, holdout, and
  counterfactual witness; substitute its seal/layer/peer; use duplicate,
  non-integer, negative, or out-of-range rows;
- set or delete each summary Boolean, including `constructible`,
  `constructibility_witnessed`, `leakage_proof`, `zero_route_promoted`,
  `tensor_bindings_complete`, and `non_capped_evidence`;
- make one redundant sample-count alias disagree while another remains
  correct, and delete the canonical count;
- delete canonical nested deployed metadata while supplying a top-level alias;
- delete or alter `hf_repo` and `hf_revision`;
- remove or corrupt any aggregate-only copy of the same evidence.

Retain every Revision 1–6 mutation and lifecycle gate. Run the full fake-only
suite with the exact expected test count, zero skips and zero xfails, selftest,
two byte-identical reload/seal-verified preflights, compilation, existing v2
and census suites/selftests, and `git diff --check`.

The preflight receipt and Markdown must name every executed mutation group,
report pass/fail counts without aggregation masking, bind the exact contract,
pilot, test, preflight, and input hashes, and refuse publication unless every
required case executed and both validation levels rejected it.

This file is the contract only. Implement it later by changing only the
explicitly authorized pilot/test/preflight deliverables. Report integration
acceptance separately from real science or lifecycle authorization.
