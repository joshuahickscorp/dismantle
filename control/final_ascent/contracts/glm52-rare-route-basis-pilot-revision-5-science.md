# Revision 5: close remaining science and provenance false positives

Revision 4A was audited from immutable Git blobs:

- pilot blob `0b307500c48097bc6b723eb96f2afe55c7943be5`
  (SHA-256 `478c21905378495b4b82e4a947510e177422a5899387e42dc535607ac33db618`);
- tests blob `a8839da0347b86fc1720c5572960305a0059db4d`
  (SHA-256 `2d17dd7f65c49eb7e33acf2b690b2d333767a69a260c66230a51d8796fa5b934`);
- preflight JSON blob `1ab551875388cc6680f3431405ff424210ac0e60`
  (SHA-256 `8295b32437e1f6a0557235db94fc644ab167408cb14a24ee393a5b8262b0d73c`);
- preflight Markdown blob `25b8232ebebc569443fba3896a04463769178e64`
  (SHA-256 `5c40a8d37107e6137f184bc4e203047f972b26374ec6df60779cdb34054bf5b3`).

Exact rank-64/rank-128 rejection is fixed, but Revision 4A remains rejected:
syntactically valid, freshly resealed evidence can still pass production
validation. This revision is science/provenance-only. Do not redesign or
authorize lifecycle, run a real source command, read a real model body, or
touch MOP.

## 1. Validate against an in-process authority bundle

Production validation and aggregation must receive an immutable, in-process
authority bundle constructed from the verified preflight and its exact inputs.
Receipt fields may never define or relax that authority.

The bundle must contain the exact expected:

- pilot, test, v2, and preflight artifact hashes;
- census file hash and canonically verified `receipt_sha256`;
- rehydration file hash and verified `seal_sha256`;
- official tensor-index file hash;
- shard name, size, hash, revision, and target-to-shard mapping;
- five target layer/expert/role/route-count tuples per shard;
- four panel-layer capsule, sidecar, and allowed-member identities;
- target and eligible-peer tensor metadata.

Every corresponding partial field must equal the authority value, not merely be
a nonempty 64-hex string. Missing authority is a production validation failure.
The production validator must not fall back to live moving files or accept an
authority object serialized inside the partial.

## 2. Require authentic capsule and member identities

For every panel layer, require the sealed sidecar and verify its documented
seal. Missing, stale, malformed, or mismatched sidecars fail closed. Remove all
fallbacks that synthesize a member hash from a capsule hash, layer, filename,
or explanatory metadata.

For both and only `pre_router_hidden` and `topk_indices`, independently bind and
cross-check:

- capsule file SHA-256;
- exact archive member name;
- raw archive-member byte SHA-256;
- normalized C-contiguous array-byte SHA-256;
- dtype, original shape, normalized shape, and C-order row count;
- sidecar value, census inventory value, and measured value.

Raw member bytes and normalized array bytes are distinct typed identities. Do
not copy one digest into both fields unless independently hashing both byte
streams proves equality. Sidecar equality must be true for each member.

## 3. Make every typed row witness externally bound

For target fit/holdout, shared hidden fit, peer down fit, exclusions, and
counterfactual reserves:

- recompute the witness from ordered
  `(member seal, layer, flattened C-order row index)` identities;
- require `n_rows == len(identities)` and exact agreement with the applicable
  tensor/capsule row count;
- require every identity's member seal to equal the authority-bound
  `pre_router_hidden` member seal for that layer;
- require every identity's layer to equal its enclosing layer/target;
- reject negative, duplicate, non-integer, or out-of-range row indices;
- for peer down samples, additionally bind the exact peer expert ID;
- verify fit/holdout disjointness and every sealed exclusion relationship from
  the typed identities themselves.

A self-consistent witness using an attacker-chosen member seal is invalid.
Bare-index hashes remain diagnostic-only and may not satisfy any integrity or
promotion gate.

## 4. Cross-bind score rows to their rank blocks

For every rank-64 and rank-128 arm-3 gate/up/down score, require:

- exact requested and emitted rank in both the block and each deployed witness;
- mandatory basis and coefficient witness `rank`, shape, dtype `<f2`, layout
  `C`, physical identity, serialized-byte hash, and typed deployed hash;
- exact projection-specific shapes;
- block hidden-basis hash/identity equal to the gate/up score basis
  hash/identity;
- block down-basis hash/identity equal to the down score basis hash/identity;
- coefficient identity bind the exact shard, layer, expert, projection, rank,
  source tensor, and deployed bytes;
- recomputation of every typed deployed hash from its metadata and serialized
  byte hash.

No witness rank is optional. A block-level rank or
`non_capped_evidence=true` cannot fill a missing score-witness field.

## 5. Enforce corresponding-rank physical distinctness

Compare corresponding score rows, not aggregate hash sets.

For each layer and target projection:

- the rank-64 and rank-128 basis identities and deployed hashes must differ;
- the rank-64 and rank-128 coefficient identities and deployed hashes must
  differ;
- shapes and serialized byte lengths must match their respective ranks;
- within one layer/rank, all targets must reference the same shared hidden
  basis and the same shared down basis;
- a target-specific coefficient may be referenced only by its exact target,
  projection, and rank.

Reject reuse at even one corresponding row. Other distinct hashes elsewhere in
the panel cannot mask a collision.

## 6. Require source, measurement, and tensor evidence

Production partials must have exactly:

- `fake_measurement == false`;
- `authorized_measure_path == true`;
- `measurement_mode == "production"`;
- mandatory source-before and source-after bindings with equal device, inode,
  mode, size, and full hash, matching the sealed shard authority;
- complete target and eligible-peer gate/up/down triplets.

For every tensor require the authority-bound name, projection, shard, dtype,
shape, byte offsets, payload length, serialized hash, finite-value result, and
non-overlap/in-bounds proof. A boolean such as
`tensor_bindings_complete=true`, an empty peer object, or a shape/dtype-only
record is never sufficient.

## 7. Bound all scientific scalars and sample counts

Every score scalar must be a finite real number. Require:

- `-1 <= mean_row_cosine <= 1`;
- `-1 <= constant_mean_cosine_null <= 1`;
- `surplus_over_null == mean_row_cosine - constant_mean_cosine_null` within a
  frozen numerical tolerance;
- `beats_null` exactly equal the comparison result.

Reject NaN, infinity, booleans used as numbers, out-of-range cosines, and
inconsistent derived fields.

Require exact score-row uniqueness by
`(shard, layer, expert, role, rank, projection)`. Bind each score's holdout
sample count to the typed holdout/counterfactual witness and require the
reported count to equal the number actually scored. Enforce the sealed
9 rank-64 promotion, 9 rank-128 promotion, and 27 rank-128 between/below rows
only after all evidence checks pass.

## 8. Aggregate only semantically validated evidence

Aggregate must call production validation with the immutable authority bundle
and no fake/reduced-rank escape hatch. It must independently repeat:

- exact rank and witness-rank checks;
- score-to-block basis binding;
- corresponding-rank physical-distinctness checks;
- authority hash equality;
- typed member/layer/sample-count checks;
- tensor completeness and source-before/after equality;
- finite cosine bounds and score-row uniqueness.

Only then may it recompute the `9 + 9 + 27` floors. Perfect cosines cannot
rescue missing, fabricated, self-consistent-but-unauthorized, or cross-rank
reused evidence.

## Required mutation gate

Start from a fully authority-bound production-like fixture. Independently
mutate, reseal, and require both production validation and aggregate refusal
for:

- any correct-length code, preflight, census, rehydration, tensor-index,
  capsule, sidecar, or member hash changed to another nonzero digest;
- missing measurement mode or authorized-path flag;
- missing or unequal source-before/source-after identity;
- tensor records missing offsets, lengths, hashes, bounds, or peer triplets;
- a witness with missing rank, physical identity, byte hash, shape, dtype, or
  layout;
- a score basis hash that differs from its block basis;
- one corresponding rank-128 basis or coefficient payload replaced by rank-64
  payload evidence;
- target, shared, peer, or counterfactual member-seal substitution with the
  same integer row indices and a recomputed self-hash;
- wrong layer, duplicate/out-of-range row, inconsistent `n_rows`, or wrong
  scored sample count;
- cosine `NaN`, infinity, below `-1`, above `1`, or inconsistent null/surplus
  fields;
- missing sidecar and any attempted synthesized-member fallback.

Retain all original, Revision 1–4A gates. Run the full fake-only suite,
selftest, preflight twice with reload/seal verification, compilation, existing
v2 and census suites/selftests, and `git diff --check`. Report lifecycle
readiness separately; this science revision cannot authorize real acquisition,
measurement, release, or aggregation.
