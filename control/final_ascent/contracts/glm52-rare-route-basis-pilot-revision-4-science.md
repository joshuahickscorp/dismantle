# Revision 4A: make science and provenance validation semantic

Audit target:

- pilot `868956757599f98118f94d93e6c7a46dcf0fe6c5592131651f5e8a8bb1b22385`;
- tests `f0c7685d29c55b5dddb94f0c0ac4f7fffdf457bae2c30bff1f34fa904b1977ad`.

The preflight round-trip seal and dual-payload float16 scorer are fixed, but
integration remains rejected. This is a focused science/provenance correction.
Do not redesign lifecycle in this pass. Do not run any real source command or
touch a real source body or MOP.

## Exact rank semantics

For production partials:

- the rank-64 block must have `requested_rank == 64`,
  `emitted_rank_hidden == 64`, and `emitted_rank_down == 64`;
- the rank-128 block must have `requested_rank == 128`,
  `emitted_rank_hidden == 128`, and `emitted_rank_down == 128`;
- each of gate/up/down must carry finite deployed-basis and
  deployed-coefficient witnesses with `<f2`, C-order, exact shapes/ranks, and
  nonempty 64-hex hashes;
- the rank blocks must reference distinct physical bases and coefficient
  payloads;
- a claimed `non_capped_evidence` boolean cannot substitute for these checks.

The current direct mutation—changing rank-64 metadata to rank 1 and rank-128
metadata to rank 2—must make both `_validate_partial` and aggregate fail.

Tiny fake worlds may use reduced matrices only through an explicit test-only
rank map stored in injected `Runtime`, never through receipt-controlled data.
Production validation must refuse fake measurements and reduced ranks. Test
helpers may invoke a separate `allow_fake=True` validator only when the
in-process injected runtime is in test mode; serialized input cannot enable it.

## Exact panel and partial validation

Production `_validate_partial` must require and semantically verify:

- schema, format version, kind, `fake_measurement == false`, authorized
  measurement mode, exact shard ID/name/size/hash;
- exact five unique panel targets for the shard, with the sealed
  layer/expert/role/route-count tuple—especially count-1 and count-0 targets;
- exact code, test, preflight, census, rehydration, tensor-index, capsule, and
  member hashes;
- exact source before/after identity equality;
- complete target and eligible-peer tensor triplets, offsets, shapes, dtypes,
  and hashes;
- typed target split, shared-fit, peer-fit, and counterfactual witnesses;
- shared basis and coefficient identities/hashes at both ranks;
- finite, unique, complete gate/up/down score rows and deployed-payload
  witnesses;
- constructibility, zero/control exclusions, leakage proof, and every false
  fence.

Recompute structured witness hashes; reject dummy all-zero or syntactically
valid but semantically unbound hashes. Reject duplicate target rows and
duplicate score rows.

Build a fully valid production-like fake fixture in tests, then mutate every
field family above one at a time. The existing sparse helper must no longer
pass production validation.

## Typed counterfactual identities

Counterfactual reserves and scoring witnesses must use the same typed
`(member seal, layer, C-order row index)` identities as shared hidden fits.
Remove bare-index SHA witnesses from any integrity or promotion decision.
Tests must mutate member seal and layer while retaining integer row indices and
prove validation fails.

## Provenance seals

Verify, do not merely copy:

- route-census `receipt_sha256`;
- rehydration `seal_sha256`;
- authoritative official tensor-index SHA-256.

Use each artifact's documented canonical seal algorithm/key. If an upstream
format is not canonically verifiable, fail preflight and report the exact
format mismatch; never accept a nonempty seal as evidence.

Bind the exact four panel-layer capsule files/members from the sealed census
inventory. Validate capsule file hash, member raw-byte hash, normalized array
hash, dtype, shape, and sealed sidecar equality for both
`pre_router_hidden` and `topk_indices`.

## Aggregate recomputation

Aggregate must call production semantic validation with no fake/reduced-rank
escape hatch. It must independently require exact rank values and deployed
payload witnesses, then recompute the 9+9+27 score gates. Missing or
fabricated rank/basis/coefficient evidence fails even when all cosines are
0.97.

## Tests and gate

Add direct mutation tests for:

- rank 1/2 mislabeled as 64/128;
- wrong sealed route counts;
- dummy typed hashes;
- missing code/input/capsule/member hashes;
- missing or malformed deployed basis/coefficient evidence;
- duplicate target/score rows;
- counterfactual member/layer mutation;
- invalid census seal;
- invalid rehydration seal;
- wrong tensor-index hash;
- aggregate with perfect scores but invalid evidence.

Run the complete 50-test baseline plus new tests, selftest, preflight twice
with reload/seal verification, `py_compile`, v2 tests/selftest, census
tests/selftest, and `git diff --check`.

Change only the four intended pilot/preflight deliverables. Report exact
remaining lifecycle gaps honestly; do not claim real lifecycle readiness.
