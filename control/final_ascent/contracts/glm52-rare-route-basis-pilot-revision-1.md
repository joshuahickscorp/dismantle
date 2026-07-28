# Revision 1: close rank, leakage, byte, and lifecycle gaps

The preregistered panel and F1/F3 arithmetic are valid, but independent
scientific, test, and lifecycle audits found P0 gaps. Reconcile every item below
before the implementation can be integrated or any real source command can be
authorized.

The implementation/revision task still must not fetch, read, release, or modify
any real source body.

## 1. Exclude every panel route row from shared fits

The original contract inconsistently says to exclude panel holdouts in one
place and all panel routed rows in another. Use the stronger rule:

- for each layer, form the union of **all** route-row IDs for every panel target
  in that layer, including target fit and holdout rows;
- exclude that union from the shared hidden fit;
- exclude that union from every peer's routed rows before building shared
  SwiGLU `Z`;
- exclude all panel target experts/weights entirely from shared fits;
- prove co-routed top-8 rows cannot re-enter through a peer.

One physical shared basis identity must remain byte-identical for every target
that references it. Do not create target-specific leave-one-out bases.

Add fake tests where a target and peer co-route on the same row and prove the
row is absent from both hidden and down shared fits.

## 2. Require the requested numerical rank

Use an exact deterministic target split:

- `n_hold = max(1, ceil(n_routes / 5))`;
- `n_fit = n_routes - n_hold`;
- deterministic permutation/order is fixed by the sealed seed;
- count 1 remains holdout-only.

Constructibility is based on actual post-exclusion rows and emitted numerical
rank, not the old `MIN_ROUTE_ROWS=32` diagnostic:

- a target-local rank-128 basis requires `n_fit >= 128`; under this split that
  means at least 160 total routed rows;
- a target-local rank-32 residual requires `n_fit >= 32`; under this split that
  means at least 40 total routed rows;
- a shared hidden rank-128 basis requires at least 128 distinct post-exclusion
  `X` row IDs and 128 emitted columns;
- a shared down rank-128 basis requires at least 128 distinct
  `(peer_expert_id, row_id)` samples after exclusions/capping and 128 emitted
  columns;
- analogous rank-64 shared fits require at least 64 samples/columns.

Fail closed on rank cap, padding, duplicate-row inflation, or deficient
numerical rank. Never bill requested rank 128 while emitting fewer columns.

Add boundary tests at 159/160 target routes, 39/40 target routes, pooled
127/128, rank-deficient matrices, and repeated rows.

## 3. Seal counterfactual reserve before fitting

Zero-route diagnostics need an honest held-out contextual pool.

For each panel layer:

- derive a fixed layer-global counterfactual reserve from the complement of the
  union of all panel-routed rows;
- seal the reserve size, seed, ordered row IDs, and SHA-256 in preflight;
- exclude the reserve from shared hidden fits and all peer-Z fits;
- use the same reserve for every zero target and every constructible arm;
- label all zero results `COUNTERFACTUAL_DIAGNOSTIC`;
- keep `zero_route_representation_validated=false` unconditionally.

Fail preflight/measurement if the complement is too small for the sealed reserve.
Add tests that reserve rows cannot enter any fit.

## 4. Complete the promotion gate

The decisive `LAYER_H_LAYER_PEER_Z` treatment must run at both ranks for the
three promotion targets (nine gate/up/down scores):

- rank 64: minimum of nine `>= 0.85` and median of nine `>= 0.96`;
- rank 128: independently require minimum `>= 0.85` and median `>= 0.96`.

For arm 3 at rank 128:

- every route-conditioned between/below target tensor must reach `0.91`;
- there are 27 such scores: nine non-promotion routed targets times three
  tensors;
- the three zero targets and their nine scores are excluded permanently.

`shared_rare_route_candidate_survives_bounded_panel=true` and
`wider_bounded_pilot_authorized=true` require all:

- both promotion rank gates;
- all 27 between/below scores;
- requested ranks emitted without cap;
- no leakage/Gaussian/output-side/all-row promotion;
- every partial/lifecycle/integrity gate.

Keep full traversal false regardless.

Add tests that omit one promotion tensor, omit one of the 27 routed scores,
include a zero diagnostic, cap a rank, or fail either rank-specific promotion
gate.

## 5. Correct the dual-residual ledger

The current v2 ABI carries one basis identity/rank per tensor payload. Until a
separate fused dual-basis ABI is implemented and proven, bill the residual as a
second payload with its own three tensor headers per expert.

Exact rank-32 residual cost per expert:

- bases: `524800`;
- coefficients: `655360`;
- headers: `768`;
- total: `1180928`.

Across 19,456 experts:

- residual total: `22976135168`;
- add F3 rank-64 base total `56419758592`;
- corrected F5 total: `79395893760`;
- exact BPW: `20676014/24522459` (about `0.843146`).

The lower `79380951552` figure is forbidden; it omits `14942208` bytes of
residual headers. Regenerate preflight and add exact regression tests.

## 6. Seal all fit semantics in preflight

Preflight must bind:

- seed;
- split rounding and ordering;
- counterfactual reserve size;
- per-peer contribution cap;
- row deduplication versus `(peer,row)` multiset semantics;
- rank-deficiency behavior;
- exact basis identities;
- every candidate component total and receipt hash.

Deterministic regeneration must be byte-identical.

## 7. Harden acquisition

Do not reuse `glm52_rehydrate_window.py` as the implementation. It has an
environment-overridable 60 GB floor, accepts arbitrary/multiple shards, can
inherit cache paths, writes before refusal, and can expose an unverified final
body.

Require:

- literal `DISK_FLOOR_BYTES = 75_000_000_000`, no environment override;
- free-minus-sealed-size gate before source/cache/ledger/download mutation;
- an exclusive lifecycle lock, then a second free-space and state gate under
  the lock;
- exact confirmation and only shards 35/86/264;
- component-wise `lstat` rejection of symlinked roots/intermediates/targets;
- forced isolated cache variables (assignment, never `setdefault`);
- same-filesystem staging, exact size/full-hash verification, then atomic final
  publication;
- unique non-overwriting quarantine on mismatch;
- exactly one complete physical source copy by inode/content.

A completed hash-named HF cache blob is a second source body even if it does not
match `model-*.safetensors`. Refuse/avoid completed cache duplicates. Retain
metadata and incomplete files, but after acquisition and after release prove
there is no second complete copy of the sealed shard content.

The lifecycle lock may be the only control-plane file created after the first
read-only disk gate; re-run the disk gate immediately under the lock before any
other write.

Add tests for:

- equality at the disk floor passes and one byte below refuses without state
  mutation (apart from no lock creation before the first gate);
- environment override attempts have no effect;
- completed hash-named cache duplicate refusal;
- peak disk accounting includes staging/cache allocation;
- concurrent acquire exclusion;
- no unverified final path exposure;
- unique quarantine collisions.

## 8. Harden release and receipt publication

Reuse concepts, not unsafe details, from the old five-shard release controller.

Require:

- lifecycle lock and full in-process gate rerun;
- both `lsof` and argv/process probes must execute successfully and be clean;
- nonzero/missing probe fails closed;
- component-wise no-symlink containment without resolving first;
- open target with `O_NOFOLLOW`, compare `fstat`, size, and full hash of the
  opened identity;
- revalidate the partial and source identity immediately before deletion;
- unlink the exact basename relative to a trusted pilot-root directory
  descriptor;
- no resolved-path unlink, glob, recursion, or cache cleanup;
- unique `O_EXCL` receipt staging, file `fsync`, atomic replace, directory
  `fsync`, and post-write seal verification.

A scientifically failed but complete, integrity-valid partial **must remain
releasable**. Release gates integrity/safety only; aggregate owns the scientific
fail/survive decision.

Do not inspect MOP to establish protected-path safety. Prove containment from
the exact support/pilot root without touching MOP.

Add adversarial tests for intermediate symlinks, TOCTOU identity mutation,
missing/nonzero process probes, receipt staging collision/symlink, replace
failure, valid scientific failure release, and arbitrary retained
cache/incomplete sentinels.

## 9. Aggregate scope

Aggregate must:

- require three valid partials and three valid release receipts;
- prove all three published bodies absent and no complete cache duplicate;
- separate lifecycle integrity from scientific verdict;
- exclude zeros from every promotion aggregate;
- publish `zero_route_representation_validated=false`;
- call this a bounded routed rare-transfer pilot, not validation of zero-route
  experts, layer 78, all shared-basis families, or the full population;
- keep every safety fence and full traversal false.

## Required rerun

Regenerate the source-body-free preflight. Run:

- full fake-only rare-route tests;
- rare-route `selftest`;
- `py_compile`;
- existing v2 tests and selftest;
- route-census tests and selftest;
- `git diff --check`.

Commit only the four intended preflight deliverables. Exclude `.serena`, real
source files, lifecycle partials/receipts, caches, and temporary artifacts.
Report exact pass counts and corrected receipt/byte hashes.
