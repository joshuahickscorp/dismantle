# Temporal Gravity runtime receipt/profiler hardening — Revision 1

Revise the existing candidate in place. The first pass is not promotable even
though its targeted tests pass. Keep all prior requirements in
`tg-runtime-receipt-profiler-hardening.md`; the requirements below are
additional and controlling.

Do not read model bodies, run a heavy benchmark, touch MOP, change runtime
defaults, or claim a TPS/TG/capability promotion.

## P0: replace self-assertion with live bindings

The acceptance path may not validate a field merely by checking that it looks
like a path, hash, seal, provider, hardware name, or PASS Boolean. For every
load-bearing binding, either recompute it from a named raw runner observation
or refuse:

- resolve the repository root and compare receipt Git commit and dirty state to
  live Git state; bind the exact candidate source blobs as well as the commit;
- resolve the artifact index path, require it to be the exact requested path,
  hash its bytes, recompute every defined index/semantic seal from the canonical
  index contents, and reject path aliases or a hash-valid different path;
- resolve, open with strict JSON, and SHA-256 the capability register and the
  exact approval receipt; validate the approval receipt's own canonical seal;
  require register and approval receipt to name the same exact index hash,
  artifact family, capability verdict, and both `G_math` and `G_live` outcomes;
  receipt-local gate PASS fields alone are never evidence;
- resolve and hash the executable bytes actually invoked by the runner; bind a
  real build identity when available or carry typed unavailable evidence, but
  never accept an arbitrary executable path/hash pair;
- bind hardware, device registry, OS/build, RAM, and available power state to
  raw runner probes with named source/method. A caller-authored string is not
  hardware evidence;
- bind every resolved flag, provider, `true_batch_1`, non-speculative state,
  fallback state, and accelerated-token-source state to one raw execution
  record emitted by the invoked runner. Environment reconstruction and
  duplicate summary Booleans cannot substitute for that record;
- resolve and hash every source receipt/input named by `source_hashes`; reject
  unrecognized, missing, mutable-without-hash, or receipt-local dummy hashes.

When current repository authority has no APPROVED capable provider/receipt,
the gate must fail closed with a precise diagnostic. Unit fixtures may build an
isolated temporary repository, artifact, register, approval receipt,
executable, and raw-probe record, but production must never accept the fixture
marker or fixture root.

## P0: capability approval closure

`approval_receipt_path`, `approval_receipt_sha256`, and
`approval_receipt_seal` are currently load-bearing. The validator must:

1. resolve the path under the permitted authority root;
2. hash the live bytes and compare the declared SHA-256;
3. strict-parse the live receipt;
4. recompute its canonical seal while excluding only the seal field defined by
   its versioned schema;
5. validate the schema/version and reject unknown fields or aliases;
6. require its exact predecessor/register hash and exact artifact index hash;
7. obtain `APPROVED`, `G_math=PASS`, and `G_live=PASS` from this live receipt
   and reconcile them with the live register.

Missing or incompatible authority schema must refuse. Do not invent an upgrade
for current legacy receipts.

## P0: semantic seals are not authenticity

An attacker can recompute a public semantic hash after forging payload fields.
Treat deterministic seals only as integrity checks. Acceptance also requires
the live external bindings above and reconciliation to raw per-token evidence.
Add attacker-resealed tests that preserve all internal arithmetic while
changing, independently:

- commit/dirty state;
- artifact path, family, semantic seal, or approval receipt;
- executable path/hash;
- provider/flags/speculation/fallback;
- hardware/device registry;
- sample sequence, window indices, context/decode length, output trace;
- source receipt/input path or hash;
- counter source/method/value.

Each must fail for the external mismatch, not merely because a stale seal was
left behind.

## P0: raw samples and timing

Bind the exact ordered cold/warm samples to the runner execution record. The
gate recomputes sample counts, p50, p95, median, TPS, TTFT, prefill, window
indices, trace length, context, and decode length. No producer aggregate or
caller threshold controls TG classification.

Freeze the percentile convention in the schema and test odd/even lengths,
boundary ranks, repeated samples, sub-millisecond values, and a receipt with a
fabricated 0.5 ms aggregate over 100 ms raw samples.

## P0: physical evidence

`source` and `method` strings are not proof by themselves. Measured physical
bytes, operations, command buffers, timestamps, waits, and dispatches must
reconcile to a raw instrumentation record carrying:

- instrumentation schema/version and source blob hash;
- executable/build identity;
- request/run/token identity;
- monotonically ordered raw events or samples;
- exact derivation used for the per-token value.

If this evidence is not available, emit typed `unavailable` with a reason and
do not use it for a physical claim. Arbitrary positive integers with plausible
source/method labels must fail.

## Profiler acceptance independence

The production `gravity_profiler_acceptance.py` entrypoint must validate the
same live repository/artifact/capability/approval/executable/raw-run bindings,
not just presence-shaped fields. It may share strict parsing and canonical
schema helpers, but its tests must use independently assembled adversarial
inputs rather than only mutating the module's own fixture constructor.

Production CLI values may only tighten frozen floors. Diagnostic-below-floor
mode must emit a schema that is terminally ineligible for promotion and cannot
be mistaken for an accepted production receipt.

## Required tests

Add source-body-free isolated tests for all items above, plus:

- duplicate keys at the root and every nested binding;
- JSON `NaN`, `Infinity`, `-Infinity`, trailing content, and Boolean integers;
- missing/wrong/stale register and approval receipt;
- approval receipt with a valid byte hash but invalid canonical seal;
- register/approval/index cross-binding mismatch;
- symlink/path-alias substitution;
- wrong live executable bytes;
- live dirty-state mismatch;
- plausible but fabricated physical counter provenance;
- deterministic replay of a complete isolated valid fixture.

Run the targeted Python suites, profiler self-test, Rust example compilation,
formatting, and `git diff --check`. Report exact file hashes and remaining
typed-unavailable evidence.

All capability/status/launch/authorization fences remain false. No capable
provider, TPS/TG milestone, real model run, or MOP action is authorized.
