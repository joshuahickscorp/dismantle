# Temporal Gravity runtime receipt/profiler hardening — Revision 2

This is a controlling addendum to Revision 1. Revise the existing candidate in
place and preserve every earlier false-fence/no-model/no-MOP/no-promotion
requirement.

The first candidate's 39 green tests are not acceptance evidence. An immutable
audit executed production-accepting counterexamples. Every counterexample below
must become an independently assembled regression test.

## 1. Frozen TG promotion policy

Caller `--target-ms` never controls promotion. Freeze canonical milestone
thresholds in the versioned schema/source. A CLI value may request a stricter
diagnostic target only; it cannot increase the allowed milliseconds or produce
`promotion_verdict=PASS`.

Promotion requires the exact sustained 2K/8K/32K windows, at least 80 ordered
decode samples per context, all integrity/capability/evidence gates, and an
explicit frozen milestone actually achieved from recomputed warm samples.

Mandatory regression: 50 ms samples with `--target-ms 100` must return
non-promotion/failure and zero achieved TG milestone. Test every TG20/TG10/TG5/
TG2/TG1 boundary immediately above/equal/below the frozen threshold.

## 2. Strict artifact and external authority parsing

Hashing bytes is not enough. Strict-parse the artifact index, capability
register, approval receipt, and every JSON source input with nested duplicate
key, NaN/infinity, and trailing-content rejection. Validate exact
schema/version/closed fields and canonical seals.

Mandatory regressions: hash-valid artifact index with nested duplicate keys,
`NaN`, `Infinity`, `-Infinity`, or trailing content must refuse.

## 3. Complete live binding and seal coverage

Revision 1's external live bindings are mandatory. Additionally, the
deterministic semantic seal must cover every canonical execution fact:

- live Git commit/dirty state and exact source blob manifest;
- artifact path/hash/family/index/semantic seals;
- register and approval paths, byte hashes, canonical seals, predecessors,
  exact index hash, verdict, and both gates;
- actual executable path/bytes/build identity and loaded code identities;
- hardware and raw probe sources/methods;
- provider, raw execution record, resolved flags, `true_batch_1`,
  speculation/non-speculation, accelerated source, and fallback;
- workload, seed, tokenizer/prompt, context/decode, ordered output trace;
- every ordered cold/warm/all sample and window index;
- recomputed prefill, TTFT, p50/p95/median/TPS;
- bytes/operations/command-buffer/memory/swap/thermal/attribution facts;
- every source receipt/input path/hash.

The seal is only integrity, not authenticity; external live comparisons remain
required. Changing any one field and recomputing the public seal still refuses
if the live binding does not match.

Provider is mandatory, never optional. Resolved flags are an exact closed map
bound to raw runner evidence. Attribution is mandatory for production. A zero
logical/physical byte claim is accepted only when its schema explicitly allows
zero and raw instrumentation proves it; otherwise use typed unavailable.

Reject substring evasions such as `synthetic_v2`, case/Unicode variations, or
plausible arbitrary provenance. Production measured physical values require
the bound instrumentation record defined by Revision 1.

## 4. Cross-language canonicalization

Publish one language-neutral canonical receipt/seal specification and frozen
test vectors consumed by Rust and Python:

- UTF-8/Unicode normalization and escaping;
- object key ordering;
- integer/float representation, including IEEE-754 bit encoding where used;
- negative zero and subnormal handling;
- arrays and unavailable objects;
- exact exclusion rule for a self-seal field only.

Rust and Python must produce byte-identical canonical payloads and SHA-256 for
ASCII and non-ASCII paths/identities. Do not rely on serde defaults versus
Python `ensure_ascii` defaults.

Freeze statistics identically. For even sample counts, define median as the
mean of the two center values; define p50/p95 convention and index arithmetic.
The runner and gate share the same frozen vectors. Mandatory regression:
samples `1..80` must agree on median `40.5` and the frozen nearest-rank p50;
the runner output must pass its own gate. Include odd/even/repeated/boundary
fixtures.

Rust `source_hashes.runner_example` hashes the actual source file bytes from
the build/source manifest, not the `file!()` path string.

## 5. Profiler production parity

`gravity_profiler_acceptance.py` production acceptance requires the same live
Git, artifact, register, approval, executable, hardware, provider, raw
execution, workload, source manifest, semantic seal, context/window/sample,
output trace, and physical-instrumentation bindings as the speed gate.

It must refuse when any of these are missing:

- semantic seal or source hashes;
- hardware/executable/workload;
- provider, `true_batch_1`, or `non_speculative`;
- exact artifact/capability cross-hash;
- actual approval/register/executable/source files.

Its fixture constructor and marker-removal helper can never satisfy production.
Use an isolated authority tree with real temporary files and a terminal
`fixture_only=true` root that production refuses. Tests may validate mechanics
through an explicit non-production entrypoint only.

## 6. Mandatory exact counterexamples

Independently build/reseal one otherwise complete receipt and mutate each:

- Git commit to `deadbeef` or dirty mismatch;
- artifact path to a different/nonexistent index;
- arbitrary artifact semantic seal;
- nonexistent approval path plus forged hash/seal;
- absent provider and empty resolved flags;
- nonexistent executable with arbitrary SHA;
- arbitrary/missing source hashes;
- fake hardware/probe sources;
- fabricated prefill/median/window/bytes/ops/CB/memory/attribution;
- plausible but unbound physical value/source/method;
- cross-artifact capability hashes;
- Unicode path/identity canonicalization;
- 80-sample even median mismatch;
- public reseal after every attack.

Each production speed and profiler entrypoint refuses for the specific live or
arithmetic mismatch. Direct Python APIs cannot bypass production by setting an
`allow_synthetic`, `production=False`, relaxed threshold, or diagnostic flag
and still emit a promotion-eligible schema.

Run all affected tests, independent adversarial fixtures, profiler selftest,
Rust/Python canonical-vector parity, Rust example compilation/formatting, and
`git diff --check`.

No capable provider, TPS/TG milestone, real model body, MOP action, or
authorization-fence transition is permitted.
