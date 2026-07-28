# Temporal Gravity runtime receipt/profiler hardening — Revision 3

This addendum controls Revision 2. Revise the frozen Revision-2 candidate in
place. Preserve every prior no-model-body, no-heavy-run, no-MOP, false-fence,
fixed-milestone, live-binding, physical-counter, and deterministic-receipt
requirement.

Revision 2 is `REVISE`, not promotable, for two executable defects.

## 1. One executed Rust/Python canonicalization authority

Revision 2 documents compatible algorithms but does not execute a
cross-language equality gate. Rust defines `fixed_float`, `seal_encode`,
`canonical_json`, and `semantic_seal`, yet invokes only the frozen statistics
vectors. Python tests loop `FROZEN_CANONICAL_VECTORS` through Python functions
only.

Create one checked-in typed vector artifact that is consumed by both Rust and
Python. It must cover at least:

- `-0.0`, ordinary finite fractions, smallest subnormal, and large finite
  values;
- Unicode, control characters, slash/backslash/quote escaping, arrays, sorted
  object keys, booleans, and null;
- a full semantic-seal receipt body;
- semantic-root-only exclusion;
- an actual runner-shaped raw execution record with floating measurements;
- an actual speed-gate embedded binding.

Rust must expose a no-model canonical selftest that emits or verifies exact
canonical bytes and SHA-256 values for every vector. Python must invoke that
compiled Rust selftest and compare its bytes/hashes with Python results.
Same-language duplicate implementations or comments are not cross-language
evidence.

All runner/gate/profiler hashes of semantic payloads must go through that
single canonical payload encoding. In particular, the Rust raw execution
record must not call plain `canonical_json(&execution_body)` on unencoded
floating values while Python calls a bit-hex float encoder. Use
`seal_encode`/canonical-payload bytes consistently, and use the same canonical
payload SHA-256 at the Python embedded-binding callsite.

An exact counterexample such as `{"x": 0.1}` must produce the same bytes and
SHA-256 on both sides. Revision 2 currently differs:

- raw Rust JSON bytes hash:
  `2b018c8f6dc2b0c13bf7d78415e6b376163debaaa178934fb6be1f1bc5ecf6b8`;
- bit-hex Python payload hash:
  `0c5b4b93ce105e316714525912c78ca24e65f03094b91d4f57b8f508a6197ddc`.

The successor must make this case identical and prove it at the real
runner-to-gate insertion point.

## 2. Receipt construction must use the frozen median

Revision 2's helper computes the even-sample median correctly, but the runner
receipt construction still emits:

`sorted.get(sorted.len() / 2)`

For samples `1..80`, that is `41.0`; the frozen authority and Python gate
require `40.5`. The actual receipt field
`decode_ms_per_token_median` must call the frozen median implementation.

Add a runner-shaped 80-sample integration vector. It must execute the real
receipt-construction callsite, pass the real Python gate, and assert median
`40.5`, p50 `40`, and p95 `76`. Testing the helper alone is insufficient.

## 3. Immutable correction and exit

Retain Revision-2 authorized file scope only:

- `tools/gravity_glm_tps.rs`;
- `tools/gravity_glm_tps_speed_gate.py`;
- `tools/gravity_metal_live_profiler.py`;
- their two focused test files.

No `.serena` file is deliverable. Return exact Git blobs and SHA-256 identities,
complete test counts, and a clean diff. The candidate remains a source-body-free
runtime/receipt closure only; it cannot claim `BASE_TRUE_TPS`, TG2/TG1, capable
provider status, HIDE promotion, or any authorization transition.
