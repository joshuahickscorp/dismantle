# Temporal Gravity runtime receipt and profiler hardening

Implement the cheapest correctness work that can run before a capable flagship
provider exists. This task must not read model bodies, run a heavy benchmark,
touch MOP, change runtime defaults, or claim a TPS promotion.

## Authority

Read:

- `NUMERIC_PARITY_V2.md`;
- `HAWKING_BASE_TRUE_TPS.json`;
- `HAWKING_PROFILER_RECOVERY_RECEIPT.json`;
- `HAWKING_FINAL_ASCENT_STATUS.json`;
- `odyssey/launch/SUBSTRATE_CAPABILITY.json`;
- `tools/condense/glm52_runtime_speed_gate.py`;
- `tools/condense/tests/test_glm52_runtime_speed_gate.py`;
- `tools/condense/gravity_profiler_acceptance.py`;
- `crates/hawking-core/examples/gravity_glm_tps.rs`.

Live source and current receipts override prose.

## Required result

Harden the runtime runner and acceptance gate so no future
`BASE_TRUE_TPS` receipt can be promoted unless it binds and validates:

- current Git commit and clean/dirty state;
- exact Gravity index path, file SHA-256, semantic/index seal, artifact family,
  and capability-approval receipt path/hash/seal;
- explicit proof that the exact index hash is APPROVED for both `G_math` and
  `G_live`;
- hardware identity: OS/build, CPU, GPU, RAM, device registry, and power mode
  when available;
- exact executable/build identity and complete resolved runtime flags;
- speculation explicitly off and no accelerated token source active;
- cold and warm sample windows, sample counts, per-token durations, p50/p95,
  TPS reconciliation, TTFT, and prefill;
- verified-once behavior and zero unverified reads;
- logical and physical bytes/token;
- operations/token and physical command buffers/token, with a typed
  unavailable reason rather than a fabricated zero;
- peak RSS, memory pressure, swap before/after/growth, and thermals when the
  host exposes them;
- context length, decode length, seed, prompt/tokenizer identity, and exact
  output-token trace hash;
- source receipt/input hashes and a deterministic semantic seal.

The gate must refuse a missing, stale, malformed, wrong-hash, wrong-provider,
capability-refused, capability-NOT_RUN, speculation-on, fallback, incomplete,
non-finite, internally inconsistent, or under-attributed receipt.

Do not silently accept old receipt schemas. A migration diagnostic may explain
why an old receipt is refused, but it cannot upgrade it.

## Profiler accounting

Represent attribution explicitly. The sum of attributed categories plus an
`unattributed` category must reconcile to measured wall time within a frozen
tolerance. Operations and command buffers must distinguish logical planned
work from physical submitted work. An unavailable physical counter is
`unavailable` with a reason, never integer zero.

Add source-body-free fake fixtures that exercise:

- one complete valid receipt;
- every mandatory field missing independently;
- stale/wrong index and capability hashes;
- `G_math` or `G_live` refused/not-run;
- speculation or fallback active;
- cold/warm count, TPS, TTFT, prefill, byte, operation, CB, memory, swap,
  thermal, and attribution inconsistencies;
- NaN/infinity, Boolean-as-integer, duplicate JSON keys, alias/fallback fields,
  and attacker-resealed malformed receipts;
- clean deterministic replay.

## Authorized files

Change only:

- `tools/condense/glm52_runtime_speed_gate.py`;
- `tools/condense/tests/test_glm52_runtime_speed_gate.py`;
- `tools/condense/gravity_profiler_acceptance.py`;
- `crates/hawking-core/examples/gravity_glm_tps.rs`;
- directly corresponding new source-body-free tests if indispensable.

Do not change runtime kernels, default flags, model artifacts, status/fence
files, or receipts representing real measurements.

## Gates and report

Run all affected Python/Rust tests, compilation/format checks, and
`git diff --check`. Report exact files/hashes, tests, remaining unavailable
host counters, and integration risk. State explicitly:

- no capable provider was created;
- no TPS/TG milestone was promoted;
- no real model body or MOP was touched;
- all authorization fences remain false.
