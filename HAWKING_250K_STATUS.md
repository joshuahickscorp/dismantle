# 250k rebuild — status

Live campaign record. Updated as lanes land. Every number here is measured with the frozen
instruments, never estimated, unless it is explicitly labelled an estimate.

## Where the number is

```
                          campaign start        now
LOC (authority)                 430,633     see below
LOC (gated, + unearned
     generated reclassified)    433,505
directories                         131
source files                      1,196
crates                               20
public symbols                    9,508
functions                        14,642
files over 1500 lines                26

targets at 250k:  <=60 dirs, <=450 files, <=12 crates, <=4,000 symbols, <=7,500 functions
distance to primary:            183,505
```

## Phase 1 — complete

The campaign mandates the semantic graph before any architecture work. It is built,
repaired, and its findings are recorded.

| instrument | state |
|---|---|
| `control/SEMANTIC_GRAPH_SCHEMA.json` | frozen contract between extraction and analysis |
| `tools/graph/hawking_graph.py` | 25,069 nodes, 412,852 edges, 41s, byte-reproducible |
| `tools/graph/hawking_analyze.py` | eight analyses, 8.8s |
| `tools/graph/behaviour_map.py` | binds the constitution to the graph |
| `tools/graph/viewer/` | offline Cytoscape viewer, no network |
| `HAWKING_250K_GRAPH_FINDINGS.md` | what it says, including what it got wrong |

Agreement against the frozen measurement authority: LOC exact, functions +3.4%, public
symbols +2.8%.

## Phase 2 — complete

| artifact | state |
|---|---|
| `REBUILD_BEHAVIOUR_CONSTITUTION.json` | 210 behaviours, 13 domains, after an adversarial repair pass that deleted 20 non-behaviours |
| `REBUILD_BLACKBOX_TEST_MATRIX.json` | 86 runnable now, each blocked one naming its exact missing fixture |
| `REBUILD_DATA_MIGRATION_CONTRACT.json` | 24 persisted-state obligations |
| `REBUILD_PERFORMANCE_BASELINE.json` | 19 harvested historical metrics |
| `tools/verify/blackbox.py` | 86/86 passing, exit 0 |
| `tools/verify/perfgate.py` | 11 measurable metrics unsandboxed, refusals proven |
| `REBUILD_PERFORMANCE_BASELINE_MEASURED.json` | the real capture, **base TPS included** |
| `tests/fixtures/hide_workspace_v1/` | a real durable workspace the rebuild must read back |
| `tests/fixtures/generation_golden_v1/` | a model-free bit-exact greedy-argmax golden |

## Phase 3 — architecture, complete

Two independent clean-room designs, then a reconciliation.

- `HAWKING_VNEXT_ARCHITECTURE_A1.md` — top-down from semantic authority, six cores
- `HAWKING_VNEXT_ARCHITECTURE_A2.md` — bottom-up from four traced vertical slices, nine
  components, written without reading A1
- `HAWKING_250K_ARCHITECTURE_DECISION.md` — the reconciliation and the plan of record

They converge: A1 estimates ~166k with a realistic band of 200–230k, A2 estimates ~188k with
a band of 180–230k, and both name the same top risk. A2's nine components fold into A1's six
cores without strain, so the campaign's six-core constraint is met.

Two A1 hypotheses were tested and are resolved, both against A1:

- shader lattice generation — **half true**. The dtype and predecode axes parameterise
  cleanly (0.90–0.91 line similarity); the row-blocking axis does not (0.386). Worth
  1,500–1,900 lines, not 5,000.
- residency twins — **refuted**. `gravity_glm.rs` and `gravity_glm_resident.rs` share seven
  function names and no implementation; the best shared pair is 0.336 similar and 197 of the
  resident file's 204 functions have no counterpart at all.

## Phase 4 — the build, in flight

| lane | scope | state |
|---|---|---|
| `s2-lab` | Core C, 81,368 LOC of laboratory | building `lab/`, cutover in progress |
| `s3-hide` | Core D agent core, 83,430 LOC | **landed at 73,367**, one gate open |
| `s3b-tests` | the 87 assertions S3 dropped, plus the >1500 regression | running |
| `recomp-bridge2` | Core E, the 5,376 an independent review proved available | running |
| `recomp-p5-tests-docs` | docs and condense tests | running, predates this campaign |

### S3 result, in full

Scope 83,430 -> **73,367**, a real elimination of 10,063 with `relocated` and `facade` both
zero. The HIDE subsystem falls 107,129 -> 91,661. It did **not** reach its 52,500 target and
stopped with a measured gap of 20,867, which is the correct behaviour: the lane eliminated
unprotected mass and restructured the numbered `host_ops_0..5` chunking into semantic
`host_cmds/` modules, but did not complete the clean-room re-expression of the surviving
73k. `control/S3-report.md` carries the irreducible list, naming for every surviving block
the constitution behaviour that forbids deleting it. That list is the deliverable that
matters more than the number.

Gates after the merge: `cargo build --workspace` green, black-box 86/86, capability
inventory clean. **Two gates open:**

- the test inventory lost 162 logical assertions. 82 went with modules that were genuinely
  deleted; **87 came from `host_tests_0..3` and tested a host surface that still exists**,
  so they are owed a re-expression rather than a deletion. Lane `s3b-tests` is doing it.
- `files_over_1500_lines` is 27 against a campaign start of 26, from
  `crates/hawking-context/src/memory_classes.rs` at 2,533 lines. Same lane splits it.

### One flake found, and it is not S3's

`cargo test` after S3 showed `gravity_glm::tests::gpu_expert_table_icb_flag_defaults_off_and_requires_table_hit`
failing, in `hawking-core`, which S3 never touched. It passes with `-- --exact` and passes
with `--test-threads=1`: it reads a process-global environment flag that a sibling test
mutates concurrently, so it is **order-dependent, not a regression**. The pre-S3 run had it
green purely because the scheduling differed.

Recorded because a rung gated on "no new test failures" cannot be decided by a
non-deterministic suite, and because the tempting move — blaming the change in front of you —
would have been wrong here. Nine other failures are genuinely pre-existing and all GPU-bound
(`device_only_mlp_acceptance_vs_f64_reference`, `shader_registers_...`, seven `q8kv_seq*`).

### How much the black-box gate actually proves

Worth stating plainly, because "86/86 green" reads stronger than it is. Of 210 behaviours,
86 are runnable today; the rest name an exact missing fixture. Coverage is uneven:

```
domain          runnable / total        domain          runnable / total
HIDE_SESSION        12 / 21             GRAVITY              9 / 17
CONTEXT_OS           8 / 18             SOURCE               8 / 11
AGENTS               8 / 12             ARTIFACT             5 / 15
SECURITY             8 / 20             GENERATION           5 / 20
BRIDGE               5 /  7             SERVER               5 / 22
FABRIC               6 /  8             CLI                  4 / 23
ACCEL                3 / 16
```

So for a HIDE change, roughly half of the relevant behaviours are actually exercised. That
is why the inventory gate matters independently, and it is why it caught what the black-box
gate did not.

Not started, deliberately: Core B device and runtime, which holds the protected performance
and numerical contracts and is where a wrong boundary is expensive rather than merely
wasteful.

## The performance gate is now armed, with one caveat

The protected metric that matters most is captured:

```
base_tps.llama1b_decode_tps        median 130.4 tok/s      spread 122.6 .. 136.2
build.cargo_check_s                       0.370 s                0.335 .. 0.409
build.cargo_build_warm_s                   88.1 s                79.1  .. 94.5
startup.help_s                          0.00866 s              0.00786 .. 0.01088
transform.shard_write_verify_B/s       1.045e+08            9.90e+07 .. 1.60e+08
transform.pack_indices_B/s             4.387e+07            3.30e+07 .. 4.75e+07
numeric_parity.gravity_verify_s         0.0668 s               0.0580 .. 0.0735
kernel.bench_q4k_shapes_us             7.496e+05            6.10e+05 .. 8.60e+05
```

Two metrics stay unavailable and are recorded as such rather than faked: accelerated TPS and
`doctor` both need a GGUF that is not on disk. `glm52_math_preserve_tps` is skipped —
the artifact *is* present, so this one is a capture-cost decision, not an absence.

**The caveat is the spread.** Base TPS varies 122.6–136.2, about ±5%, because this capture
ran with five Grok lanes holding most of 28 cores. A gate that has to resolve 2% cannot be
decided by comparing two single captures taken hours apart under different load. That is
what `perfgate.py --paired` exists for: interleaved ABAB sampling with a sign test, so
contamination cancels. **Core B's performance gate must use the paired mode**, and any
absolute number quoted from this baseline carries its load conditions with it.

## Instruments this campaign added

```
tools/loc/hawking_generation_audit.py   closes the generated-source loophole; today it
                                        reclassifies 2,872 unearned lines back into the count
tools/campaign/rung_gate.py             the one command that closes or refuses a rung, with
                                        the anti-gaming refusal: a check that stops running
                                        is a failure, not a neutral
control/REBUILD_ACCOUNTING_RULES.json   five ledgers, generation rules, topology rules
control/LANE_MAP.md                     lane scopes and the one known merge collision
```

## Honest baseline, including what is already broken

`cargo build --workspace` green. `pytest tools/ ramanujan/ odyssey/`:
**2,045 passed, 16 failed, 44 skipped, 58 errors.** Every failure is environmental or a
stale reference and all are enumerated in `HAWKING_250K_BASELINE.json`.

One inherited defect matters: `tools/condense/glm52_terminal_proofs.py` opens
`tools/condense/glm52_external_baselines.py` as evidence, and that file was deleted in
`791ced2c` as a retired controller. Twenty-one tests error on its absence. It is the third
recorded instance in this repository of a reduction reported as capability-preserving while
a real consumer broke, and it is why the accounting rules make the inventory gate
non-skippable.

## What the prior arcs leave, and what they do not

The 430,284 floor is **not citable** — campaign section 11. The at-floor seal over 396,468
lines is **void**: every refusal behind it rested on public crate API, self-tested schemas,
`cfg(test)` surface, unwired-but-protected kernels, a host-selected tuning lattice, or a
crate manifest boundary, and this campaign declassifies all six.

What survives as evidence is the destructive probe: deleting every named protection stops
the *current* architecture at 349,455. That measures how much of the current tree is
load-bearing for the current architecture. It is not a bound on a different one.
