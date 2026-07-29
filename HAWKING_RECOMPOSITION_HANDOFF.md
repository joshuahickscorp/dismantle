# Semantic recomposition arc — continuation handoff

State at handoff. Branch `arc/300k-integration`. Everything below is measured with the pinned
tools, never estimated.

## Where the number is

```
                 arc start      now       ladder position
directories            179      131       rung 1 green (-25% target, hit -27%)
source files         1,227    1,195       rung 1 at 982, not reached
crates                  33       20
hide crates             20        7       target <=7 MET
hide directories        59       23       target <=30 MET (-61%)
hide files             349      326       target <=209 not met -- 13 real modules
                                          replaced 6 concatenations, deliberately
public symbols       9,412    9,502
functions           14,587   14,635
files over 1500         30       26       BELOW baseline -- debt paid
LOC                430,284  430,146       -138
```

Prior arc floor was 430,284 under subtraction. This arc rejected that as the semantic floor
and set out to beat it by rewriting architecture. It has not yet, and the reason is
consistent across every programme: **topology compresses hard, LOC does not.**

## Measure with these, nothing else

```
python3.12 tools/loc/hawking_loc.py                    # LOC authority
python3.12 tools/loc/hawking_topology.py               # dirs/crates/files/APIs/functions
python3.12 tools/loc/hawking_inventory.py --snapshot P # writes BOTH inventories
python3.12 tools/loc/hawking_inventory.py --gate A B   # diffs both, fails on either
```

`--gate` is not optional. It exists because a merge deleted `tools/adapters/verify_grades.py`
(220 lines, live `__main__`) and was reported capability-preserving. The detector had not
failed; only the assertion half was ever run. `--gate` refuses to pass when a snapshot half is
missing.

`hawking_topology.py --diff` fails when **any** dimension regresses, so a rung cannot be
claimed by trading structure for lines.

## Six programmes, five closed with named contracts

| programme | verdict | the contract |
|---|---|---|
| 3 runtime graph | **rejected** | WKV-7's fixed per-head S state and token-shift cannot share an executable LayerOp core with a transformer's growing KV + RoPE + MHA while staying bit-exact on argmax/top-k, holding 1 command buffer/token, and being net-negative. IR cost 2,250 to delete 480; functions 154 -> 170. Six irreducible operators named, three per family. |
| 4 function merging | **0.95%** | 138 twins of 12,258 functions, net -3,157. Converges with the prior arc's 1.02% from an unrelated instrument. 12 near-misses recorded where text similarity would have merged behaviourally different code. |
| 1 laboratory | **decomposed** | `glm52_state` fit none of the six categories; its TOCTOU lease split out to `engine/lease.py` and the fusion proved **historical, not semantic**. No blocking contract. 12 modules, 105 files, LOC +156. |
| Bridge/Fabric | **OVERTURNED** | claimed floor 20,426; reviewer proved it is where the lane stopped, not a floor. Honest floor 15,148, **5,376 available**. The dual-source `bridge_surface` lockstep has no crate dependency either way, so one side can generate the other. |
| 2 HIDE | **topology real, debt paid** | 20->7 crates and 59->23 dirs genuine. First attempt was concatenation (real elimination 1,416, not 8,634); a second lane decomposed it into 13 modules with zero transplant markers and files-over-1500 below baseline. |
| 5 tests/docs | **not run** | |

Track V's four slices all measured. Every one: the path is tiny, the floor beneath it is not.
Campaign 2,540 path against a 27,594 science floor — measured twice from opposite directions,
by a Track S scout at 27,500 and a Track V builder at 27,594.

## Debt paid

**HIDE concatenation — resolved.** Thirteen real modules replaced the mega-files. `host.rs`
went from 12,169 lines to a largest-HIDE-file of 1,388; no HIDE file exceeds 1,500; the
repository count fell from 39 to **26, below the 30 it started at**; every transplant marker
is gone. The seven crates and twenty-three directories stand. HIDE files rose 209 -> 326
because genuine modules cost more files than concatenations, which is the correct trade under
an arc that ranks file topology above line count.

Verified by the controller: `--gate` passes on both halves, hide-backend 311 passing,
hide core/protocol/kernel suites pass, hawking builds untouched.

## In flight at handoff

```
recomp-hdecomp              decompose host.rs and friends, keep 7 crates / 23 dirs
recomp-probe-destructive    destructive probe in the NEW architecture, scratch only
recomp-revlab-adversarial   attack the laboratory and Bridge results, both unreviewed
```

The probe's decisive question is the one that settled the last floor: **is the mass above
target even inside the protected set?** Last arc it was not — the residue lived in laboratory,
live HIDE product and `hawking-core`, none of which the refusals contained.

## Two controller errors this arc, both caught by review not by gates

1. Merged HIDE as recomposition when it was concatenation. My topology tool reported
   `files_over_1500` 30 -> 39 at that exact merge and I read past it.
2. Deleted `verify_grades.py` and reported capability preserved, because I ran only half the
   inventory diff.

Procedural fix, not yet habitual: **the adversarial reviewer runs before the merge.** Both
failures were found by review afterwards. The same lesson appeared in the prior arc when the
one unreviewed floor turned out to be 0.9% early.

## Rollback

Tags: `arc-437k-green`, `arc-447k-green`, `arc-rung-475k-green`, `arc-rung-550k-green`,
`arc-honest-561k`, `arc-430k-green`, `arc-floor-sealed-437k`,
`HAWKING_SEMANTIC_CORE_FLOOR_COMPLETE` (prior arc endpoint).
Baselines: `HAWKING_RECOMPOSITION_BASELINE.json`, `HAWKING_TOPOLOGY_BASELINE.json`.
Verdicts: `control/RGRAPH-verdict.json`, `FMERGE-verdict.json`, `VLAB-verdict.json`,
`REVHIDE-verdict.json`, `RECOMP-SCOUT-COMPARISON.json`.

## What has not been done

- LOC ladder: 400k not reached, 350k mandatory attempt not reached
- Files ladder: rung 1 (982) not reached
- Programme 5, tests and docs recomposition
- Independent review of the laboratory and Bridge results (in flight)
- Destructive probe (in flight)
- Formal declaration of the single binding behaviour contract that blocks descent
- HIDE split — correctly not attempted; it earns zero credit and is gated on the floor

## The honest read

Five programmes have now returned named behaviour contracts rather than reductions. The
architecture compresses in structure and refuses in volume. Whether 350k is reachable depends
almost entirely on the laboratory science floor (27,594, measured twice) and HIDE's product
floor (reviewer's honest estimate 100,000) — and both were argued as irreducible by lanes that
were trying to reduce them.
