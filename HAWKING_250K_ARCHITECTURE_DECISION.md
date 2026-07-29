# Architecture decision — reconciling A1 and A2

Controller-authored, after both clean-room architectures were produced independently.
A1 is top-down from semantic authority. A2 is bottom-up from four traced vertical slices,
written by a lane that was instructed not to read A1 and confirmed it did not.

## They converge, and that is the finding

| | A1 | A2 |
|---|---|---|
| method | top-down: what single thing owns each decision | bottom-up: trace four slices, let boundaries fall where the traces couple |
| components | 6 cores | 9 components |
| estimate | ~166k, stated realistic band 200–230k | ~188k, stated band 180–230k |
| verdict on 250k | reachable, 200k stretch not claimed | reachable, 200k needs lab ≤30k and agent ≤40k |
| named top risk | generated executors turn out net-positive or lose bit-exactness | decomposing `gravity_glm_resident` device↔runtime without breaking TPS, dispatch count or bit-exact sampling |

Two independent derivations landing on the same band, and naming the *same* top risk from
opposite directions, is the strongest evidence this campaign has that 250,000 is a real
target rather than an aspiration. It is not proof. Both are estimates and the campaign
grades only code.

A2's nine components fold cleanly into A1's six cores, which is why the disagreement on
count is not a disagreement at all:

```
Core A  artifact/evidence   <- C1 artifact
Core B  model runtime       <- C2 device + C3 runtime
Core C  experiment          <- C5 lab
Core D  HIDE OS             <- C6 context + C7 agent + C9 surface
Core E  external systems    <- C4 serve + C8 contract
Core F  verification        <- (A2 folds this into glue; A1 makes it explicit)
```

The campaign permits at most six cores. Six it is, with A2's nine as the internal module
boundaries inside them. That is the decision.

## Where they disagree, A2 wins

Both disagreements are places where A1 asserted a generation win and A2 measured the
constraint that blocks it.

**Residency twins.** A1 proposed generating `gravity_glm.rs` (3,584, host-resident) and
`gravity_glm_resident.rs` (13,035, device-resident) from one forward spec, on the grounds
that the second file's own header says its discrete decisions use the same host arithmetic
so token identity is bit-exact against the first. A2, having traced the decode path, budgets
only a 20% compression on the resident file and names the reason: the boundary carries
1 command buffer and ~210 dispatches per token, and mis-drawing it costs the protected
numbers. **Adopt A2's 20%.** A1's generation idea survives as a hypothesis with a named
falsifier, not as a plan line.

**Shaders.** A1 proposed generating `quant.metal` (5,621) from a template plus the variant
table that defines the host-selected A/B lattice. A2 calls the 15,766 shader lines "mostly
irreducible without changing kernels (protected perf)". **Adopt A2's position for planning**
and test A1's separately — the test is cheap and self-contained: generate the existing
variants from a table, diff the generated `.metal` against the committed one byte-for-byte,
and measure amplification. If it reproduces exactly at ≥4x amplification the lever is real
and free; if it does not, the question is closed with evidence instead of opinion.

The pattern in both: A1 reasoned from shape, A2 measured from execution. On this codebase
execution wins, and the prior arc's rejected runtime-graph contract is the standing reminder
of what happens when it does not.

## The plan of record

A2's per-component targets, with A1's two generation hypotheses carried as falsifiable
side-experiments rather than as budget.

```
C1 artifact       4,000
C2 device        28,000
C3 runtime       32,000
C4 serve          4,500
C5 lab           35,000
C6 context        7,500
C7 agent         45,000
C8 contract       8,000
C9 surface       12,000
CLI/glue/verify  12,000
                -------
                188,000   plan centre; band 180k-230k
```

Against a gated baseline of 433,505 that is a reduction of roughly 245,000 lines. The two
components carrying most of it are C7 agent (80k → 45k) and C5 lab (70k → 35k), and both
are flagged by their own author as the least certain: C7's 40% depends on how much HIDE
surface the Behaviour Constitution retains, and C5's on elimination discipline holding
against the temptation to archive rather than delete.

## Build order

Ordered by reduction-per-risk, and by what unblocks what.

1. **C5 lab** — largest single block, lowest product risk, entirely process code whose
   *outcomes* are protected and whose *shape* is not. In flight as slice S2.
2. **C7 + C6 + C9, HIDE** — largest remaining block. Gated on the Behaviour Constitution
   settling which surfaces are protected, because that decides 45k versus 50k+.
3. **Verification** — tests rebuilt from the constitution. 47,440 test LOC is fully
   rewritable and the coverage that matters is behavioural, not test-function count.
4. **C8 + C4, contract and serve** — one schema authority generating the rest. The prior
   arc's Bridge review already found 5,376 lines available here and identified that the
   dual-source lockstep has no crate dependency either way, so one side can generate the
   other.
5. **C2 + C3, device and runtime** — last, because it is where the protected performance
   and numerical contracts live and it is the one place a wrong boundary is expensive rather
   than merely wasteful.

## Side-experiment 1, resolved: the shader lattice is half generable

A1 proposed generating `crates/hawking-core/shaders/quant.metal` (5,622 lines) from a
template plus the variant table that defines its host-selected A/B lattice. A2 called the
shaders "mostly irreducible without changing kernels (protected perf)". The test was cheap
and it has been run.

The file holds **66 kernels, 3,855 lines of kernel bodies** and about 1,767 lines of shared
headers and helpers. The names are exactly the combinatorial lattice A1 predicted —
`gemm_q4_k_{v3,v4}_{predec}_{pair}_{2r,3r,4r,8r}_{inline}_{f16s}_{nox,halfreg}`, crossed with
q3_k/q4_k/q6_k/q8_0, batched, mma and fused-swiglu.

Line-level similarity between variants that differ on exactly one axis:

```
axis                            pair                                    similarity
f16 scales      gemm_q4_k_v4_predec_pair        vs …_f16s                    0.911
f16 scales      gemm_q4_k_m_fused               vs …_f16                     0.898
predec          gemm_q4_k_m_batched_v3w         vs …_predec                  0.763
inline          gemm_q4_k_v4_predec_pair_2r     vs …_2r_inline               0.667
row blocking    gemm_q4_k_v4_predec_pair_2r     vs …_pair_3r                 0.386
```

**Both were partly right.** The dtype and predecode axes are near-pure parameterisation and
generate cleanly. The row-blocking axis does not: `2r` and `3r` share 39% of their lines
because they are hand-unrolled with different register allocation, and a generator for them
would have to encode the tuning rather than the pattern — at which point the generator is
the code and the amplification collapses.

So the lever is real but smaller than A1 assumed and larger than A2 allowed: on the order of
**1,500–1,900 lines** from the parameterisable axes, not 5,000 and not zero. That is inside
A2's C2 budget of 28,000 rather than an argument to lower it, so **the plan of record does
not change** — but the question is now closed with a measurement instead of two opinions,
and the row-blocking families are named as hand-written for a reason.

## Side-experiment 2, resolved: the residency twins are not twins

A1's headline claim was that `gravity_glm.rs` (3,584, host-resident) and
`gravity_glm_resident.rs` (13,035, device-resident) are one model's forward pass written
twice, held bit-identical by hand, and therefore generable from a single forward spec with
two residency strategies. The evidence for it was the resident file's own header, which says
its discrete decisions "use the same host arithmetic as `crate::gravity_glm::forward_impl`
so token identity is bit-exact against the host-state path".

**Refuted.**

```
gravity_glm fns                                     135
gravity_glm_resident fns                            204
function names present in both                        7
of those, pairs with >=0.75 line similarity           0
best shared pair   batched_mlp     30 vs 83 lines   0.336
                   indexer_topk    63 vs 124 lines  0.267

resident functions with no counterpart at all       197, totalling 9,992 lines
resident kernel/function bodies                     10,250 of 13,035 file lines
```

The two files share seven names and no implementation. The resident path is device buffer
arenas, replay plans, expert offset tables and command encoding — work that simply does not
exist in the host path. The header's bit-exactness claim is about **results**, not about
code: the two agree on the token they produce, which is a property that had to be
established precisely because the implementations differ.

So there is no 16,619-line twin to collapse. A2's 20% compression on the resident file —
extracting shared encode helpers and expert table builders — stands, and on this evidence it
is if anything generous.

## Both side-experiments point the same way

A1 was wrong twice, in the same direction: it reasoned from names, file headers and shape,
and in both cases the code turned out more different than the prose describing it. A2, which
traced execution, was closer both times.

That is worth stating plainly because it is the campaign's own thesis under test. The
instruction was to build the graph before designing, precisely so that structure would be
measured rather than assumed — and the two places where this controller assumed instead, it
was wrong. The plan of record stays A2's, and the remaining A1 contribution is the six-core
grouping, which A2's nine components fold into without strain.

**Consequence for Core B:** its reduction does not come from collapsing duplicated
executors, because there are none. It has to come from dead kernel variants, the
parameterisable shader axes measured above, shared encode helpers, and rewriting the test
surface. That is a thinner story than A1 told, and it is the true one.

## What would falsify the plan

- C5 lands above 45k with a named observable behaviour behind every remaining line. That
  would mean the laboratory's mass *is* protected science, and the campaign's binding
  contract is the product decision the prior arc already named.
- C7 lands above 60k because the constitution retains fleet, swarm and personalize surfaces.
- Either generation hypothesis fails its byte-diff or amplification test, which costs the
  plan nothing directly but removes the slack that covers what has been underestimated.
- The device↔runtime decomposition costs more than 2% of base or accelerated TPS, or more
  than one command buffer per token, or breaks bit-exact argmax. That is A2's Risk A and it
  is the boundary most likely to end the descent.
