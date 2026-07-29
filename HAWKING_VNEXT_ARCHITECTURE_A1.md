# vNext Architecture A1 — the six-core recomposition

Controller-authored. This is **architecture 1 of the two the floor rule requires**. It is
derived top-down from semantic authority: what single thing owns each decision. A second,
independent architecture (A2) is derived bottom-up from the vertical slices and is written
separately, by a lane that does not read this file.

A1 is a hypothesis until the semantic graph validates or refutes its cluster boundaries.
Where the graph disagrees with a boundary drawn here, the graph wins and this file is
amended with the disagreement recorded.

## The premise

The tree is 430,175 active LOC. A prior arc sealed 396,468 of that as "at floor" across three
subsystems. Every one of those refusals rests on something this campaign declassifies:
public crate API, self-tested product schemas, code protected despite zero callers,
`cfg(test)` surface, a host-selected tuning lattice, and a crate manifest boundary. Under
the new rules none of them is a behaviour. So the at-floor seal is void and the whole tree
is in scope.

What the destructive probe measured still stands as *evidence*: with every protection
deleted the current architecture stops at 349,455. That is not a floor on the rebuild — it
is a measurement of how much of the current tree is load-bearing *for the current
architecture*. A different architecture has a different number.

## The one observation that drives everything

Three families of repetition account for most of the mass, and each is the same mistake:
**a decision that should live in a table lives in code, once per case.**

| family | today | the repeated decision |
|---|---|---|
| residency twins | `gravity_glm.rs` 3,584 + `gravity_glm_resident.rs` 13,035 = 16,619 | one model's forward pass written twice, host-resident and device-resident, held bit-identical by hand |
| kernel lattice | `kernels/` 12,219 + `shaders/` 15,766 + `metal/` 5,012 = 32,997 | a host-selected A/B permutation space, written out by hand instead of generated from the variant table that defines it |
| campaign controllers | ~44 controllers across `tools/condense`, `tools/foundry`, `tools/campaign`, `tools/prometheus`, `ramanujan`, `odyssey` | declare, acquire, measure, allocate, pack, verify, receipt, promote — re-typed per experiment, including its own argparse, resource policy, status file, ledger append, receipt shape and resume logic |

### Correction, recorded rather than quietly fixed

A1's first draft claimed the four `gravity*.rs` files were four implementations of one
container and codec, worth 20,661 lines. **That was wrong**, and the check that caught it
took two minutes: function-name Jaccard across the four is 0.007–0.054 and type-name
Jaccard is 0.000, and reading their headers shows why. `gravity.rs` is the container reader
and PQ tensor codec — one thing, correctly one file. `gravity_glm.rs` and `gravity_llama.rs`
are *forward passes* for two model families, each graded against its own numpy oracle.
Those belong to Core B, not Core A, and the two families are genuinely distinct.

Name overlap is weak evidence and does not settle it — the same instrument would miss two
codecs that agree in structure while disagreeing in every identifier, which is exactly what
G2's control-flow-signature clone analysis exists to catch. So the four-codec claim is
*withdrawn pending the graph*, not disproved.

What replaced it is better evidenced. `gravity_glm.rs` (host-resident state) and
`gravity_glm_resident.rs` (device-resident state) are **the same model's forward pass,
written twice**. The second file's own header says so: discrete decisions "still use the
same host arithmetic as `crate::gravity_glm::forward_impl` so token identity is bit-exact
against the host-state path". Sixteen thousand lines are held in agreement by hand, and the
only axis of variation is where a tensor lives. That is a compile-time specialization target
in the precise sense of campaign section 7.3 — one forward spec, two residency strategies,
both generated — and the bit-exactness that is protected becomes a property of the generator
rather than of a human keeping two files in step.

Plus one structural repetition that is not a family but is the largest single count: roughly
two hundred root-level `*_RECEIPT.json` / `*_LEDGER.jsonl` / `*_VERDICT.json` shapes, each
produced by bespoke code. There is one receipt concept here, instantiated two hundred times.

Campaign section 7.2 is the instruction and section 7.3 is the escape hatch for the places
where a runtime abstraction would cost speed. This architecture applies both, under the
amplification and reproducibility rules in `control/REBUILD_ACCOUNTING_RULES.json` — because
generation is also the easiest way to fake this number, and the rules are what separate the
two.

## What is explicitly NOT retried

The prior arc's runtime-graph programme is a **rejected contract**, and this architecture
does not reopen it. WKV-7's fixed per-head S state and token-shift cannot share an
*executable* LayerOp core with a transformer's growing KV + RoPE + MHA while staying
bit-exact on argmax and top-k, holding one command buffer per token, and being net-negative.
Six irreducible operators, three per family, were named.

A1 does something different in kind. There is no shared runtime core, so there is nothing
to be generic over at execution time. The shared thing is a **spec language and a
generator**; each family's executor is generated, distinct, and takes exactly its own path.
Bit-exactness is preserved because no code is shared at run time. This is campaign section
7.3 applied literally, and it is untried — the rejected result does not bear on it.

If the generated executors turn out net-positive in LOC, or lose bit-exactness, or add a
command buffer, that is A1's first failing contract and it gets recorded as such rather than
argued around.

## Core A — Artifact and Evidence

Owns: source identity, hashes and manifests, artifact container I/O, receipts, capability
gates, checkpoint and resume, rollback and release.

- **One container.** Header, manifest, hash chain, section index. Model-family agnostic.
- **One codec surface**, specialized per family from a family table at compile time. The
  family table is the reviewed artifact; the four codec bodies stop existing.
- **One receipt authority.** Every receipt in the tree is the same typed record —
  `{schema, commit, inputs, method, measurement, verdict}` — over a schema registry.
  A receipt kind becomes a registry entry, not a module.
- **One capability gate.** The `SUBSTRATE_CAPABILITY.json` mechanism (artifact-hash-keyed,
  independent of the fence, unlisted artifact means refused) is a security outcome and is
  preserved exactly. Its *implementation* is not.

Replaces: `gravity*.rs`, `tq.rs`, `gguf.rs`, `sidecar.rs`, `mixed_quant_store.rs`,
`cost_ledger.rs`, `quant*.rs`, `profile.rs`, and the receipt/ledger halves of the laboratory.

## Core B — Model Runtime

Owns: execution graph, operator registry, model families, KV and state, sampling,
base and accelerated providers, CPU and Metal backends.

- **One model-family spec language.** A family declares its operators, its state shape, its
  dtype lattice and its numeric contract. The generator emits that family's executor.
- **One kernel variant table.** The A/B permutation lattice is data. `quant.metal` becomes
  a template plus a variant table; the 5,621 lines are generated output, and what is counted
  is the template and the table.
- **One provider registry.** CPU, Metal, and Fabric are providers behind one selection
  policy, not three parallel stacks.
- Sampling, KV and tokenizer are single authorities already in spirit; they become so in
  fact.

## Core C — Experiment and Transformation

Owns: experiment IR, operators, datasets and evidence, resource scheduler,
measure/allocate/pack, comparison and promotion, status and receipts.

This is the largest reduction and the least risky, because almost none of it is observable
behaviour — it is scientific *process*, and the protected part is the semantics of the
result, not the shape of the runner.

- **A campaign becomes a spec, not a module.** One declarative experiment IR:
  preregistration, source admission, measurement plan, allocation policy, pack program,
  verification gates, promotion rule, burial rule.
- **One engine** executes any spec: scheduling, resource policy, checkpointing, resume,
  ledger append, status derivation, receipt emission.
- **One governance authority** — the hash-chained append-only ledger, tier promotion
  refusal, author-is-not-admitter tribunal rule, and burial-is-not-deletion, all of which
  are protected scientific semantics and all of which are currently implemented twice.

Preserved exactly, because they are the science: refusal semantics, the Escape Receipt gate,
the capability gate, replay determinism, and the rule that a measurement on synthetic
activations is not a measurement.

## Core D — HIDE Operating System

Owns: session and event state, Context OS, memory, tools, effects and permissions, agents
and worktrees, objects, connectors, automations, and the YOU/CHAT/IDE lenses.

The invariant that survives untouched: **YOU, CHAT and IDE are three lenses on one session**,
and a handoff capsule carries a claim, never a capability. Safety lives at type boundaries —
connectors that are declared-unconstructible, job capabilities whose fields are private so
there is no widening path, objects whose hash is their identity, a compile view that exposes
derivatives rather than raw bytes, a dead-letter path that never drops, and a graph forget
that is real deletion including dangling edges. Those are security outcomes and they are
protected. The twenty crates that currently express them are not.

- **One session state machine**, one event log, one permission evaluation point.
- **One host command registry.** `host_ops_0..4`, `host_tests_0..3`, `host_support_0..1` is
  mechanical chunking of a single command surface; it becomes a registry plus generated
  dispatch.
- **One memory model** across the declared classes, with the forget path as the only
  deletion authority.

## Core E — External Systems

Owns: Bridge, Fabric, CLI, SDK and protocol generation, model vault.

One event authority and one schema authority, from which are generated: Rust, Python and
TypeScript types, the CLI surface, Bridge schemas, HIDE bindings, Fabric metadata, and docs.
The prior arc's Bridge review already found 5,376 lines available here and identified that
the dual-source `bridge_surface` lockstep has no crate dependency either way, so one side
can generate the other. A1 takes that finding as its starting point rather than re-deriving it.

## Core F — Verification

Owns: black-box behaviour tests, numeric and discrete parity, security and adversarial
tests, migration, performance, offline recovery.

Tests are generated from `REBUILD_BEHAVIOUR_CONSTITUTION.json` wherever the behaviour is
table-shaped, and hand-written wherever it is not. The count that matters is behaviour
coverage, never test-function count. The anti-gaming rule is load-bearing and comes from
this repo's own history: **a check that skips is not a check that passes**, and
`tools/verify/blackbox.py` exits non-zero when a previously runnable check becomes
unrunnable.

## Crate budget

Target is 12. A1 spends 10 and holds 2 in reserve.

```
hawking-evidence     Core A
hawking-runtime      Core B, hand-written
hawking-kernels      Core B, generated from spec + variant table
hawking-lab          Core C, thin Rust surface; the bulk is Python
hide-os              Core D, session/context/memory/permission/tools
hide-surface         Core D, lenses + app-facing surface
hawking-bridge       Core E, schema/event authority + generation
hawking-cli          Core E, the binary
hawking-verify       Core F
hawking-contracts    shared types, generated from the one schema source
```

Python packages: `lab/` (Core C) and `verify/` (Core F). TypeScript: one generated client.

## Estimated landing

Honest estimates, stated as estimates, to be replaced by measurements:

```
Core A    ~18k     from 20,661 codec + receipt/ledger apparatus
Core B    ~22k     from 52,976 model + kernels + shaders + metal
Core C    ~34k     from ~129,363 laboratory
Core D    ~45k     from ~119,000 HIDE + app + context
Core E    ~14k     from ~29,000
Core F    ~25k     from ~47,440 test LOC
shared     ~8k
          -----
          ~166k
```

That estimate is deliberately optimistic about Core C and D and deliberately does not
include what A1 has underestimated, which on this codebase's history is always more than
zero. The realistic band is 200k–230k, which clears the 250k primary target with room and
puts the 225k preferred target in range. 200k stretch is not claimed.

The number that decides this is not in this document. It is in the graph.

## What would falsify A1

- The graph's communities do not correspond to the six cores — meaning the semantic
  subsystems are cut differently from how authority is cut, and A1's boundaries are
  invented rather than found.
- Generated executors are net-positive in LOC, or lose bit-exactness, or cost a command
  buffer. That is section 7.3 failing on this codebase and it kills the Core B plan.
- The receipt families turn out to be genuinely distinct rather than two hundred instances
  of one shape.
- Behaviour coverage analysis finds that the laboratory's mass *is* observable behaviour
  under the constitution, in which case Core C's reduction is not available and the
  campaign's binding contract is the product decision the prior arc already named.
