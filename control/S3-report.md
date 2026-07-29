# S3 report — HIDE core (C6 context + C7 agent)

## Verdict

**Target not reached.** Scope LOC **83,430 → 73,367** (Δ **−10,063**). Plan target **52,500**. Remaining gap **20,867**.

Gates that are green: blackbox **86/86** runnable, scope crates build, lib+integration tests **648 passed / 0 failed / 0 ignored**, generation audit introduces no new unearned LOC, workspace fixture still readable, `relocated=0`, `facade=0`.

Gates not met: **crate count still 4** (target 1–2), **scope LOC above 52,500**, full clean-room re-expression of the surviving core not finished, `files_over_1500_lines` still **27** (campaign start 26) because `memory_classes.rs` remains 2,533.

Git commits for this worktree could not be recorded from the sandboxed agent environment (`index.lock` / object DB **Operation not permitted** on the linked worktree git dir). Working tree changes are present on disk under `crates/hide-*` and `control/S3-*`.

## What changed (this rung)

### Eliminated (unprotected / unwired)

| Item | Approx LOC | Observable reason it is not protected |
|------|------------|----------------------------------------|
| `hide-core` `state.rs` capsule store | 1,353 | File header: **not wired into the host**; live checkpoints are `CheckpointStore` / event-log KV. No constitution behaviour requires a second capsule authority. |
| `hide-core` browser evidence/DOM/driver stack | 2,047 | Constitution only requires **browser_search declared** (BC-CONTEXT_OS-018). Zero external `hide_core::browser` callers. |
| Personalize training loops (RLEF, eval miner, meta-router, world, curate, kv_handoff) | 2,310 net | Not in HIDE_SESSION / CONTEXT_OS / AGENTS / SECURITY behaviour list. Capture **record + scrub-on-write store** kept for workspace open + personalization connector. |
| In-src `host_tests_0..3` (~4,284) | large | Tests are rewritable; replaced by compact `tests/host_slice_core.rs`. |

### Restructured (mechanical chunking removed)

- Deleted `host_ops_0.rs` … `host_ops_5.rs` and `host_tests_0..3` includes.
- Replaced with semantic `host_cmds/`:
  - `lifecycle.rs` — open, health, initialize
  - `intent_entry.rs` / `intent_handlers.rs` / `intent_effects.rs` — intent surface
  - `turn.rs` — turn build / generate
  - `tools_workspace.rs` — tools, policy, workspace, process, fleet
  - `verify_checkpoint.rs` — verify plane, diffs, goals, checkpoints
  - `jobs_memory.rs` — jobs, memory ledger, side chat
- This satisfies the accounting rule against **mechanical numbered chunking**. It is **not** a full clean-room densification of the method bodies; rewrite credit ≈ 0 after shared prelude cost.

### Unchanged crates in scope

- `hide-kernel` (19,995) — tooling, machine, program runtime, verify plane
- `hawking-context` (9,804) — compiler, classed memory, privacy

## Crate shape decision (not executed)

**Target: two crates**

1. **`context` (C6)** — today’s `hawking-context` (compile, budgets, classed memory, forget-as-deletion, privacy construction). Kept separate because forget/compile manifests are independently testable and used outside the host.
2. **`agent` (C7)** — merge `hide-backend` + `hide-kernel` + remaining `hide-core` runtime/contracts the host owns.

**Not one crate:** merging context into agent is a packaging option A2 allows later; semantic ownership should stay distinct for the forget and privacy type boundaries.

**Why merge did not land this rung:** `hide-core` is imported by ~50 files outside S3 scope (events, orch, index, research, protocol, serve, acp, fleet). Absorbing it requires coordinated dep rewires those slices own; doing it incompletely creates a dual package or a facade, both forbidden.

## Old → new behaviour map (protected slice)

| Behaviour | Old locus | New locus (this rung) | Status |
|-----------|-----------|----------------------|--------|
| YOU/CHAT/IDE one session | `surfaces.rs`, lenses | unchanged | kept |
| Handoff claim-only | `surfaces::handoff_*`, lenses capsule | unchanged | kept |
| Intent → ack | `host_ops_0::handle_intent` | `host_cmds/intent_entry.rs` | re-homed |
| Context compile | `host_support_1` + `hawking-context` | unchanged | kept |
| Single permission eval before tool | `evaluate_tool_policy` / `permission_verdict_for` | `host_cmds/tools_workspace.rs` + effects | re-homed |
| Tool dispatch | `dispatch_tool` | `host_cmds/tools_workspace.rs` | re-homed |
| Classed memory + forget deletion | `memory_classes.rs` | unchanged | kept |
| Privacy construction fail-closed | `privacy.rs` | unchanged | kept |
| Objects hash identity / derivative-only / dead-letter | `objects_*` | unchanged | kept |
| Connector constructible only local_folder+rss | `connector_abi_*` | unchanged | kept |
| Unwired state capsules | `state.rs` | **deleted** | released |
| Browser evidence stack | `browser_*` | **deleted** | released (declaration remains) |
| Personalize training loops | `personalize_*` | **deleted** (record/store remain) | released |

## Measurements

| Metric | Before | After |
|--------|--------|-------|
| Scope LOC (4 crates) | 83,430 | 73,367 |
| Combined active LOC | 440,886 | 425,418 |
| Hide subsystem | 107,129 | 91,661 |
| files >1500 | 27 | 27 |
| tiny forwarders | 13 | 13 |
| Owned crate count | 4 | 4 |
| Blackbox runnable | 86/86 | 86/86 |

Ledger: see `control/S3-ledger.json`. `relocated=0`, `facade=0`, `generated=0`.

## Irreducible remainder (the valuable list)

Everything still present that could not be removed without breaking an **observable** constitution or fixture behaviour. Each line names the behaviour that forbids deletion.

### Context OS (`hawking-context`, ~9.8k)

| Mass | Forbidding observable behaviour |
|------|----------------------------------|
| `memory_classes.rs` (~2.5k) | BC-CONTEXT_OS-001 six classes + write-authority type boundary; BC-SECURITY-011 / BC-SECURITY-020 forget is real deletion including dangling edges; pin/expire/disable; export must not resurrect |
| `compiler.rs` + `sources.rs` + `budget.rs` + `manifest.rs` | Context compile for a turn with class budgets and durable `context.compiled` shape (slice 3.2) |
| `privacy.rs` | BC-CONTEXT_OS-003/004 construction fail-closed; ephemeral end purges; BC-SECURITY-005 |
| `personal_tools.rs` | BC-CONTEXT_OS-012 typed effects registry |
| `memory.rs` (SQLite/FTS store) | Durable memory substrate used by compile/recall paths |
| kv/rot/capability/profiles/recall/embed/fidelity | Supporting contracts for compile/meter/rot and KV handoff seams used by host |

### Contracts + objects (`hide-core` remaining ~7.9k)

| Mass | Forbidding observable behaviour |
|------|----------------------------------|
| `event.rs` + persistence JSONL | One event log / ordering authority; workspace fixture `.hide/log/events.jsonl` |
| `api.rs` Intent/UiEvent | BC-HIDE_SESSION-002 intent path; wire-B projections |
| `ids.rs`, `types.rs`, `error.rs`, `config.rs` | Shared identity and config surface used across hide + hawking crates |
| `permission.rs` | Single permission evaluation contract (Allow/Ask/Deny) |
| `tool.rs` | Tool registry/dispatcher types tools must pass through |
| `objects_*` (~2.8k) | BC-CONTEXT_OS-005/006/013/014 content-hash identity, derivative-only compile view, dead-letter never silent-drop |
| `automation_*` (~1.8k) | BC-AGENTS-001..003,011 automation authority non-widening, kinds, stop/idempotency, durability |
| `project.rs` / persistence stores | Workspace fixture layout authority; session records |
| `runtime.rs`, `security.rs`, `observability.rs`, `plugin.rs`, `supervision.rs` | Runtime supervisor state, security defaults, health — consumed by host/serve |

### Kernel (`hide-kernel` ~20k)

| Mass | Forbidding observable behaviour |
|------|----------------------------------|
| `machine.rs` + session/plan/govern | Agent step loop plan→act→observe→verify for turns and fleet launcher |
| `tooling_fs/edit/shell/git/mcp/search/memory/proc` + `tools.rs` | Declared-effect tools; scope enforcement; permission-gated dispatch targets |
| `verify.rs` + `verify_plane.rs` | Deterministic verification plane receipts (host static analysis path) |
| `program_runtime_*` | Sandboxed programmatic tools: prepare write proposals only, no ambient authority |
| `extension_registry.rs` | Unified capability registry ABI (effects, scope, provenance) |
| `security_*` (redaction, sandbox, audit, storage) | Secret redaction on personalize store; sandbox refuse paths |
| `checkpoint/cooperate/projection/skills/subagent/search/runtime_client` | Kernel product seams used by host and fleet |

### Host / agent (`hide-backend` remaining ~35.7k)

| Mass | Forbidding observable behaviour |
|------|----------------------------------|
| `host_cmds/*` + `host_support_*` + `host.rs` | Full intent surface, dual-ish turn paths still both live (`run_turn_core` default + `run_turn_kernel`), policy, tools, memory intents, checkpoints, jobs |
| `surfaces.rs` + `lenses_*` | YOU/CHAT/IDE lenses; claim-only handoff; projects/swarm fixture modes (BC-AGENTS-004..012, BC-HIDE_SESSION-008/009) |
| `connector_abi_*` + `connectors.rs` | Constructible only local_folder+rss; declared families include browser_search; no ambient credentials; every write is effect with receipt |
| `services_*` | Workspace open, session registry, goals, jobs — fixture/workspace durability |
| `policy.rs` + `security.rs` | Policy decision records; config-driven rules |
| `rpc.rs` + `commands.rs` + `initialize.rs` | Protocol method surface used by hide-serve |
| `supervisor.rs` + `model_provider.rs` + `process.rs` | Runtime child + HTTP generate path |
| `replay/rewind/ui_bus/approval/interrupt/live_thread` | Time-travel, UI bus, approval gates, control intents |
| `classed_writers.rs` + `memory.rs` (host ledger) | Classed write caps; host memory ledger distinct from classed forget |
| `compat*` (~2.4k) | Repo instruction fold into compile (Claude-compat); not constitution-core but still load-bearing for `repo_instructions` on every turn |
| `program.rs` | Host handles for program runtime |
| `speculation_safety.rs` | BC-SECURITY-001/015/016/017 draft tokens cannot hit durable sinks |
| Remaining personalize capture | Workspace personalization store + connector list/append |

## Why the remaining ~20.9k to target was not taken

1. **Clean-room re-expression of kernel tooling + host turn paths** is the only way A2 budgets C7 at 45k. That is a multi-thousand-line rewrite with golden behavioural tests, not a second elimination pass.
2. **Dual turn path** (`run_turn_core` + `run_turn_kernel`) is still both reachable; collapsing to one authoritative path needs a product lock on which path is product (estimated 1–3k after matrix lock).
3. **Crate merge** to agent+context requires out-of-scope Cargo/use rewires (events, orch, serve, acp, fleet, protocol).
4. **`memory_classes.rs` >1500** needs densification or a semantic split that preserves forget/export properties without raising file count of mega-modules.
5. **Sandbox blocked git commits** in this worktree; controller should commit from an unsandboxed profile if desired.

## Recommended next commits (controller / next agent)

1. Collapse to **one turn path**; delete the other after blackbox+host tests agree.
2. Dense rewrite of `memory_classes.rs` (target <1500) preserving forget edge-clearing tests.
3. Merge `hide-kernel` → `hide-backend` modules; update fleet only.
4. Merge remaining `hide-core` into agent package; rewire external deps (separate coordinated commit).
5. Only then attempt true clean-room densification of tooling_* with property tests generated from the constitution.

## Definition-of-done checklist

| Requirement | Status |
|-------------|--------|
| New HIDE core provides in-scope behaviours | **Partial** — eliminations + re-home; not full clean-room |
| Old in-scope crates gone | **No** — still 4 crates |
| Blackbox 86/86 runnable | **Yes** |
| cargo build/test no new unit failures | **Yes** for lib+integration; doctests pre-existing fail |
| Generation audit no new unearned | **Yes** |
| Ledger reconciles to measured delta | **Yes** (see ledger) |
| Target 52,500 | **No** — stop with measured gap 20,867 |
