# HAWKING vNext Architecture A2

**Authority:** independent clean-room design, bottom-up from four vertical slices  
**Date:** 2026-07-29  
**Tree measured at:** working tree rev via `tools/loc/hawking_loc.py`  
**Constraint:** did not read `HAWKING_VNEXT_ARCHITECTURE_A1.md` or `control/RECOMP-SCOUT-COMPARISON.json`  
**Binding rules:** `control/REBUILD_ACCOUNTING_RULES.json`  
**Companion map:** `control/A2-component-map.json`

---

## 0. Method

Entry points first, not directories. For each campaign slice:

1. Find the product entry (CLI flag, HTTP route, Python `__main__`, codegen binary).
2. Walk the live call graph far enough to name every component that mutates state, performs I/O, or can fail the slice.
3. Record file paths and line numbers so a second person can re-walk the path without this prose.

Then: ask what **minimum set of components** can carry all four traces. Boundaries are justified only by a slice coupling (shared state, shared lifecycle, shared failure policy, or shared performance budget)—not by taxonomy or a preferred component count.

**Baseline (measured, `hawking_loc.py`):**

| Dimension | Value |
|---|---|
| Combined active LOC | **431,120** |
| Rust | 247,204 |
| Python | 126,163 |
| TypeScript | 15,678 |
| Shader | 15,766 |
| Shell | 3,277 |
| Markdown (counted lean) | 23,010 |
| Subsystem hawking / hide / laboratory / shared | 170,792 / 107,129 / 130,068 / 23,131 |
| Topology: source files / public symbols / functions | 1,197 / 9,523 / 14,660 |
| Files >1500 lines | 26 |
| Workspace crates | 20 members (Cargo.toml) |

Targets at 250k: ≤60 directories, ≤450 source files, ≤12 workspace crates, ≤4,000 public symbols, ≤7,500 functions.

---

## 1. Slice traces

### Slice 1 — `.gravity` artifact → runtime → server → one HIDE turn

**Intent:** load a sealed Gravity artifact, serve tokens over HTTP, and have HIDE complete one user turn against that serve.

#### 1.1 Entry

| Step | Location | What happens |
|---|---|---|
| CLI parse | `crates/hawking/src/main.rs:709–913` (`Cmd::Serve`) | `--gravity` / `HAWKING_GRAVITY` optional; otherwise `--weights` required. |
| Resolve shard | `main.rs:747–758` | `GravityEngine::resolve_entry(&gpath)` on macOS only; non-macOS hard-errors. |
| Hand to serve | `main.rs:894–913` | `hawking_serve::run(ServeOptions { weights: resolved, request_timeout_secs: gravity_request_timeout_secs, … })`. |

#### 1.2 Artifact open + engine load

| Step | Location | State / writes / failures |
|---|---|---|
| Serve bootstrap | `crates/hawking-serve/src/lib.rs:433+` (`run`) | Sets env levers (`HAWKING_QWEN_*`, profile plan). Builds `EngineConfig`. |
| Dispatch load | `lib.rs:547` → `hawking_core::model::load_engine` | `crates/hawking-core/src/model/mod.rs:27–84` |
| Container detect | `mod.rs:68–73` | `GravityEngine::is_gravity` (magic `GRAVITY\0`) or `is_activation_aware` (`.aap` + index). |
| Load body | `crates/hawking-core/src/model/gravity_engine.rs:206–359` | Opens shard, reads index, hashes index, loads tokenizer + chat template, opens Metal, dispatches family. |

Concrete load sub-steps (`gravity_engine.rs`):

1. **Magic / format** — `is_gravity` reads 8 bytes (`:68–75`); fail if neither gravity nor activation-aware.
2. **Shard mmap** — `GravityShard::open` in `crates/hawking-core/src/gravity.rs:68+` (header JSON, tensor table, integrity on `read_tensor`).
3. **Index + artifact hash** — parent `model.gravity.index.json` or `model.activation_aware.index.json` (`:222–249`); `index_sha256 = SHA-256(index_bytes)` (`:240–244`). **Protected outcome:** capability and health surfaces key off this hash.
4. **Architecture** — `architecture.model_type` from index (`:260–265`); currently wires `"llama"` and `"glm_moe_dsa"` only (`:331–348`).
5. **Tokenizer** — path from shard header or `tokenizer/tokenizer.json` beside shards (`:272–309`); **fail closed** if missing (refuses to serve raw token ids).
6. **Chat template** — `tokenizer/chat_template.jinja` (`:313–327`); **glm_moe_dsa refuses load without it** (protects against fluent garbage).
7. **Metal context** — `MetalContext::new_with_trace` (`:330`).
8. **Family GPU model** — `GravityLlamaGpu::open_with` or `GravityGlmGpu::open_dir_with` (`:331–342`). Resident GLM path lives in `gravity_glm_resident.rs` (**13,035 LOC measured**).

**State touched:** mmap of shard bytes (read-only), GPU buffers for weights/KV, host tokenizer state, optional index hash string.

**Writes:** none to the artifact; process-local GPU/host memory only.

**Fail modes:** bad magic, missing index, dual indexes, missing tokenizer/template, unknown arch, Metal init failure, memory budget (`mod.rs:44–62` for GGUF path; gravity path inherits Metal OOM later).

#### 1.3 Token generation

| Step | Location | Notes |
|---|---|---|
| `Engine::generate` | `gravity_engine.rs:362–431` | encode → `forward` prefill → loop sample → `forward_at` decode → SSE sink. |
| Sample | `crates/hawking-core/src/sample.rs` + GPU sample shaders | Seeded `Sampler`; **bit-exact argmax/top-k protected**. |
| Decode step | `model.forward_at(&[next], pos)` | One token; cost-ledger hooks in `cost_ledger.rs` attribute **command buffers / dispatches** (production path holds **1 CB ~210 dispatches/token** — protected). |

Explicit non-claim (`gravity_engine.rs:457–461`): Gravity does **not** implement continuous-batch prefill slots; serve falls back to single-stream `generate`. Claiming multi-seq would be a facade.

#### 1.4 HTTP serve

| Step | Location | Notes |
|---|---|---|
| Gravity meta | `hawking-serve/src/lib.rs:558–607` | Populates `GravityServeMeta` (index sha, arch, chat template, timeouts). |
| Router | `hawking-serve/src/http.rs:183+` | `/v1/chat/completions`, `/v1/hawking/generate`, `/healthz`, `/metrics`, `/v1/models`. |
| Chat path | `http.rs:524–640` | Parses body; for gravity GLM uses artifact template via `glm_chat::render_glm_chat`; tools rejected for `glm_moe_dsa` today (`:541–544`). |
| Stream | SSE token channel from engine lock | Failures: timeout (gravity default raised), abort, sample/stop. |

#### 1.5 One HIDE turn against that serve

| Step | Location | Notes |
|---|---|---|
| HIDE HTTP | `crates/hide-serve/src/main.rs:17–28` | `BackendHost::open_workspace` → `hide_serve::router`. |
| Intent route | `hide-serve/src/lib.rs:44,111–115` | `POST /v1/hide/intent` → `host.handle_intent`. |
| Accept turn | `hide-backend/src/host_ops_0.rs:237–627` | `Intent::SubmitTurn {…}`; only **accepted** turns spawn generation. |
| Spawn | `host_ops_2.rs:61–131` | Agentic kernel path or single-shot fallback. |
| Context + kernel | `host_support_1.rs:77–259` (`run_turn_kernel`) | Compiles context (Slice 3), then `AgentKernel::start_run` / `step`. |
| Model call | `model_provider.rs:1–20,391–435` | **HTTP only** to supervised `hawking serve`: preferred `POST /v1/hawking/generate` SSE; no Rust dep on hawking-core (Cargo isolation is not protected, but the **process boundary is load-bearing for ops/security**). |

**Slice 1 end-to-end state writes (HIDE side):** event log (`context.compiled`, tool/plan events), UI bus events, working-memory turn guard, optional project-brain upsert. **Inference side:** none durable.

**Where it can fail after serve is up:** serve down / wrong base URL (`RuntimeUnavailable`), permission deny on tools mid-turn, context compile error, kernel max_steps, SSE parse failure.

---

### Slice 2 — experiment specification → execute → checkpoint → receipt → resume

**Intent:** one declarative experiment runs, survives process death, seals a receipt, and resumes without redoing completed steps.

#### 2.1 Entry

| Step | Location | What happens |
|---|---|---|
| Engine package | `tools/condense/engine/__init__.py` | Declares authority for lifecycle verbs: precheck, measure, allocate, pack, seal, monitor, resume, report. |
| CLI | `tools/condense/engine/runtime.py:398+` (`main`) / `run_campaign` (`:378`) | Loads `ExperimentSpec`, constructs `CampaignRuntime`. |
| Spec | `tools/condense/engine/spec.py:159+` (`ExperimentSpec`) | `campaign_id`, phases, steps, resource class, authorization fences, reopen conditions. `load_spec` / `validate_spec` (`:215–317`). |

#### 2.2 Execute

| Step | Location | Notes |
|---|---|---|
| Runtime open | `runtime.py:161–182` | Singleton lease (`lease.py`), resource governor, operator registry, scheduler, checkpoint store, receipt store, state machine. |
| Run loop | `runtime.py:244–347` | For each phase/step: skip if `StateMachine.is_step_done`, else `run_handler`, mark step, checkpoint. |
| Handlers | `runtime.py:48–115` builtins + registered campaign handlers | Heavy work (pack, measure Metal) is **registered operators**, not new state machines. |
| Operators | `tools/condense/engine/operators.py` | Classifies modules; campaign-specific code in `tools/condense/glm52_*.py`, `gravity_*.py` registers here. |
| State machine | `state_machine.py:83–178` | Legal transitions only; `IllegalTransition` aborts. Snapshot/from_snapshot for resume. |

**State touched:** work root under campaign id; operator side effects (artifact shards, measurements, logs).

**Fail modes:** lease conflict, illegal transition, missing operator, fence forced open (`_handle_precheck_fences`), resource governor reject, handler exception.

#### 2.3 Checkpoint

| Step | Location | Notes |
|---|---|---|
| Store | `tools/condense/engine/checkpoint.py:148–237` | Atomic JSON + `seal_sha256` over unsigned body; hash-chained `events.jsonl`. |
| Save | `:175–190` | Writes `checkpoint.json` with `event_head`, `event_count`, `state`. |
| Load integrity | `:192–227` | Seal mismatch, campaign_id mismatch, split-brain (count ahead of log / head mismatch) → hard error. |
| Resume state | `:229+` | Empty initial state if no checkpoint. |

#### 2.4 Receipt

| Step | Location | Notes |
|---|---|---|
| Seal API | `tools/condense/engine/receipts.py:34–42` | `seal_receipt` / `verify_receipt` over canonical JSON. |
| Store | `ReceiptStore.write/read` (`:93–119`) | Path keyed by `campaign_id`. |
| Runtime emit | `runtime.py` run completion | `RunResult.receipt_path` populated when sealed. |

#### 2.5 Resume

Resume re-enters `CampaignRuntime` with same `campaign_id` root → `CheckpointStore.resume_state` + `StateMachine.from_snapshot` → scheduler skips completed steps → continues phases.

**Parallel historical systems (same slice shape, separate code):**

| Instance | Path | Role |
|---|---|---|
| Condense engine | `tools/condense/engine/*` | Intended single controller |
| Odyssey | `tools/odyssey/checkpoints.py`, `receipts.py` | Second checkpoint+receipt stack |
| Foundry lab | `tools/foundry/lab_harness/receipt.py` | Third receipt writer |
| GLM52 scripts | `tools/condense/glm52_*.py` | Often hand-roll receipts beside the engine |

These are **repetition**, not additional product requirements (see §4).

**Coupling to Slice 1:** pack operators write `.gravity` shards + `model.gravity.index.json` that Slice 1 loads; parity/measure operators call Metal paths. Lab must speak artifact bytes and may call into **device/runtime** for measurement, but must not own the serve or HIDE turn loop.

---

### Slice 3 — YOU request → Context OS → permission → tool → memory → handoff to CHAT/IDE

**Intent:** a YOU-surface user turn compiles context, gates tools, mutates memory, and can hand off a **claim capsule** (not a capability copy) to CHAT or IDE.

#### 3.1 YOU request entry

| Step | Location | Notes |
|---|---|---|
| FE → HTTP | `app/src/*` (TypeScript, **15,714 LOC measured** under `app/src`) | Posts intents; listens on `/v1/hide/events`. |
| Intent | `hide-core/src/api.rs:9` `Intent::SubmitTurn` | Also surface switch / handoff intents. |
| Host | `host_ops_0.rs:237` `handle_intent` | Records intent; effects only if accepted. |

Surface graph doctrine: `hide-backend/src/surfaces.rs:1–8` — YOU/CHAT/IDE are **lenses on one session**; handoff carries a **CLAIM**, never a capability grant.

#### 3.2 Context OS

| Step | Location | Notes |
|---|---|---|
| Compile | `host_support_1.rs:129–179` | `ContextCompiler` + `TokenCounter::discover_from_env` / heuristic. |
| Sources | same | `CodeIndexContextSource`, `ClassedMemoryContextSource` (per-class budgets), optional repo instructions. |
| Substrate | `crates/hawking-context/src/{compiler,sources,budget,manifest,memory,memory_classes}.rs` | Owns packing, manifests, classed memory. |
| Seal | `host_support_2.rs:60–97` `seal_compiled_manifest` | Capability, rot, meter attached; durable `context.compiled` event (`host_support_1.rs:234–245`). |

**State:** compiled prompt string + manifest; working-memory turn guard (`WorkingTurnGuard`) clears on all exits (`host_support_1.rs:118–127`).

**Fail modes:** role registry miss, compile error, tokenizer discovery optional (heuristic falls back).

#### 3.3 Permission

| Step | Location | Notes |
|---|---|---|
| Engine trait | `hide-core/src/permission.rs:107–180` | `PermissionEngine::evaluate` → Allow/Ask/Deny (+ risk). |
| Host policy | `host_ops_2.rs:955–1005` `evaluate_tool_policy` / `permission_verdict_for` | Builds `PermissionRequest`, records durable `policy.decision`. |
| Derive | `hide-backend/src/policy.rs:122+` | Maps engine verdict + declared tool effects → `PolicyDecision` (Allow/Ask/Deny/sandbox/reviewer…). |
| Config | `hide-backend/src/security.rs:22–70` | Policy rules from `HideConfig`. |

**Protected outcome:** denied tools do not execute; decisions are recorded. Implementation of the engine is not protected.

#### 3.4 Tool

| Step | Location | Notes |
|---|---|---|
| Dispatch | `host_ops_2.rs:439–445` `dispatch_tool` | Attributes session/run; goes through tool dispatcher. |
| Registry | `hide-backend/src/tools.rs` + `hide-kernel/src/tooling_*.rs` | FS, edit, shell, git, MCP, search, memory, proc. |
| Kernel loop | `hide-kernel/src/machine.rs` + `host_support_1` step loop | Plan → act (tool) → observe → verify. |

**Fail modes:** PolicyDenied, tool I/O error, sandbox refuse, approval required and not granted.

#### 3.5 Memory

| Step | Location | Notes |
|---|---|---|
| Host memory intents | `host_ops_1.rs:339–387` | `memory_add` / `supersede` / `record_outcome` / `revalidate`. |
| Classed store | `hawking-context/src/memory_classes.rs` | Multi-class store used in compile. |
| **Forget = real deletion** | `memory_classes.rs:1291–1298, 755` | Documented and tested: not a tombstone; export must not resurrect. |
| Ledger | `hide-backend/src/memory.rs` | Host-facing memory ledger (supersede preserves history for that ledger's model—distinct from classed forget). |

**Protected outcome:** user forget removes the row. Supersede semantics on the host ledger are a separate product behaviour (history preserved)—do not conflate with forget.

#### 3.6 Handoff YOU → CHAT / IDE

| Step | Location | Notes |
|---|---|---|
| Create | `surfaces.rs:110–171` `handoff_create` | Seals `HandoffCapsule` with claims + deliberate exclusions; emits `you.handoff.created`. |
| Receive | `surfaces.rs:186+` `handoff_receive` | Target surface receives capsule; tests assert CHAT capability is **not widened** (`:329–352`). |
| Graph | `hide-backend/src/lenses_*` + `SurfaceGraph` | One session graph; active surface switch. |

**State writes:** capsule in graph; events; UI badges. **Must not write:** expanded grants on the receiving surface.

---

### Slice 4 — model registry → adapter → Bridge → Fabric declaration

**Intent:** every model family is a registered declaration; Bridge surfaces (capabilities, events, schemas) and Fabric placement declarations are **generated from that registry**, not hand-maintained.

#### 4.1 Registry + adapters

| Step | Location | Notes |
|---|---|---|
| Trait | `hawking-adapters/src/abi.rs:316–323` `FamilyAdapter` | `descriptor() → FamilyDescriptor` (ABI fields, evidence, support level). |
| Families | `hawking-adapters/src/families/*.rs` | Ten family modules, **~72–120 LOC each**, same shape: aliases, evidence table, `FamilyAbi { … fabric_partition_boundaries … }`. |
| Registry | `registry.rs:11–91` `FamilyRegistry` / `builtin_registry` | Register, get, validate evidence + ABI completeness. |

#### 4.2 Bridge (what the slice actually hits)

"Bridge" in the live tree is **not** primarily `hawking-research/src/bridge.rs` (that file is research↔issue/memory helpers for bible ch.08). The product Bridge on this slice is:

| Piece | Location | Role |
|---|---|---|
| Codegen bridge surface | `hawking-adapters/src/generate.rs:1–8, 912+` | Emits Bridge endpoint status types, runtime capabilities, SDK types, HIDE capabilities, schemas. |
| Event adapters | `hawking-events/src/adapters/*` | Project legacy models (`UiEvent`, `StreamEvent`, seed FSM, protocol `Item`) into canonical events. |
| Canonical envelope | `hawking-events/src/envelope.rs` | `ProducingSurface` includes Bridge; `Subsystem` includes Bridge/Fabric/Adapter. |

Generate note (`generate.rs:624`): `"generated_from": "… FamilyRegistry + bridge surface + runtime profiles"`.

#### 4.3 Fabric declaration

| Step | Location | Notes |
|---|---|---|
| Emit | `generate.rs:90–91, 1064–1092` `fabric_declarations_json` | Schema `hawking.fabric.declarations.v1`: event categories + per-family placement + `fabric_partition` from ABI. |
| Artifact | `crates/hawking-adapters/generated/fabric_declarations.json` | Checked-in golden; drift test fails on diff. |
| Schema | `generate.rs:446+` `fabric_placement.schema.json` | Event kind `fabric.placement`. |
| Consumer (optional multi-node) | `hide-fleet/src/fabric_placement.rs` etc. | Placement simulator / pipeline; **declaration note says implementation is a parallel lane**. Slice 4 product requirement stops at **declaration correctness**, not multi-node execution. |

**State:** none at runtime for declarations (pure data + codegen). **Fail modes:** incomplete ABI validation, evidence grade insufficient for support level, golden drift.

**Capability-by-hash (adjacent, Slice 1/2 science):** `hawking-seed-c/src/gravity.rs` `CapabilityProof { artifact_index_sha256, rate, g_math, g_live }` and `decide` refuse cross-artifact or cross-rate proofs (`:66–180` region). This is the scientific admission gate keyed by artifact hash—protected outcome, currently in seed-c rather than the adapter crate.

---

## 2. Component set the traces imply

Nine components. Not six. Count fell out of couplings: every boundary either (a) is crossed by a named slice step with a different lifecycle/failure policy, or (b) would force a protected hot path to know about an unrelated surface.

| # | Component | Owns | Slices | Boundary (must know) | Must NOT know |
|---|---|---|---|---|---|
| C1 | **artifact** | `.gravity` / activation-aware container framing, tensor table, index JSON schema, index SHA-256, PQ/codec decode contracts shared with packers | S1 load; S2 pack/seal; S4 hash-keyed proofs | Byte layout, integrity, self-describing tokenizer/template pointers | Metal queues, HIDE intents, HTTP, family marketing names |
| C2 | **device** | `MetalContext`, shader embed/compile, kernel dispatch APIs, quant primitives used at encode time | S1 decode; S2 measure | One command buffer lifecycle per token, dispatch counts, buffer lifetimes | Chat templates, tools, experiment phases, Fabric topology |
| C3 | **runtime** | `Engine` trait, `load_engine`, Gravity/GGUF family forwards, tokenizer, sampler, KV, cost ledger hooks | S1 generate; S2 parity TPS | Logits and GenStats only | HIDE, axum routes, experiment checkpoints |
| C4 | **serve** | OpenAI + native HTTP, SSE, slots, gravity serve meta/timeouts, chat template render at the wire | S1 HTTP; S3 model provider client | HTTP request lifecycle | Context compiler internals, tool policy, Fabric placement |
| C5 | **lab** | `ExperimentSpec`, state machine, checkpoint, receipt, lease, governor, operator registry, forge/pack operators | S2 entire; produces C1 inputs | Campaign work roots, sealed receipts | HIDE session graph, live serve batching |
| C6 | **context** | Context compiler, budgets, sources, classed memory, forget-as-deletion, rot/meter | S3 compile/memory | Manifest + packed prompt | Metal, gravity codecs, Fabric nodes |
| C7 | **agent** | Host, kernel machine, tools, permission application, connectors (declared-unconstructible), surfaces/handoff, supervisor of serve child | S3 entire; S1 last mile | Session event log, policy decisions, handoff capsules | Shader sources, experiment state machines, family ABI tables |
| C8 | **contract** | Wire types (`Intent`/`UiEvent`), canonical events, family registry + adapter descriptors, Bridge codegen inputs, Fabric **declarations**, capability-proof policy (seed gravity decide) | S3 protocol; S4 entire; S1 hash field names | Pure data + pure functions + generators | Live GPU encode, mutable session stores |
| C9 | **surface** | YOU/CHAT/IDE UI, intent client, event render | S3 UX | Presentation + IPC to hide-serve | Any Rust internal type except generated SDK types from C8 |

### Why not fewer?

- Merging **device** into **runtime** recreates today's `hawking-core` monolith and re-couples shader edit cycles to model-family edit cycles. S1's performance contract is owned by device; S1's numerical contract (argmax) is owned by runtime sampling + device kernels together but the **dispatch budget** must be reviewable without reading GLM resident code.
- Merging **lab** into **runtime** couples Python experiment I/O to the Metal token path and reintroduces controller sprawl Slice 2 already tried to kill.
- Merging **context** into **agent** is tempting (only S3 uses it) but the forget/deletion and compile manifests are independently testable and are imported by research/index paths; the slice still supports a library boundary. If implementation later inlines context into agent as modules inside one crate, that is a packaging choice—**semantic ownership stays distinct**.
- Merging **contract** into **agent** would make Fabric declarations and family ABI require linking the host—Slice 4 is model-free and must stay compileable without Metal or a workspace.

### Why not more?

- **hide-fleet Fabric execution**, **research KG**, **odyssey training**, **ramanujan**, **bench oracles** do not appear as required steps in any of the four slices. They may remain as optional packages outside the twelve-crate ceiling or as lab operators. A2 does not invent a "science mega-crate" for them.
- **GGUF dense path** stays inside **runtime** as an alternate loader behind the same `Engine` trait—Slice 1's gravity path and the GGUF path share generate/serve, so splitting "gguf-runtime" would duplicate the trait boundary without a slice forcing it.

### Cross-slice shared services (not extra components)

| Shared thing | Owner | Consumers |
|---|---|---|
| `artifact_index_sha256` | C1 produces; C3 exposes on Engine; C4 health; C8 capability proof | S1, S2, S4 |
| Canonical JSON seal (`seal_sha256`) | C5 (lab) primary; C8 may re-export pure seal for agent verify | S2, S3 verify plane |
| Canonical event envelope | C8 | S3 bus, S4 fabric event kinds |
| HTTP generate contract | C4 defines; C7 consumes | S1↔S3 |

---

## 3. Genuinely shared vs merely colocated

### Genuinely shared (every / most slices couple them)

| Pair | Evidence |
|---|---|
| Artifact index hash ↔ capability admission | GravityEngine hashes index (`gravity_engine.rs:240–244`); seed-c `CapabilityProof` binds hash+rate; serve health exposes hash (`http.rs:233`). |
| Chat template bytes ↔ serve render ↔ load refuse | Load requires template for GLM; serve renders with it; missing template is fail-closed at both ends. |
| `Engine::generate` ↔ serve SSE ↔ HIDE `HttpModelProvider` | Single token stream contract; HIDE has no second inference engine. |
| Context compile recipe ↔ kernel objective ↔ `context.compiled` event | `run_turn_kernel` and `run_turn_core` share the recipe (`host_support_1.rs`). |
| Family registry fields ↔ Fabric declaration rows | `fabric_declarations_json` maps `fabric_partition_boundaries` per family (`generate.rs:1074–1085`). |
| Permission evaluate ↔ tool dispatch | `dispatch_tool` path records policy; tools cannot bypass (`host_ops_2` + `policy.rs`). |

### Merely colocated today (no slice couples them)

| Colocated pair | Why slices do not couple them |
|---|---|
| `gravity_glm_resident.rs` (13,035) and `model/qwen_dense.rs` (10,218) in `hawking-core` | S1 gravity never enters Qwen GGUF forward; GGUF serve never opens GravityShard. Same crate only for historical "one Engine crate". |
| `cost_ledger.rs` (2,666) and HIDE host | Ledger is decode instrumentation; no HIDE turn reads it. |
| `hide-backend` connector ABI and `hawking-adapters` family ABI | Both say "ABI" but different products; no slice threads a connector through model family registration. |
| `hawking-research/bridge.rs` and adapter Bridge codegen | Name collision only; research bridge never appears on Slice 4's generate path. |
| `tools/condense/glm52_state.py` (5,148) and `engine/runtime.py` | Campaign-specific controller residue beside the generic engine; Slice 2's declared authority is the engine package. |
| `host_ops_0.rs`…`host_ops_5.rs` as separate files | Mechanical split of one host impl (duplicated imports on all six); not semantic modules. Topology rule: mechanical chunking ≠ decomposition. |
| `hide-kernel` tooling_edit vs tooling_shell as separate security domains | Real difference in effects, but both are only reached through agent dispatch+permission—do not force separate crates. |
| Odyssey checkpoint store vs condense CheckpointStore | Same slice shape, zero shared callers. |
| `app/` personalize UI experiments vs classed memory forget | UI can call forget; personalize training loops in backend are not on the YOU→handoff happy path. |

### Currently separated but every relevant slice couples

| Separated today | Should share a boundary |
|---|---|
| `tools/condense/gravity_format.py` and `hawking-core/src/gravity.rs` | Dual implementations of one container; S1 and S2 both depend on identical framing—**single codec ownership in C1**, with language choice explicit (see §5). |
| Receipt seal in condense / odyssey / foundry | One pure seal function (C5 or C8). |
| Event models: `UiEvent`, `StreamEvent`, seed events, protocol items | Already partially adapted in `hawking-events`; finish the consolidation under C8 so S3 and S4 do not grow new buses. |
| Capability proof (seed-c) and artifact hash (runtime) | Same protected outcome; live in C8 policy + C1 hash production. |

---

## 4. Repetition inventory

For each: instances, rough LOC, data-vs-semantics.

### R1. Receipt seal / canonical JSON hash

| Instance | Path | LOC (measured, file) |
|---|---|---|
| 1 | `tools/condense/engine/receipts.py` | 119 |
| 2 | `tools/condense/engine/checkpoint.py` (`_canonical`, `_sha256`, seal) | 243 (shared helpers with receipt) |
| 3 | `tools/odyssey/receipts.py` | ~80 |
| 4 | `tools/odyssey/checkpoints.py` | ~230 |
| 5 | `tools/foundry/lab_harness/receipt.py` | ~60 |
| 6+ | ad-hoc in `glm52_*.py` (`_canonical_sha256`, receipt builders) | scattered; e.g. xet modules ~100+ each |

**Variation:** **mostly data** (schema field sets, schema id strings). Seal algorithm (canonical JSON + sha256) is identical.  
**Reduction:** one `lab.seal` module; schema ids become table rows. Estimated elimination **~1.5–2.5k LOC** after one implementation remains.

### R2. Family adapter modules

| Count | Path | LOC |
|---|---|---|
| 10 families | `hawking-adapters/src/families/*.rs` | **1,004 total measured** |

**Variation:** **data** — aliases, evidence paths, ABI field values, null reasons. Control flow is identical (`impl FamilyAdapter for X { descriptor() }`).  
**Reduction:** JSON/TOML/RON table + one loader, or keep const Rust tables but single macro/file. Generator already exists (`generate.rs`). Amplification likely ≥4× if specs stay compact → admissible under `generation_rules`. Estimated active LOC after: **~200–400** (loader+schema) + counted generator.

### R3. Host ops mechanical fan-out

| Files | LOC measured |
|---|---|
| `host_ops_0`…`5` + support | ~5,263 for ops alone; host+ops+support+tests dominate hide-backend's 37,742 |

**Variation:** **semantics mixed with packaging**. Ops files duplicate the same `use` prelude (all six match `use crate::memory::`). The methods are real but the file cut is not a semantic boundary.  
**Reduction:** re-module by **capability** (intent router / turn runner / tool policy / memory intents / workspace)—not by line quota. Estimated packaging-only savings small; **real** savings from deleting dual turn paths (`run_turn_kernel` vs `run_turn_core` vs single-shot) once one is authoritative—**estimated 1–3k** after behaviour matrix locks which path is product.

### R4. Checkpoint stores

| Instance | Semantics |
|---|---|
| Condense `CheckpointStore` | Hash chain + seal + split-brain checks |
| Odyssey `CheckpointStore` | Sharded payload + inject_corrupt test hooks |
| HIDE `services` CheckpointStore / rewind | Session rewind, not campaign |

**Variation:** **semantics differ** between campaign checkpoints and HIDE rewind. Campaign pair (condense vs odyssey) is **data/schema variation** on one idea.  
**Reduction:** merge campaign checkpoints only. Leave HIDE rewind in agent. **~400–800 LOC** net.

### R5. Chat template rendering

| Instance | Path |
|---|---|
| Gravity/GLM | `hawking-serve/src/glm_chat.rs` (175) |
| Arch switch | `http.rs:660+` `render_chat` / deepseek / default |
| Artifact-stored template | minijinja (or equivalent) over artifact text |

**Variation:** **data** for template text; **semantics** for tool-call scaffolding differences.  
**Reduction:** one render pipeline parameterized by template + tool schema support flag. **~200–400 LOC**.

### R6. Model-forward families (GGUF)

| Module | LOC measured |
|---|---|
| qwen_dense | 10,218 |
| deepseek_v2 | 4,337 |
| rwkv7 | 3,454 |
| llama | 648 |
| qwen_moe | 46 (thin) |

**Variation:** **semantics** (arch-specific graphs). Shared pieces (attention, MoE gate, quant GEMV) already partially factored into kernels.  
**Reduction:** not "one table"—but extracting repeated prefill/decode scaffolding and env-lever reads can still cut **estimated 15–25%** of family files without changing numerics. Treat as rewrite credit with golden decode hashes as oracle.

### R7. Laboratory campaign scripts (`tools/condense/glm52_*`, gravity labs)

| Bucket | LOC measured |
|---|---|
| `tools/condense` total | **61,457** |
| Engine core (`engine/*.py` without fixtures) | ~2.5–3k |
| Remainder | campaign-specific measure/pack/doctor |

**Variation:** mix. Operator **interfaces** are data (registry); many scripts reimplement control flow the engine already provides (**eliminable**), while scientific kernels (functional gauntlet, parity) are **semantic**.  
**Reduction plan:** move durable science into lab operators + specs; delete controllers. **Estimated 30–50k** laboratory LOC becomes specs + fewer operators, with generation only where amplification ≥4×.

### R8. Cost ledger vs ad-hoc timing

Single large module (2,666) plus scattered Instant timings in examples. **Semantic** (bucket taxonomy is load-bearing for BASE_RUNTIME_MAXIMIZED). Keep one; delete example duplicates. Low LOC win, high clarity.

---

## 5. Crate / package plan (≤12 workspace crates)

Language is **not protected**. Choice is per component.

| Workspace member | Lang | Components | Rationale |
|---|---|---|---|
| `crates/artifact` | **Rust** | C1 | Hot path reads mmap + optional GPU decode; must match serve. Python packers **call into** this via CLI/`#[pyo3]` or subprocess `hawking artifact`—not a second codec. |
| `crates/device` | **Rust + Metal** | C2 | Shaders embedded; 1 CB / ~210 dispatch discipline lives here. |
| `crates/runtime` | **Rust** | C3 | Engine + families + sample + tokenizer. Depends: artifact, device. |
| `crates/serve` | **Rust** | C4 | axum. Depends: runtime. |
| `crates/hawking-cli` | **Rust** | thin CLI | Dispatches generate/serve/bench/doctor; no business logic. Depends: serve, runtime. |
| `crates/context` | **Rust** | C6 | Model-free; SQLite/memory. |
| `crates/contract` | **Rust** | C8 | protocol + events + registry + pure capability decide + codegen binary. **No** Metal. |
| `crates/agent` | **Rust** | C7 | Merges today's hide-backend + hide-kernel + hide-core runtime pieces that are not pure contract. Depends: context, contract; HTTP client to serve. |
| `crates/hide-cli` | **Rust** | hide-serve bin | Thin axum wrapper around agent. |
| `crates/fabric` | **Rust** | optional C8 consumer | Placement simulator only if multi-node remains product; else delete from default members and keep declarations-only in contract. **Counts toward 12 if kept.** |
| *(not a crate)* `lab/` | **Python 3.12** | C5 | Experiment engine + operators. Invokes `hawking` CLI / artifact tool for pack measure. Stays out of Cargo workspace (like today's condense) so decode builds stay lean. |
| *(not a crate)* `surface/` | **TypeScript** | C9 | Vite/Tauri app; consumes generated `sdk_types.d.ts` from contract. |

**Workspace total if fabric kept:** 10 Rust crates. If fabric folded into contract as `contract::fabric_decl` only: **9**. Headroom under 12 for a `bench` crate if golden harnesses must stay Rust-linked.

**Explicit language moves:**

| Today | A2 | Why |
|---|---|---|
| Dual gravity codec Python+Rust | Rust-only C1 + Python bindings/CLI | Byte-identical container is a protected scientific/runtime seam; two codecs are a defect. |
| Family adapters as hand-written Rust modules | Data tables (RON/JSON) + small Rust loader **or** keep const data if codegen amplification fails 4× | Variation is data (R2). |
| Lab in mixed one-off scripts | Python package `lab` with one engine | Slice 2 already says this; finish the migration. |
| HIDE host split across 5 crates | One `agent` + one `contract` | Slice 3 does not cross crate-quality boundaries at hide-core vs hide-backend—it crosses intent→compile→policy→tool. |

**No permanent facades:** old crate names may exist for one migration rung as re-exports **counted as facade LOC** and must hit zero before close (`REBUILD_ACCOUNTING_RULES.facade`).

---

## 6. LOC estimate per component

Labels: **measured** = `wc` / `hawking_loc.py` on current tree; **estimated** = derived below.

### 6.1 Current absorption (measured inputs)

| Component | Primary current paths | Measured LOC (approx) |
|---|---|---|
| C1 artifact | `gravity.rs` 3,155 + gravity format/codec Python ~1.5k + tests | ~5–7k rust+py |
| C2 device | metal 5,012 + kernels 12,219 + shaders 15,766 | **33,000** |
| C3 runtime | gravity_glm* 16,619 + gravity_engine 503 + model/* GGUF ~19k + engine/sample/tokenizer/cache/… | ~45–55k |
| C4 serve | hawking-serve 5,790 + glm_chat etc. | ~6k |
| C5 lab | tools/condense 61,457 + foundry 6,863 + odyssey tools + campaign | ~70k+ |
| C6 context | hawking-context 9,804 | ~10k |
| C7 agent | hide-backend 42,111 + hide-kernel 19,995 + hide-core non-contract + hide-serve 803 + hide-fleet parts | ~70–85k |
| C8 contract | hide-protocol 5,097 + hawking-events 2,168 + hawking-adapters 4,179 + hide-core api/permission types + seed-c gravity policy | ~15–20k |
| C9 surface | app/src 15,714 | ~16k |
| Other / split | CLI 4,848, seed-c rest, research, index, orch, eval, speculate, bench, ramanujan, markdown… | remainder to 431k |

Subsystem check (measured): hawking 170,792 + hide 107,129 + laboratory 130,068 + shared 23,131 = 431,120.

### 6.2 Target estimates (estimated, with derivation)

Method: start from measured absorption; apply only (a) elimination of non-slice behaviour, (b) rewrite compression with stated factor, (c) generation where amplification ≥4× moves LOC to generator+spec.

| Component | Target LOC | Derivation |
|---|---|---|
| C1 artifact | **4,000 estimated** | Keep rust container+codec (~3k measured gravity.rs can compress ~20% by sharing PQ tables with device). Delete Python duplicate format (~1.5k eliminated). Tests rewritten as vectors. |
| C2 device | **28,000 estimated** | Shaders 15,766 mostly irreducible without changing kernels (protected perf). Host metal+kernels 17,231 → rewrite dispatch tables / merge CPU fallback paths **~15%** → ~14.6k; total ~30k then trim dead kernel variants **~2k** → 28k. **Falsifier:** any shader merge that changes golden decode or TPS. |
| C3 runtime | **32,000 estimated** | Gravity resident 13,035: extract shared encode helpers / expert table builders **20%** → ~10.4k. GGUF families 19k → shared scaffold **20%** → ~15k. Engine/sample/tokenizer/cache ~8k. Sum ~33k + cleanup. Cost ledger stays (~2k) inside runtime. |
| C4 serve | **4,500 estimated** | 5.8k − template unification − dead workload packs not on slices. |
| C5 lab | **35,000 estimated** | Engine core 3k + operators for gravity/glm science keep ~20k semantic + specs/fixtures ~8k. Eliminate duplicate controllers/receipts **~30–40k** from today's 70k lab-ish pile. Stretch: 25k if more science moves to sealed receipts only. |
| C6 context | **7,500 estimated** | 9.8k − test rewrite compression − dead personal_tools overlap with agent tools **~20%**. Forget path stays. |
| C7 agent | **45,000 estimated** | 80k → remove mechanical host_ops, dual turn paths, unused lenses/personalize if not in behaviour constitution, collapse tooling registries. **~40%** reduction is aggressive but slice-guided (S3 happy path is compile+policy+tool+memory+handoff, not every lens). Floor nearer 50k if constitution retains fleet/swarm. |
| C8 contract | **8,000 estimated** | Adapters as data + generator (R2), events single model, protocol schemas, capability decide pure. Codegen output excluded only if registry amplification ≥4× and CI-reproducible. |
| C9 surface | **12,000 estimated** | 15.7k − dead preview routes; keep three lenses. |
| CLI + glue + benches + docs-as-code | **12,000 estimated** | Thin bins, blackbox harness, minimal bench. |
| **Total** | **~188,000 estimated** primary plan | Band: **180k–230k** depending on agent/lab constitution scope. |

**250k reachability:** **Yes, believed reachable** under the accounting rules, with the primary plan landing near **200–220k** before optional fabric/research extras. Stretch 200k needs lab ≤30k and agent ≤40k.

**Single biggest obstacle:** **C3+C2 performance and numerical coupling** inside the Gravity GLM resident forward—13k LOC of encode/dispatch that must preserve 1 CB/~210 dispatches and bit-exact sampling while being decomposed. Mis-drawing that boundary costs the protected TPS/dispatch numbers. Second obstacle: **laboratory elimination discipline** (easy to "archive" scripts—that is relocation, credit zero).

---

## 7. Three riskiest boundaries (adversarial)

### Risk A — `device` ↔ `runtime` (dispatch ownership)

**Claim:** runtime owns family graph; device owns CB encode; crossing is typed kernel APIs without per-token heap or extra CB.

**Failure mode:** "clean" abstraction inserts an extra command buffer, copies logits host↔device, or loses ICB encoding—**>2% TPS regression** or dispatch count inflation.

**Falsifying evidence:**

- Golden decode hash change on fixed seed/prompt/artifact.
- Cost ledger report: `command_buffers_submitted` ≠ 1 per token on the base path, or `dispatches_encoded` moves outside measured band without a trade receipt.
- `hawking bench` / gravity TPS examples regress >2% vs sealed baseline on the same machine class.

### Risk B — `lab` Python packer ↔ `artifact` Rust codec

**Claim:** one codec ownership in Rust; Python operators invoke it.

**Failure mode:** pack path drifts (Python still writes headers Rust cannot read) or subprocess tax makes transformation throughput miss protected targets.

**Falsifying evidence:**

- Round-trip: Python operator pack → Rust `GravityShard::open` + tensor hash mismatch.
- `tools/loc` generation audit fails reproducibility.
- Transformation throughput (pack tensors/s) drops >2% vs current Python-inline path on the same corpus.

### Risk C — `agent` ↔ `serve` HTTP process boundary

**Claim:** keep HTTP (or equivalent IPC) so agent does not link Metal; Slice 1 last mile stays process-isolated.

**Failure mode:** latency/complexity pushes a future rung to "just link runtime into agent", re-creating a monolith and coupling HIDE releases to shader changes; or SSE semantic drift breaks tool loops.

**Falsifying evidence:**

- Turn TTFT or tokens/s through HIDE regresses >2% vs direct serve client on identical prompts.
- Blackbox matrix: SubmitTurn runnable → unrunnable (anti-gaming rule).
- Capability surface: agent process gains authority to open raw artifact weights (security outcome change).

---

## 8. What this architecture refuses

- Permanent old+new double engines, double receipt authorities, double event buses.
- Compatibility facades left after a rung (facade LOC must be 0 to close).
- Counting relocated laboratory scripts as eliminated.
- Splitting files solely to game `files_over_1500` without semantic criteria.
- Aiming at six components because another document did.

---

## 9. What I did not read

Plain list of deliberate and practical omissions:

**Deliberately unread (campaign ban):**

- `HAWKING_VNEXT_ARCHITECTURE_A1.md`
- `control/RECOMP-SCOUT-COMPARISON.json`

**Not read in full (scout breadth only; entry points + follow-the-slice):**

- Bulk of `gravity_glm_resident.rs` (13k)—read structure, encode symbols, cost-ledger hooks; not every expert path.
- Bulk of `model/qwen_dense.rs`, `deepseek_v2.rs`, `rwkv7.rs`—sizes and dispatch role only.
- Most `tools/condense/glm52_*.py` bodies—inventory + receipt/checkpoint touchpoints, not every measurement.
- `tools/bench/**` oracles, `ramanujan/**`, `odyssey/**` training (except checkpoint/receipt modules).
- Full `docs/hide-bible/**` and most `docs/gravity/**` (opened names + container docs index only).
- `app/src` component tree beyond LOC and intent client role.
- `hide-fleet` fabric pipeline/failure/agent binaries beyond placement declaration consumer role.
- `hawking-orch`, `hawking-index`, `hawking-eval`, `hawking-speculate`, most of `hawking-seed-c` beyond capability proof.
- Shader bodies (`.metal`)—LOC and embed model only; not kernel algebra review.
- `HAWKING_CAPABILITY_INVENTORY.json` beyond counts/entrypoint lists.
- Prior rung verdicts under `control/*-verdict.json` (except binding `REBUILD_ACCOUNTING_RULES.json`).
- Vendor and archived packs.

---

## 10. Implementation order (for a reader who will build it)

1. **Freeze blackbox matrix** behaviours for the four slices (not done in this design pass; prerequisite).
2. **C1 artifact** single codec + hash API; make Python pack call it; golden container vectors.
3. **C2/C3 split** inside today's hawking-core without behaviour change; prove TPS + dispatch + decode hash.
4. **C5 lab engine** sole campaign controller; delete duplicate receipt/checkpoint stacks (eliminate, don't archive).
5. **C8 contract** registry-as-data + fabric declarations generator; kill hand-written family modules if amplification holds.
6. **C7 agent** single turn path; permission+tool+memory+handoff; collapse host_ops packaging.
7. **C9 surface** against generated SDK only.
8. Topology pass: directories ≤60, files ≤450, symbols/functions gates.

---

## 11. Verdict

| Question | Answer |
|---|---|
| Is 250,000 combined active LOC reachable? | **Yes**, with the nine-component plan and aggressive but slice-justified lab+agent elimination. Preferred 225k is plausible; stretch 200k requires lab/agent floors above. |
| Biggest obstacle? | **Preserving the Metal token path (1 CB, ~210 dispatches, bit-exact sample) while decomposing `gravity_glm_resident` / device from runtime**—not directory topology. |
| Is this A1? | **Unknown and irrelevant.** Derived only from four slice traces and measured tree facts. |

---

*End of A2.*
