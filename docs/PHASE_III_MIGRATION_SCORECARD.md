# Phase III migration scorecard

This branch is the structural refactor authority for `refactor/event-horizon`.
The live ascension worktree remains foreign and is not part of this scorecard.

## Product boundary

HCLI is the current headless product surface. The Rust `hcli` binary in
`crates/hide-backend` owns the consolidated HIDE backend: durable sessions,
backend protocol, tools, receipts, local-model composition, and runtime-facing
control. Python `hcli` remains the orchestration and resident skin where its
comparative advantage is the long-lived AgentOS loop and experimental provider
integration. Those are different ownership layers, not a second visual app.

The desktop frontend, localhost `hide-serve` transport, and ACP server are
intentionally absent from this phase. They are recoverable from Git history and
may be rebuilt only after the external VMCP boundary is hardened. No current
visual authority is implied by this document.

## Measured checkpoint

All LOC values come from `tools/loc/hawking_loc.py`; they are physical lines of
tracked executable source, including comments and blanks. The Phase III base is
`eec54c2ae`.

| checkpoint | active LOC | active files | Python | Rust | TypeScript | Rust code-only share |
|---|---:|---:|---:|---:|---:|---:|
| Phase III base `eec54c2ae` | 1,827,340 | 3,657 | 945,772 | 664,648 | 17,005 | 40.736% |
| after HCLI boundary deletion | 1,803,951 | 3,550 | 945,772 | 657,320 | 1,095 | 40.872% |
| after future-farm pruning | 1,609,847 | 3,273 | 751,583 | 657,320 | 1,095 | 46.485% |
| after Rust/headless compaction | 1,399,885 | 3,025 | 712,010 | 490,150 | 1,095 | 40.599% |

The future farm deletion removed 278 uncalled Python files (192,748 physical
Python lines), leaving 61 tracked future-farm files: 57 retained Python
modules, the existing ledger and memory-traffic probe, and the retained test
helper shell script. The later Rust/headless compaction removed 171 uncalled
Rust example/shader files (168,974 physical lines) and 74 caller-free
headless Python files (39,566 physical lines). The current LOC measurement
includes this scorecard, which is now part of the tracked branch.

The Rust workspace census is now 20 packages, 12 binaries, and 73 examples.
The two retired resident-server entry points were `hide-headless` and the old
dirty-tier `research_server`; HCLI is the remaining HIDE backend binary.

The branch must report the post-pruning measurement after the deletion commit;
historical receipts and generated graphs are not counted as source reduction.

## Python-to-Rust migration register

The register is deliberately narrow. A Python module is not a migration target
just because Rust can express it; it must have a durable/runtime/control-plane
role, a measured caller, and a clear Rust owner.

| concern | current authority | Rust owner | disposition |
|---|---|---|---|
| backend command ingress and JSONL control | `crates/hide-backend/src/bin/hcli.rs` + `hcli_bridge` | `hide-backend` / `hide-protocol` | consolidated; visual clients deferred behind VMCP |
| durable HIDE sessions and receipts | `hide-backend::services`, `replay`, `rewind` | `hide-backend` | Rust authority; no Python duplicate for this surface |
| local model/runtime serving | `hawking-serve` and `hide-backend::supervisor` | Hawking runtime crates + `hide-backend` | Rust authority; Python may supervise its resident body |
| AgentOS resident loop and comparative provider integrations | `hcli.agentos`, `hcli.runtime`, provider modules | none yet | retained in Python until a parity-tested Rust owner exists |
| ordinary crash-safe Python writes | `hcli.persist` | none yet | retained; narrow, tested Python advantage, not a second HIDE store |
| VMCP perception | external `visionmcp` plus `tools/vmcp` adapters | external VMCP boundary | retained as a boundary; visual rebuild waits for hardening |
| one-shot research and retired experiment runners | `tools/future` | none | delete uncalled producers; retain receipts and live call-path modules |

## Bridge register

| bridge | owner | reason it exists now | removal condition |
|---|---|---|---|
| Python AgentOS -> resident/native provider | Python `hcli.agentos` | long-lived resident and hardware-specific provider work is not yet parity-ported | a Rust resident loop passes restart, receipt, and negative-control parity |
| HCLI backend -> `hawking serve` | Rust `hide-backend::supervisor` over HTTP | keeps the engine crate boundary narrow while the runtime remains independently served | a stable hardened VMCP/runtime transport replaces the compatibility launcher |
| VMCP adapter -> external `visionmcp` | `hcli.vmcp_adapter` / `tools/vmcp` | VMCP is an external package and must remain an independent sensory authority | hardened VMCP contract covers visual/IDE clients and evidence semantics |

No bridge is allowed to grow a second scheduler, durable state store, or
semantic API. A future Rust port must delete or repatriate its Python owner in
the same change that moves the callers.

## Future-farm pruning rule

The retained `tools/future` set is the transitive closure of current production
imports plus the explicit lifecycle and FPGA acceptance call paths. The removed
files had no production import/call path outside the future farm. They are not
silently reclassified as documentation: their executable source is removed and
their historical receipts remain immutable. Future test commands must target a
retained owner or a current acceptance test; there is no separate uncalled
future test farm in the product build.
