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
Host-wide process observation and startup orphan reaping are now native in
`hide-backend::process_inspector`; Python exposes only a compatibility adapter
for the retained `/processes` and registry callers.

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
| after Rust/headless compaction | 1,399,893 | 3,025 | 712,010 | 490,150 | 1,095 | 40.599% |
| current clean-tree census | 1,394,764 | 3,013 | 707,288 | 489,679 | 1,095 | 40.736% |

The future farm deletion removed 278 uncalled Python files (192,748 physical
Python lines), leaving 60 tracked future-farm files: 57 retained Python
modules, the existing ledger and memory-traffic probe, and the retained test
helper shell script. The later Rust/headless compaction removed 171 uncalled
Rust example/shader files (168,974 physical lines) and 74 caller-free
headless Python files (39,566 physical lines). The current LOC measurement
includes this scorecard, which is now part of the tracked branch.

The Rust workspace census is now 18 packages, 12 binaries, and 73 examples.
The two retired resident-server entry points were `hide-headless` and the old
dirty-tier `research_server`; HCLI is the remaining HIDE backend binary.
The former one-consumer `hawking-index-query` package is now a binary and
Python-facts module owned by `hawking-index`; its command name and
`hawking.index.python_facts.v1` wire schema are unchanged.
The former library-only `hawking-bench` package is now the private
`hawking/src/bench` module owned by the `hawking` CLI; benchmark subcommands
and report schemas remain unchanged.

The branch must report the post-pruning measurement after the deletion commit;
historical receipts and generated graphs are not counted as source reduction.

The current row is the clean-tree measurement after the small ABI, contract,
receipt-path, closure-report, stream-render, runtime-identity, Rust-REPL,
duplicate-GoalIR, test-only-checkpoint, benchmark-package, and native-process
authority consolidations made after the historical compaction checkpoint. The
code-only
Rust share is computed as
`Rust / (Rust + Python + TypeScript + shell)`; it is not inflated by excluding
the deleted Rust examples or by treating Markdown as executable source.

Against the Phase III base, the current tree is down 432,576 active physical
LOC and 644 active files. Python is down 238,484 LOC; Rust is down 174,969 LOC
because the deletion wave removed uncalled historical examples rather than
restoring them to improve a ratio. The code-only Rust share therefore moved
from the historical 40.736% checkpoint to 40.736%; the native process
authority recovers some Rust share, but this remains an open migration target
rather than a claimed success.

The tracked tree is 10,334 files / 461,352,875 bytes versus the Phase III base
of 11,058 files / 479,552,373 bytes: down 724 files and 18,199,498 bytes.

## Closure status

This scorecard is an active checkpoint, not a declaration that every Phase III
gate is closed.

| gate | current evidence | status |
|---|---|---|
| at least 10,000 active source LOC removed | base 1,827,340 -> current 1,394,764 | met |
| Rust active share materially increased | historical 40.736% -> current 40.736% after native process migration | open |
| Rust workspace check | `cargo check --workspace` | pass |
| relevant Rust tests | `cargo test --workspace --lib --bins` | pass |
| full Python HCLI suite | 1,522 passed, 49 protected/evidence failures, 7 skipped (1,578 collected) | open; no failures hidden; seven moved process-policy assertions now run in Rust |
| no new unexplained failures | focused regressions closed; missing live sovereign receipts remain | open |
| examples and launch surface reduced | 235 -> 73 examples; 14 -> 12 binaries | met |
| remaining Rust packages have an ownership boundary | 18 packages after folding two one-consumer facades | met |
| clean working tree | `git status --short` empty | met |

The 49 HCLI failures are not converted into green receipts: 48 require ten
missing live sovereign receipts (G002-G008, G012-G013, G015), and one existing
G014 negative-science receipt still reports a non-zero dead-family duration.
Those are evidence prerequisites for a later closure run, not source-code
failures that this branch may fabricate away.

## Python-to-Rust migration register

The register is deliberately narrow. A Python module is not a migration target
just because Rust can express it; it must have a durable/runtime/control-plane
role, a measured caller, and a clear Rust owner.

| concern | current authority | Rust owner | disposition |
|---|---|---|---|
| backend command ingress and JSONL control | `crates/hide-backend/src/bin/hcli.rs` + `hcli_bridge` | `hide-backend` / `hide-protocol` | consolidated; visual clients deferred behind VMCP |
| durable HIDE sessions and receipts | `hide-backend::services`, `replay`, `rewind` | `hide-backend` | Rust authority; no Python duplicate for this surface |
| local model/runtime serving | `hawking-serve` and `hide-backend::supervisor` | Hawking runtime crates + `hide-backend` | Rust authority; Python may supervise its resident body |
| host process observation and startup reaping | former `hcli.processes` implementation | `hide-backend::process_inspector` via `hcli processes` | consolidated; Python is a thin compatibility adapter and model tools remain read-only |
| AgentOS resident loop and comparative provider integrations | `hcli.agentos`, `hcli.runtime`, provider modules | none yet | retained in Python until a parity-tested Rust owner exists |
| ordinary crash-safe Python writes | `hcli.persist` | none yet | retained; narrow, tested Python advantage, not a second HIDE store |
| VMCP perception | external `visionmcp` plus `tools/vmcp` adapters | external VMCP boundary | retained as a boundary; visual rebuild waits for hardening |
| one-shot research and retired experiment runners | `tools/future` | none | delete uncalled producers; retain receipts and live call-path modules |

## Bridge register

| bridge | owner | reason it exists now | removal condition |
|---|---|---|---|
| Python AgentOS -> resident/native provider | Python `hcli.agentos` | long-lived resident and hardware-specific provider work is not yet parity-ported | a Rust resident loop passes restart, receipt, and negative-control parity |
| Python process skin -> native HCLI | `hcli.processes` -> `hcli processes --json` | preserves legacy REPL/tool result shape while Rust owns host inspection and signals | Python process callers migrate directly once the native binary is the installed entry point |
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
