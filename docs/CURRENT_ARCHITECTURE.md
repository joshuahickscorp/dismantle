<!-- DOC_STATUS: CURRENT -->
# Current architecture

Updated 2026-09-04 on the Event Horizon refactor branch. This document is the
human-readable architecture source of truth. Census files, capability graphs,
and receipts are point-in-time evidence; they may describe an older commit and
must not be treated as a second architecture map.

## Operating shape

```text
model/specimen metadata ──┐
                          v
                      HCLI / AgentOS ── typed tools, work units, gates,
                          │              missions, receipts, resident control
                          v
                hawking-core / hawking / hawking-serve
                          │
                 CPU reference + Apple Metal runtime
```

HCLI is the Python control plane. Rust owns model execution and serving. HCLI
may plan, invoke, and verify runtime work, but a receipt or a plan never raises
the capability ceiling of an unmeasured artifact.

## Ownership map

| Concern | Canonical home | Boundary |
|---|---|---|
| Product CLI, UI, command ingress | `hcli.cli`, `hcli.app`, `hcli.commands`, `hcli.controller`, `hcli.tui` | User-facing control surface |
| AgentOS work and lifecycle | `hcli.agentos` plus canonical `hcli.goal`, `hcli.workunit`, `hcli.scheduler`, `hcli.mission`, `hcli.verifier_pipeline` | Scheduling/proposal is not verification |
| Runtime and provider execution | `hcli.runtime`, `hcli.engine`, `hcli.backends`, `hcli.session`, `hcli.context`, `hcli.models` | Provider output is evidence only after the verifier accepts it |
| Crash-safe persistence | `hcli.persist` | The shared text/bytes/JSON atomic writers; specialized compare-and-swap remains in its owner |
| Doctor diagnosis | `tools.doctor.engine` and `tools.doctor.*` | Metadata/receipt diagnosis; no weight loading or hardware claim |
| Gravity representation search | `hcli.gravity`, `hcli.agentos.flash_representation_experiment`, `tools/gravity_*.py`, Rust runtime crates | Search/compile is distinct from physical qualification |
| Status verification | `tools.verify.status_causality` | A status may assert only what its actual probe establishes |
| Roadmap and reachability | `tools.roadmap`, especially `tools.roadmap.capability_reachability` | Definitions/imports are not calls; receipts are citations, not callers |
| Odyssey and ModelLake | `tools.odyssey`, including `tools.odyssey.modellake_promote` | Specimen lifecycle and promotion stay separate from Doctor/Gravity |
| Sensory evidence | `hcli.vmcp` and the `visionmcp/` package | External VMCP surface; no duplicate sensory authority |
| Machine/runtime identity | `hcli.machine`, `hcli.genomes`, `hcli.runtime_iface` | Identity and learned metadata do not authorize physical results |
| HIDE IDE | `hide-*` crates | HIDE may depend on Hawking; Hawking-side HIDE edges are a known inversion debt |

`hcli.agentos` is a public namespace and ownership surface; it is not a second
copy of the core lifecycle classes. Goal compilation, WorkUnit identity,
Mission state, and verifier orchestration each retain their established
canonical modules.

`tools/future/` and `tools/hcli/bootstrap/` are historical research and fossil
surfaces, not live product authorities. Their fixtures and receipts remain
available for audit and reproducibility. In particular, the scientific
metabolism and resident-supervisor records use explicit sidecar names so they
cannot be mistaken for `hcli.workunit.WorkUnit` or the live resident control
loop.

## Rust workspace boundary

The default build surface is the Hawking inference/serving workspace plus its
context, index, orchestration, research, event, adapter, and bake tools. HIDE
crates are workspace members but are outside `default-members`. `hawking-serve`
serves the runtime; `hide-backend` consumes it over HTTP.

Cargo metadata currently shows six Hawking-to-HIDE edges through shared errors,
IDs, blobs, and UI event types (`hawking-context`, `hawking-index`,
`hawking-orch`, `hawking-research`, and `hawking-events`). They are documented
debt, not silently reclassified as clean layering. Removing those edges needs a
separate neutral-primitive/UI-vocabulary change and is outside the no-physical-
optimization refactor lane.

The undeclared `crates/hide-backend/src/hcli/` tree and its `hcli-backend`
wrapper are a retained, non-building historical scaffold. The live Rust HCLI
surface is the declared `hcli_bridge`, profile, research, source, and swarm
modules; the old scaffold is not a second runtime authority.

## Authority rules

1. `hcli.workunit.WorkUnit` is the live WorkUnit identity. Scientific sidecar
   records must use a distinct name when their fields are not the HCLI shape.
2. `hcli.persist.atomic_write_text`, `atomic_write_bytes`, and
   `atomic_write_json` own ordinary crash-safe writes. Compare-and-swap lineage
   writes remain specialized and are not collapsed into last-writer-wins I/O.
3. `tools.verify.status_causality` owns probe-to-status entailment. Historical
   receipt schema strings remain readable; new executable callers use this path.
4. `tools.roadmap.capability_reachability` owns static reachability analysis.
   Its Rust parity binary is an accelerator for the same facts, not a second
   semantic verdict.
5. `tools.doctor.engine` owns Doctor diagnosis. `doctor.query` dispatches to it;
   `gravity.experiment` dispatches to the bounded Flash representation runner
   only when explicitly requested.
6. `tools.odyssey.modellake_promote` owns ModelLake promotion policy. A sealed
   specimen, registry entry, or benchmark receipt alone is not promotion.
7. Physical performance claims require a protected window, live samples,
   independent verification, negative controls, and reproducible provenance.
   This refactor changes structure and callers only; it does not optimize a
   kernel, acquire weights, or reinterpret a static result as measured speed.

## State and evidence placement

- `civilization/` holds roadmap and launch-goal state.
- `receipts/` holds acceptance, provenance, and preserved historical evidence.
  Sealed bytes are not rewritten merely to make names prettier.
- `workspace/`, `.hcli/`, Cargo targets, Python caches, and campaign runtime
  outputs are generated/local state and belong outside the source authority.
- Model weights and external ModelLake payloads are local inputs, never source
  files in this repository.

## Entry points and verification

```bash
python3 -m hcli --help
python3 -m hcli.agentos.resident --help
python3 -m tools.doctor --selftest
python3 -m tools.roadmap --help
cargo check --workspace
python3 -m pytest
```

The default pytest target is the complete live `hcli/` package. Historical
`tools/future/` harnesses are run explicitly because they write evidence
fixtures and own a separate session setup.

For isolated work, set `CARGO_TARGET_DIR`, `PYTHONPYCACHEPREFIX`, and the
pytest cache directory to a temporary refactor-specific location. Compare
before/after census counts, test results, Rust checks, CLI smoke, HCLI
restart/resume, protected verifiers, capability-graph/roadmap checks, VMCP,
ModelLake identity, NR/NX, and negative controls. A failure that predates the
refactor remains classified as pre-existing until its cause changes.
