# Event Horizon support mass map

Measured on `refactor/event-horizon` after the Qwen30 future-oracle, N051
one-shot analysis, CUDA hypothesis, WorkGraph, frontier-runtime, and legacy
Ascension-supervisor retirement waves. This is the support
baseline for the family-reduction campaign; product LOC is frozen unless a
support consolidation requires an ownership correction.

## Accounting

The LOC authority counts tracked source/document lines with known code or
documentation extensions, excluding generated files, vendored files, and
documentation archives. Product is the conservative HCLI plus Rust/shader
implementation boundary used by `tools/loc/hawking_loc.py`. The remainder is
support.

| measure | LOC | files |
|---|---:|---:|
| product | 498,012 | 635 |
| support | 525,592 | 1,488 |
| active total | 1,023,604 | 2,123 |

The authoritative headline is the value reported by `tools/loc/hawking_loc.py`.

## Top 30 support subtrees

Subtrees are non-overlapping two-component path groups, sorted by counted LOC.
This table accounts for the full support total; the tail outside the top 30 is
also included in the total and must not become an UNKNOWN bucket.

| rank | subtree | LOC | primary disposition |
|---:|---|---:|---|
| 1 | `research/lab` | 93,616 | ACTIVE_RESEARCH / MIGRATE_INTO_OWNER |
| 2 | `crates/hawking-core` | 52,454 | CURRENT_VERIFICATION |
| 3 | `tools/future` | 54,797 | ACTIVE_RESEARCH / SUPERSEDED audit required |
| 4 | `tools/accelerator` | 39,624 | CURRENT_VERIFICATION / ACTIVE_RESEARCH |
| 5 | `research/hawking-experiments` | 4,695 | ACTIVE_RESEARCH / IMPORTANT_ARCHIVE |
| 6 | `tools/headless` | 28,801 | CURRENT_VERIFICATION / MIGRATE_INTO_OWNER |
| 7 | `tools/condense` | 25,925 | CURRENT_CORE_SUPPORT / MIGRATE_INTO_OWNER |
| 8 | `tools/odyssey` | 18,569 | CURRENT_CORE_SUPPORT |
| 9 | `hcli/tests` | 19,832 | CURRENT_VERIFICATION |
| 10 | `workspace/campaign` | 4,464 | ACTIVE_RESEARCH / IMPORTANT_ARCHIVE |
| 11 | `tools/verify` | 10,467 | CURRENT_VERIFICATION |
| 12 | `tools/roadmap` | 10,262 | CURRENT_CORE_SUPPORT |
| 13 | `tools/graph` | 9,826 | CURRENT_VERIFICATION / ACTIVE_RESEARCH |
| 14 | `docs/roadmap-lineage` | 273 | IMPORTANT_ARCHIVE |
| 15 | `workspace/docs` | 2,587 | IMPORTANT_ARCHIVE / CURRENT_META |
| 16 | `tools/acceptance` | 9,017 | CURRENT_VERIFICATION |
| 17 | `tools/odyssey_ctl.py` | 8,132 | CURRENT_CORE_SUPPORT |
| 18 | `crates/hide-backend` | 7,571 | CURRENT_CORE_SUPPORT |
| 19 | `workspace/ops` | 386 | CURRENT_CORE_SUPPORT / IMPORTANT_ARCHIVE |
| 20 | `tools/ascent` | 4,972 | CURRENT_VERIFICATION / ACTIVE_RESEARCH |
| 21 | `tools/odyssey_patient_runner.py` | 4,604 | CURRENT_CORE_SUPPORT |
| 22 | `tools/audit` | 4,466 | CURRENT_VERIFICATION |
| 23 | `receipts/audit` | 53 | IMPORTANT_ARCHIVE |
| 24 | `tools/theia` | 3,984 | ACTIVE_RESEARCH |
| 25 | `tools/agentos` | 3,981 | CURRENT_CORE_SUPPORT |
| 26 | `docs/ultragoals` | 3,783 | IMPORTANT_ARCHIVE / CURRENT_META |
| 27 | `civilization/build_state.py` | 2,685 | CURRENT_CORE_SUPPORT |
| 28 | `tools/sovereign` | 2,602 | CURRENT_VERIFICATION |
| 29 | `tools/doctor` | 2,361 | CURRENT_VERIFICATION |

## Disposition rules for the next waves

- `research/lab`: registry-backed operators and their dependency clusters
  survive only when their current scientific owner or verifier is identified;
  unowned campaign chains are candidates for extraction to result/receipt/
  reproducer and deletion. The unrun Qwen30 streamed-source/raw-logit
  preparation chain was compressed to `QWEN30_STREAMED_ORACLE_ARCHIVE.md`; the
  frozen product-side Rust examples were retained.
- `tools/headless`: the concluded N051 sensitivity allocation was already
  fully represented by its sealed receipt and bounded interpretation; its
  one-shot generator was retired.
- `tools/future`: the static CUDA literature adapter was retired after its
  sealed receipt became the canonical record; `science_corpus` retains the
  schema vocabulary needed to read those hypotheses.
  The parallel future WorkGraph runtime was then sublated into a bounded
  arrival-payload builder; HCLI `scheduler`/`dag_store` now own scheduling.
  The parallel 22-frontier runtime was subsequently retired; its sealed state
  remains archival while `hcli/frontier_scheduler.py` owns active selection.
- `research/hawking-experiments`: retain decisive findings and minimal
  reproducers; delete superseded executable campaign machinery after checking
  provenance and negative controls. The G1 evidence runners are now retired;
  their result payloads and selected experiment notes remain the archival signal;
  eight unreferenced superseded reports and 22 unreferenced G1 reports were
  compressed into `G1_ARCHIVE_LEDGER.md`; the closed Qwen30 admission-probe
  family was reduced to `QWEN30_CLOSED_ADMISSION_ARCHIVE.md`; eight unreferenced
  Ascension contract/workflow modules were reduced to
  `ASCENSION_ORPHANED_CONTRACTS_ARCHIVE.md`; 41 concluded G1 reports were
  reduced to the complete `G1_ARCHIVE_LEDGER.md` decision record.
  The three former live G1 architecture/Tabula reports were then folded into
  that same ledger and removed as parallel machine-readable authorities; the
  ledger now carries the exact doctrine and closure markers consumed by the
  current readers.
  The duplicate Frankenstein `condense/` wrapper and test tree was then
  retired; live tooling remains in `tools/condense/`, operators remain under
  `frankenstein/operators/`, and all Frankenstein data/evidence is preserved.
  The uncalled Frankenstein prototype operator generation was then retired as
  a 23,915-line cluster; current pipeline, trace/gate, and evidence
  authorities remain,
  and the disposition is recorded in `ASCENSION_PLAN_ARCHIVE.md`.
  The unreferenced numbered audit reports and Odyssey Sub-1 narrative packet
  were then compressed into `receipts/audit/EVENT_HORIZON_ARCHIVE_LEDGER.md`;
  durable findings, evidence classes, negative-science boundaries, and
  retention rules remain while 28 historical narratives leave active HEAD.
- `research/lab/operators`: the blocked Qwen scientific optimizer watcher had
  no live caller or material physical state; its implementation, launcher, and
  dedicated fixture were deleted while its blocked-runtime claim boundary was
  retained in `QWEN_SCIENTIFIC_OPTIMIZER_ARCHIVE.md`.
  The dependent Ascension V3 lifecycle/campaign, physical gatekeeper,
  tournament, notifier, source-admission, sandbox, launch-gate, launchd, and
  test chain was then retired as one broken legacy family: 14,997 active LOC
  disappeared while current manager-protocol surfaces, Qwen30 physical
  research, and receipts remained.
  The uncalled GLM52 Gravity selection/benchmark/fixture and repack-score
  cluster was then removed as a research-only generation: 3,393 operator LOC
  plus its dedicated test harnesses disappeared, while the current
  `gravity_metal` decoder and archival evidence boundaries remained.
  The uncalled Qwen state-KV and Q80 mixed-representation experiments were
  subsequently removed as a second research-only cluster; active Qwen30
  activation-weighted and Q80 capture-index owners remain.
  The uncalled DeepSeek schedule/oracle, residual-teacher admission, and Qwen
  metadata-preflight cluster was then compressed to
  `STALE_RESEARCH_CLUSTER_ARCHIVE.md`; no live DeepSeek stream or Qwen capture
  owner was changed.
  The duplicate, uncalled GLM52 activation-pack v2 generation was retired;
  v1 remains the active pack owner and the campaign registry was corrected.
  The superseded H-ROADMAP copy was then removed from active HEAD; its digest,
  provenance, and recoverability through Git history remain in PRESERVATION.md,
  while the operator-owned current roadmap remains the only resolver authority.
  The unreferenced per-lane ascent runbooks were then collapsed to the compact
  planning archive; shared density/common references and sealed findings remain.
- `tools/headless`: three uncalled Noetic/bandwidth producer runtimes were
  retired; their sealed findings remain under `receipts/headless/`, and the
  remaining adversary evidence is receipt-backed after its broken
  Noetic/VisionMCP runners were retired.
- `research/lab/tests`: tests for already-deleted Ascension supervisors and
  absent JSON contract fixtures were removed; current manager-protocol and
  Qwen30 physical tests remain. The uncalled manager-tournament readiness /
  protocol pair and generic parity-ladder scaffold were then retired as one
  family; current Rust paired-cognition and Doctor authorities remain.
- `crates/hawking-core/examples`: twelve uncalled Ascension packing, capture,
  drift, and contract programs were retired as historical executable
  scaffolding; current Flash acceptance examples and runtime modules remain.
  The standalone Qwen80 multi-layer completion assessor and its adversarial
  harness were also removed because no current acceptance or registry path used
  them.
- `crates/hawking-core/tests`: the versioned v030–v2s / Phase-2 / draft /
  prototype parity generation was retired as superseded standalone scaffolding;
  current named verification tests remain. The unreferenced low-level
  predecode/Q4K/RoPE/fusion micro-test generation was then retired; the two
  MHA tests named by the active prefill-KV census remain.
- `crates/hawking-core`, `tools/verify`, `tools/acceptance`, `tools/audit`, and
  `hcli/tests`: current verification is a valid survival reason; consolidate
  duplicated plumbing only when proof independence remains intact.
- `tools/future`, `tools/headless`, `tools/condense`, and `tools/odyssey`: audit
  complete dependency clusters against the current HCLI/Rust owner. A caller
  alone is not sufficient evidence of survival.
- `workspace/*`, `docs/*`, and `receipts/*`: classify information separately
  from executable machinery. Preserve current metadata, decisive evidence,
  Laws/Scars, and necessary provenance; compress or retire stale execution
  scaffolding where a smaller canonical representation carries the signal. The
  superseded TG `matched/` receipt root is now represented by canonical
  `matched-v2/`; its v1 timing caveat remains in the v2 evidence root.
  Generated governance rung snapshots are counted outside the active LOC
  authority; three intermediate before/after pairs were compressed to the
  canonical comparison pair, final verification snapshot, and
  `GOVERNANCE_RUNG_ARCHIVE.md`.

The next checkpoint is support <=500K, then <=250K, with the current product
boundary held at approximately 498K.
