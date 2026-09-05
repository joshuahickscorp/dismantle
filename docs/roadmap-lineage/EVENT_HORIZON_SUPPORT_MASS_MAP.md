# Event Horizon support mass map

Measured on `refactor/event-horizon` at the post-G1-execution wave. This is the support
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
| support | 730,824 | 1,958 |
| active total | 1,228,836 | 2,593 |

The five-line difference from the headline LOC tool is newline accounting in
this independent cross-check; the authoritative headline remains the tool's
reported value.

## Top 30 support subtrees

Subtrees are non-overlapping two-component path groups, sorted by counted LOC.
This table accounts for the full support total; the tail outside the top 30 is
also included in the total and must not become an UNKNOWN bucket.

| rank | subtree | LOC | primary disposition |
|---:|---|---:|---|
| 1 | `research/lab` | 139,287 | ACTIVE_RESEARCH / MIGRATE_INTO_OWNER |
| 2 | `crates/hawking-core` | 87,451 | CURRENT_VERIFICATION |
| 3 | `research/hawking-experiments` | 68,776 | ACTIVE_RESEARCH / IMPORTANT_ARCHIVE |
| 4 | `tools/future` | 60,654 | ACTIVE_RESEARCH / SUPERSEDED audit required |
| 5 | `tools/accelerator` | 39,624 | CURRENT_VERIFICATION / ACTIVE_RESEARCH |
| 6 | `tools/headless` | 34,903 | CURRENT_VERIFICATION / MIGRATE_INTO_OWNER |
| 7 | `tools/condense` | 26,556 | CURRENT_CORE_SUPPORT / MIGRATE_INTO_OWNER |
| 8 | `tools/odyssey` | 24,488 | CURRENT_CORE_SUPPORT |
| 9 | `hcli/tests` | 19,836 | CURRENT_VERIFICATION |
| 10 | `workspace/campaign` | 15,668 | ACTIVE_RESEARCH / IMPORTANT_ARCHIVE |
| 11 | `workspace/docs` | 11,725 | IMPORTANT_ARCHIVE / SUPERSEDED audit required |
| 12 | `tools/verify` | 10,467 | CURRENT_VERIFICATION |
| 13 | `tools/roadmap` | 10,262 | CURRENT_CORE_SUPPORT |
| 14 | `tools/graph` | 9,826 | CURRENT_VERIFICATION / ACTIVE_RESEARCH |
| 15 | `docs/roadmap-lineage` | 9,812 | IMPORTANT_ARCHIVE |
| 16 | `tools/acceptance` | 9,017 | CURRENT_VERIFICATION |
| 17 | `tools/odyssey_ctl.py` | 8,132 | CURRENT_CORE_SUPPORT |
| 18 | `crates/hide-backend` | 7,571 | CURRENT_CORE_SUPPORT |
| 19 | `workspace/ops` | 5,392 | CURRENT_CORE_SUPPORT / IMPORTANT_ARCHIVE |
| 20 | `tools/ascent` | 4,972 | CURRENT_VERIFICATION / ACTIVE_RESEARCH |
| 21 | `tools/odyssey_patient_runner.py` | 4,604 | CURRENT_CORE_SUPPORT |
| 22 | `tools/audit` | 4,466 | CURRENT_VERIFICATION |
| 23 | `receipts/audit` | 4,368 | IMPORTANT_ARCHIVE |
| 24 | `tools/theia` | 3,984 | ACTIVE_RESEARCH |
| 25 | `tools/agentos` | 3,981 | CURRENT_CORE_SUPPORT |
| 26 | `docs/ultragoals` | 3,783 | IMPORTANT_ARCHIVE / CURRENT_META |
| 27 | `civilization/build_state.py` | 2,685 | CURRENT_CORE_SUPPORT |
| 28 | `tools/sovereign` | 2,602 | CURRENT_VERIFICATION |
| 29 | `tools/doctor` | 2,361 | CURRENT_VERIFICATION |
| 30 | `tools/hcli` | 2,297 | CURRENT_CORE_SUPPORT / MIGRATE_INTO_OWNER |

## Disposition rules for the next waves

- `research/lab`: registry-backed operators and their dependency clusters
  survive only when their current scientific owner or verifier is identified;
  unowned campaign chains are candidates for extraction to result/receipt/
  reproducer and deletion.
- `research/hawking-experiments`: retain decisive findings and minimal
  reproducers; delete superseded executable campaign machinery after checking
  provenance and negative controls. The G1 evidence runners are now retired;
  their result payloads and selected experiment notes remain the archival signal;
  eight unreferenced superseded reports and 17 unreferenced G1 reports were
  compressed into `G1_ARCHIVE_LEDGER.md`.
- `crates/hawking-core`, `tools/verify`, `tools/acceptance`, `tools/audit`, and
  `hcli/tests`: current verification is a valid survival reason; consolidate
  duplicated plumbing only when proof independence remains intact.
- `tools/future`, `tools/headless`, `tools/condense`, and `tools/odyssey`: audit
  complete dependency clusters against the current HCLI/Rust owner. A caller
  alone is not sufficient evidence of survival.
- `workspace/*`, `docs/*`, and `receipts/*`: classify information separately
  from executable machinery. Preserve current metadata, decisive evidence,
  Laws/Scars, and necessary provenance; compress or retire stale execution
  scaffolding where a smaller canonical representation carries the signal.

The next checkpoint is support <=500K, then <=250K, with the current product
boundary held at approximately 498K.
