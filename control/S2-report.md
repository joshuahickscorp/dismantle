# S2b report — process-engine cutover (revision 6)

## Verdict

**Narrow gates green; counted Python authority restored.** Independent final
review pending. Worktree uncommitted. `lab/` is the sole flat process-engine
authority; `tools/condense/engine/**` is deleted. Science producers remain under
`tools/condense/`. No second science body, Gravity container, shim, or toy in `lab/`.

## Authority repair (r6)

r5 introduced `lab/irreducible_modules.json` as runtime authority for 43 Track V
module records — uncounted JSON (S4 command-catalog loophole). **Forbidden.**
r6 deletes it; all 43 records live in counted `lab/science_registry.py` via shared
`_m`, class aliases, and default omission. No JSON/TOML/YAML catalog, packing, or
generated Python.

## Provenance (binding A)

Frozen terminal receipts bind to exact historical git blobs, not live HEAD.

| Receipt | Instrument binding |
|---------|-------------------|
| `GRAVITY_EXTERNAL_BASELINE_MATRIX.json` | generator `2ee17e0e:…/glm52_external_baselines.py` + common `753c73dc:…/glm52_common.py` |
| `GLM52_ADAPTER_TWIN.json` / `GLM52_REFERENCE_PARITY.json` | `local_source_sha256` pins at `2a7fddad` |
| `GLM52_CORPUS_INTEGRITY.json` | instruments at `2a7fddad`; tokenizers sealed digest only |

Validators use path+commit+digest pins; no live HEAD re-hash for frozen receipts.
Root sealed artifacts were not resealed.

## What landed

| Item | Disposition |
|------|-------------|
| `lab/` | Flat IR, runtime, lease, checkpoint, receipts, governance, Track V class |
| `tools/condense/engine/` | Deleted → `lab/` |
| Science producers | Unchanged under `tools/condense/` |
| Terminal proofs | Historical binding for external + adapter + corpus |
| Capability | 1 invocable: `python3.12 -m lab --classify` |
| Inventories | `control/rungs/post-s2b.{tests,caps}.json` refreshed |

## Measurements

### Lab package

| Metric | Value |
|--------|-------|
| Flat lab Python physical LOC | 2392 |
| Generated lab LOC | 0 |
| Irreducible module records | 43 (Python authority) |
| Max line in `lab/*.py` | ≤120 |

### Main projection `ea33af24` + intended S2 paths

Path-arithmetic and archive-paired delta agree.

| Metric | Base | Projected | Δ | Cap |
|--------|------|-----------|---|-----|
| directories_all | 138 | 138 | 0 | ≤138 |
| directories_leaf | 110 | 110 | 0 | ≤110 |
| source_files | 1209 | 1207 | −2 | ≤1209 |
| rust_crates | 20 | 20 | 0 | ≤20 |
| public_symbols | 9549 | 9542 | −7 | ≤9549 |
| functions | 14691 | 14689 | −2 | ≤14691 |
| files_over_1500 | 26 | 26 | 0 | ≤26 |
| tiny_forwarders | 13 | 13 | 0 | ≤13 |
| Honest active LOC | 437285 | 437176 | −109 | <437285 |
| Generated | 0 | 0 | 0 | 0 |

Margin below base: **109**. Every topology dimension holds or decreases.

### Six-bucket ledger (no unearned credit)

| Bucket | LOC | Notes |
|--------|-----|-------|
| rewritten | 558 | old engine 2950 − new lab 2392 |
| proof/test apparatus | +338 | terminal proofs + campaign tests + shell |
| report apparatus | this `S2-report.md` | counted markdown |
| generated / relocated / facade | 0 | |

## Narrow gates

| Gate | Result |
|------|--------|
| lab imports / compileall | PASS |
| classification 43-record parity | PASS (exact vs pre-JSON authority) |
| `python3.12 -m lab --help` / `--classify` | PASS |
| terminal proofs + campaign engine | **119 passed** |
| capability_manifest `--check` | 1 invocable / 0 not |
| inventories | tests 3967; caps python_entrypoints 197 |
| generation_audit | no new S2 unearned files |
| blackbox `--only-runnable` | **86/86** |
| deleted-engine import scan | CLEAN |
| line length / whitespace | lab ≤120; clean |
| root frozen receipts resealed | **no** |

## Old → new (process slice)

| Behaviour | Old | New |
|-----------|-----|-----|
| Spec IR | `engine/spec.py` | `lab/spec.py` + `lab/campaigns.json` |
| Run/resume | `engine/runtime.py` | `lab/runtime.py` (`python3.12 -m lab`) |
| Lease / checkpoint | `engine/lease.py`, `checkpoint.py` | `lab/lease.py`, `lab/checkpoint.py` |
| Seal integrity | `engine/seal_integrity.py` | `lab/receipts.py` |
| Operators class | `engine/operators.py` | `lab/science_registry.py` (counted Python) |
| Science bodies | `tools/condense/*.py` | **unchanged** |

## Commit posture

Worktree **uncommitted** pending independent final review. Never pushed.
Forbidden root receipts remain byte-identical to `HEAD`.
