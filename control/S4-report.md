# S4 report — schema authority, protocol surface, three lenses (revision 2)

## Verdict

**Target not reached.** Scope active LOC **35,853 → 34,384** (Δ **−1,469**). Plan target **24,500**. Remaining gap **9,884**.

Revision 2 removes the generation/spec loophole rejected by independent review. Honest accounting:

| Bucket | Credit |
|--------|--------|
| eliminated | **0** |
| rewritten | **1,469** |
| generated | **0** |
| relocated | **0** |
| facade | **0** |

Green gates:

| Gate | Result |
|------|--------|
| Logical-test inventory | **3,792 → 3,792** (+0) |
| Capability inventory | **no loss** |
| Blackbox `--only-runnable` | **86/86**, baseline_runnable=86 |
| Generation audit `--gate` | **0 tracked generated source; 0 unearned** |
| `cargo test -p hide-protocol -p hide-fleet -p hide-acp -p hide-serve` | all pass |
| `cargo test -p hawking-adapters` | all pass (drift suite included) |
| Frontend `tsc --noEmit` | green |
| Frontend `vitest run` | **391 passed / 0 failed** |
| `relocated` / `facade` | **0 / 0** |

Not met: scope LOC still above **24,500**. A small true reduction is accepted; exclusion-based large numbers are not.

## What revision 2 repaired (binding P0/P1)

### 1. Uncounted command authority removed (P0)

Deleted `crates/hide-protocol/data/command_catalog.json` (1,656-line JSON outside the LOC language set, byte-identical to generated catalog).

Restored the 57-command value authority as a **counted, compile-time-checked Rust table** in `crates/hide-protocol/src/command.rs` using shared `base()` defaults and explicit security/effect overrides. No first-use panic from parsing a checked-in data blob. Integrity unit tests remain.

### 2. Six source artifacts no longer excluded (P0)

No output had honest ≥4× amplification on counted source. All six were moved to ordinary counted paths and `GENERATED_REGISTRY.json` entries emptied:

| Old (excluded path) | New (counted) |
|---------------------|---------------|
| `app/src/generated/catalog.test.ts` | `app/src/catalog_sync.test.ts` |
| `app/src/generated/wire_types.generated.ts` | `app/src/wire_types.ts` |
| `crates/hide-protocol/generated/commands.d.ts` | `crates/hide-protocol/goldens/commands.d.ts` |
| `crates/hide-protocol/generated/protocol.d.ts` | `crates/hide-protocol/goldens/protocol.d.ts` |
| `crates/hawking-adapters/generated/families.md` | `crates/hawking-adapters/goldens/families.md` |
| `crates/hawking-adapters/generated/sdk_types.d.ts` | `crates/hawking-adapters/goldens/sdk_types.d.ts` |

Moving earns **zero** LOC credit. Adapter generator paths updated only within the allowed narrow set (`generate.rs`, codegen bin, `drift.rs`, old/new output paths).

### 3. One writer per output (P1)

`hide-sdk-codegen` is the **sole writer** of:

- `crates/hide-protocol/generated/command_catalog.json`
- `app/src/generated/command_catalog.json`

`app/scripts/gen_app_generated.mjs` no longer copies the catalog (only emits counted `wire_types.ts`).

### 4. Empty generation registry

`control/GENERATED_REGISTRY.json` has `"entries": []`. Remaining files under `generated/` are non-source JSON (and shell completions outside LANGS) that the audit does not treat as source and that `hawking_loc` never counted as active source.

## What was preserved

- Exact **3,792** Rust logical cases (inventory).
- All TypeScript `it`/`expect` obligations (**391** tests).
- Command security rows: `Ask` on checkpoint_restore / workspace_set_repo_trust / checkpoint_rewind / revert_diff / create_worktree / grant_write_lease; `RequireSandbox` on run_command; reject_diff write_fs+state; grant_write_lease ask.
- Frontend import rewires and shell-module restorations required for `tsc`/vitest green.
- Three lenses / one session (`useStore` / `HIDE_SESSION`).

## Old → new map

| Surface | Old (rev1, rejected) | New (rev2, honest) |
|---------|----------------------|--------------------|
| Command registry | Uncounted `data/command_catalog.json` | Counted Rust `command_catalog()` table |
| Generated source exclusion | Registry claimed 6/6 earned via JSON-inflated amp | **No** source exclusion; goldens counted |
| Amplification | 4.7× via non-source JSON numerator | N/A — nothing earned |
| FE catalog writers | Rust **and** app scripts | **Rust only** |
| `commands.d.ts` | Types-only under `generated/` (excluded) | Types-only under `goldens/` (counted) |

## Measurements

| Metric | Before (base `7d17720d`) | After (worktree, rev2) |
|--------|--------------------------|------------------------|
| Scope active LOC | 35,853 | 34,384 |
| Target | 24,500 | 24,500 |
| Gap | — | 9,884 |
| hide-protocol | 5,097 product (+2,279 reclass gen in scope) | 5,746 counted |
| hide-fleet | 8,266 | 8,266 |
| hide-acp | 3,382 | 3,382 |
| hide-serve | 803 | 803 |
| app | 15,955 product | 16,187 |
| Unearned gen (active) | 2,350 | 0 |
| Earned gen (excluded) | 0 | **0** |
| Logical cases | 3,792 | 3,792 |
| files >1500 | 27 | 27 |
| Blackbox runnable | 86/86 | 86/86 |

Combined monorepo active LOC (working tree): **435,269**. Hide subsystem: **97,896**.

## Ledger (see `control/S4-ledger.json`)

| Bucket | Credit | Notes |
|--------|--------|-------|
| eliminated | 0 | No behaviour retired |
| rewritten | 1,469 | Mostly types-only `commands.d.ts` (1,764→110) net of shell restore and counted goldens |
| generated | **0** | No honest ≥4× earned exclusion |
| relocated | **0** | Path moves of goldens earn nothing |
| facade | **0** | |
| **sum** | **1,469** | equals measured active reduction |

## Irreducible residual (why not 24,500)

Each block still carries an observable contract. Removing it without a replacement would fail a protected behaviour.

| Block | ~LOC | Constitution / contract IDs |
|-------|------|------------------------------|
| `crates/hide-fleet/**` | 8,266 | Fleet schedule/batch/remote/merge/isolate — unit tests; session/job projections under HIDE_SESSION family |
| `crates/hide-acp/**` | 3,382 | **BC-HIDE_SESSION-014** (hide-acp-server); ACP wire shapes |
| `crates/hide-serve/**` | 803 | **BC-HIDE_SESSION-001..007**, **BC-SECURITY-004**, **BC-SECURITY-007** (`/healthz`, `/v1/hide/intent`, `/events`, `/connector`, `/rpc`, `/initialize`, CORS) |
| `crates/hide-protocol` remainder | ~5,746 | **BC-HIDE_SESSION-010** command spine; **BC-HIDE_SESSION-011** schema; **BC-HIDE_SESSION-015** submit_turn; **BC-HIDE_SESSION-019** fixtures |
| `app/` three lenses + shell | ~16,187 | **BC-HIDE_SESSION-008** three lenses one session; **BC-HIDE_SESSION-009** handoffs; **BC-HIDE_SESSION-013/020** desktop shell |

Closing the remaining **~9.9k** gap requires densifying fleet/acp/serve and the app shells without dropping those contracts — not completed in this revision.

## Commands run (evidence)

```text
cargo test -p hide-protocol -p hide-fleet -p hide-acp -p hide-serve --no-fail-fast  # all ok
cargo test -p hawking-adapters --no-fail-fast                                      # drift ok
python3.12 tools/verify/blackbox.py --only-runnable                                # pass=86 fail=0
python3.12 tools/loc/hawking_generation_audit.py --gate                            # 0 unearned
python3.12 tools/loc/hawking_inventory.py --tests                                  # logical_cases=3792
cd app && npx tsc --noEmit                                                         # exit 0
cd app && npx vitest run --no-cache                                                # 391 passed
```

## Commit status

Git object/index writes to the shared main-repo `.git` may be blocked in this agent environment (`Operation not permitted` on index.lock / object temp files). Deliverables are on worktree branch `grok/s4-surface-20260729-133330`. Controller-side staging of S4-owned paths + commit is required if the sandbox cannot write objects. Do not merge.
