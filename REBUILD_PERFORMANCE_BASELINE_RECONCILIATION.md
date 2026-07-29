# Rebuild performance baseline — historical reconciliation

Instrument: `tools/verify/perfgate.py`  
Capture: `REBUILD_PERFORMANCE_BASELINE_MEASURED.json`  
Host load at capture start: **~42 / 41 / 43** (1/5/15) on 28 CPUs — **not a clean room**.  
Metal probe in this session: **unavailable** (`metal: no Metal-capable GPU` — sandboxed executor; hardware is M3 Ultra with Metal Supported outside the sandbox).

## How much of the 2% gate is enforceable today?

| Gate clause | Enforceable now? | Why |
|---|---|---|
| no >2% regression in **base TPS** | **No** | Metal blocked in this capture environment; no measured `base_tps.*` rows. Artifacts may exist on disk (see below) but the instrument correctly records `unavailable`, not a synthetic proxy. |
| no >2% regression in **accelerated TPS** | **No** | Metal blocked; also no GGUF under `models/` for `hawking bench --profile fast`. |
| no >2% regression in **transformation throughput** | **Yes** | Three measured transform metrics (format selftest wall, shard write/verify bytes/s, pack_indices bytes/s), n=7 kept each. |
| no material **startup/compile** regression without trade receipt | **Partial** | Measured: `cargo check`, warm release rebuild after touch, binary size, `--help`, `version`. Cold release build opt-in (`--include-cold`). Doctor skipped (no GGUF). Kernel microbench blocked on Metal. |

**Bottom line:** On this capture, the 2% gate is enforceable for **build/startup (partial)** and **transformation throughput**. The campaign’s headline TPS gates (base + accelerated) are **blocked on environment + artifacts**, not on missing harness code. Re-run `--capture` under an unsandboxed profile with Metal (and GGUF if accelerated is required) to promote those rows to `measured`.

## Historical records vs this capture

| Historical file / claim | Historical number | Fixture / command | Reproduced? | Notes |
|---|---|---|---|---|
| `HAWKING_BASE_TRUE_TPS.json` (2026-07-26) | warm **0.3961** tps, cold **0.2942** tps | GLM-5.2-H0.98-Math-Preserve.gravity; context 4 / 80 decode; verify_hash | **Cannot test (Metal)** | Artifact dir still present (~86 GiB) at `~/Library/Application Support/Hawking/Models/GLM-5.2/…/GLM-5.2-H0.98-Math-Preserve.gravity`. `gravity_glm_tps` binary exists under main `target/release/examples/`. Capture marks metric `unavailable` (Metal) / default-skips multi-minute GLM unless `--include-glm-tps`. |
| `GLM52_MATH_PRESERVE_BASE_TPS.json` (earlier) | ~**0.149–0.151** tps @ ctx 4/128 | same Math-Preserve path, 12 decode tokens | **Cannot test (Metal)** | Superseded upward by memoization fix in `HAWKING_BASE_TRUE_TPS.json`. Same fixture class. |
| `HAWKING_GRAVITY_BASE_TPS.json` (2026-07-24) | **105.8** tps @ ctx 128 / 16 decode (also 68.8 @ 512, 29.2 @ 2048, 13.3 @ 8192) | `llama32-1b-R0.v2.gravity` (129 MiB) via `gravity_tps` | **Cannot test (Metal)** | Artifact **still on disk** at `CampaignS08/llama32-1b-R0.v2.gravity`. Best candidate for a short base-TPS re-measure once Metal is available. Dispatches/token historically 210. |
| `docs/BENCHMARKS.md` / `tools/bench/compare_sota.sh` | Qwen GGUF vs llama.cpp / MLX | `models/Qwen2.5-7B-Instruct-Q4_K_M.gguf` etc. | **Cannot test (fixture gone)** | No `*.gguf` under worktree or main `models/` in this inventory. Accelerated/doctor paths therefore `unavailable` with explicit path reasons. |
| `BASELINES.md` external set | llama.cpp Q4_K_M, MLX 4-bit, etc. | tuned competitor commands | **Out of scope for perfgate** | Neutrality spec, not a single instrument receipt. Perfgate does not re-run competitor benches. |
| Warm release rebuild (this capture) | median **~79 s** (min 75 / max 98) after touch of `crates/hawking/src/main.rs` | `cargo build -p hawking --release` into `/tmp/hawking-perfgate-target` | **New measured baseline** | Contaminated by load ~40+; use paired mode for A/B. |
| `cargo check -p hawking` (this capture) | median **~0.36 s** | touch leaf + check | **New measured baseline** | |
| Binary size (this capture) | **7 555 168** bytes | `HAWKING_BIN` → main checkout `target/release/hawking` | **New measured baseline** | |
| Transform pack_indices (this capture) | median **~4.67e7** bytes/s | fixture 2e6 indices × 8 round-trips | **New measured baseline** | Lab path without 1.4 TB source. |
| Transform shard write/verify (this capture) | median **~1.53e8** bytes/s | 32×4 KiB synthetic tensors, 20 loops | **New measured baseline** | `gravity_format` only (no torch/MPS). |
| gravity_format selftest (this capture) | median **~0.060 s** | `tools/condense/gravity_format.py selftest` | **New measured baseline** | Also used as numeric-parity container oracle timing (~0.061 s). |

## What is still real on disk

| Path | Status |
|---|---|
| `…/CampaignS08/llama32-1b-R0.v2.gravity` (129 MiB) | Present — historical llama base TPS fixture |
| `…/Models/GLM-5.2/…/GLM-5.2-H0.98-Math-Preserve.gravity/` (~86 GiB) | Present — historical GLM BASE_TRUE_TPS fixture |
| `crates/hawking-core/tests/fixtures/gravity_glm/glm52-tiny-R0.gravity` | Present — unit-test scale only, not scoreboard TPS |
| `models/*.gguf` | **Absent** — SOTA/accelerated/doctor GGUF path blocked |
| Main `target/release/hawking` + `examples/gravity_tps` | Present |

## Gate verification of the instrument

| Check | Result |
|---|---|
| `--list` honest inventory | 9 measurable, 1 skipped (cold), 6 unavailable with reasons |
| `--capture` | Wrote `REBUILD_PERFORMANCE_BASELINE_MEASURED.json` |
| `--compare` baseline vs itself `--gate 2.0` | **pass** (exit 0) |
| `--compare` baseline vs 5% perturbed copy | **fail** (exit 1), 9 measured metrics red |
| `--paired` ABAB smoke | Sign test p≈0.016 on n=7 for +5% B shift |

## Recommended next capture (unsandboxed / gate profile)

```bash
# Unsandboxed so Metal works; quit other heavy jobs if absolute numbers matter,
# or rely on --paired for rebuild vs tip.
export HAWKING_BIN=…/target/release/hawking
export CARGO_TARGET_DIR=…   # writable
python3.12 tools/verify/perfgate.py --list
python3.12 tools/verify/perfgate.py --capture --out REBUILD_PERFORMANCE_BASELINE_MEASURED.json --n 8
# optional multi-minute GLM scoreboard:
#   … --include-glm-tps
# optional cold compile series:
#   … --include-cold
```

Until that run, treat historical TPS JSON files as **archival receipts**, not live gate thresholds — the instrument will not silently pass on empty measurements.
