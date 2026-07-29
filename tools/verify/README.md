# `tools/verify` — rebuild performance gate

Instrument that decides the rebuild hard gate:

- no >2% regression in base TPS
- no >2% regression in accelerated TPS
- no >2% regression in transformation throughput
- no material startup/compile regression without a measured trade receipt

## Commands

```bash
python3.12 tools/verify/perfgate.py --list
python3.12 tools/verify/perfgate.py --capture --out REBUILD_PERFORMANCE_BASELINE_MEASURED.json
python3.12 tools/verify/perfgate.py --compare A.json B.json --gate 2.0
python3.12 tools/verify/perfgate.py --paired --a-cmd '…' --b-cmd '…' --n 9
```

Stdlib only (+ `statistics`). Optional: existing release binary (`HAWKING_BIN`), GGUF (`HAWKING_GGUF`), `CARGO_TARGET_DIR`.

## Design rules

1. **No silent empty measurements.** Every metric is `measured` | `skipped` | `unavailable` with a reason. Compare exits non-zero if a metric that was measured in A is no longer measured in B.
2. **No fabricated TPS.** Base / accelerated TPS need Metal + a real artifact. If either is missing, status is `unavailable` — never a synthetic proxy labeled TPS.
3. **Contamination is assumed.** This box is not a clean room. Every sample records 1/5/15 loadavg, free/active memory, and (when `ps` is permitted) whether another process held >4 cores. Prefer `--paired` (ABAB interleave + sign test) over absolute numbers hours apart.
4. **Statistics.** `n` runs including 1 discarded warm-up; report median and min–max, never mean alone. Protocol default `n=8` ⇒ 7 kept samples.

## Metric families

| Family | What | When unavailable |
|---|---|---|
| `build` | `cargo check`, warm release build (touch leaf), optional cold (`--include-cold`), binary size | no cargo / no binary |
| `startup` | `hawking --help`, `version`, `doctor --json` | no binary; doctor needs GGUF |
| `base_tps` | `gravity_tps` on llama-1B `.gravity`; optional GLM Math-Preserve (`--include-glm-tps`) | no Metal, no artifact, no example binary |
| `accelerated_tps` | `hawking bench --suite decode --profile fast` | no Metal / no GGUF |
| `transform` | `gravity_format` selftest; fixture-scale shard write/verify bytes/s; `glm52_pack` pack_indices bytes/s | missing lab scripts |
| `kernel` | `bench-q4k-shapes` (no model); `bench-kernel` marked unavailable (CLI extracted) | no Metal |
| `numeric_parity` | `gravity_format` write/verify/tamper path (CPU container oracle) | missing script |

## Compare semantics

For each metric measured on both sides, compute `delta_pct_improvement` (positive = B better). Fail if improvement &lt; `−gate` (default 2%). Higher-is-better (tps, bytes/s) and lower-is-better (seconds, bytes, µs) are handled explicitly.

## Env

| Variable | Purpose |
|---|---|
| `HAWKING_BIN` | Path to `hawking` binary |
| `HAWKING_GGUF` | Path to a GGUF for doctor / accelerated bench |
| `CARGO_TARGET_DIR` | Shared cargo target (defaults to repo or main checkout target) |
