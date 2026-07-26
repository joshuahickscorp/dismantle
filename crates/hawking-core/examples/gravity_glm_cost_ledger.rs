//! Per-token cost ledger for the sealed GLM-5.2 Math-Preserve GPU path.
//!
//! Implements the first required deliverable of `BASE_RUNTIME_MAXIMIZED_GATE.md`:
//! attribute every microsecond of one decode token so the buckets sum to
//! measured wall time, with an explicit unattributed remainder.
//!
//! Default-off instrumentation (`HAWKING_COST_LEDGER=1`). Does not change
//! decode output. This binary only runs on macOS with Metal; the executor
//! sandbox cannot create a Metal device, so hand the controller this command.
//!
//! Warm ledger (after a short residency warm-up), one attributed token:
//!
//! ```text
//! HAWKING_COST_LEDGER=1 cargo run --release -p hawking-core \
//!   --example gravity_glm_cost_ledger -- \
//!   --context 4 --warmup 8 --ledger-tokens 1 \
//!   --out reports/base_runtime/cost_ledger_warm.json
//! ```
//!
//! Cold first-token ledger (no warm-up; first-touch loads dominate):
//!
//! ```text
//! HAWKING_COST_LEDGER=1 cargo run --release -p hawking-core \
//!   --example gravity_glm_cost_ledger -- \
//!   --context 4 --warmup 0 --ledger-tokens 1 --no-verify-hash \
//!   --out reports/base_runtime/cost_ledger_cold_nohash.json
//! ```

use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_MODEL_DIR: &str = "Library/Application Support/Hawking/Models/GLM-5.2/\
    b4734de4facf877f85769a911abafc5283eab3d9/General-R0";

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hawking_core::cost_ledger::{self, SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES};
    use hawking_core::gravity_glm::gpu::GravityGlmGpu;

    // Force the ledger on for this process even if the env var was omitted
    // (the example's whole point is the ledger). Env still wins if set to 0
    // via set_enabled after resolve — we set explicitly.
    cost_ledger::set_enabled(true);

    let mut dir: Option<PathBuf> = None;
    let mut context = 4usize;
    let mut warmup = 8usize;
    let mut ledger_tokens = 1usize;
    let mut out: Option<PathBuf> = None;
    let mut verify_hash = true;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = args.next().map(PathBuf::from),
            "--context" => context = args.next().ok_or("--context needs a value")?.parse()?,
            "--warmup" => warmup = args.next().ok_or("--warmup needs a value")?.parse()?,
            "--ledger-tokens" => {
                ledger_tokens = args.next().ok_or("--ledger-tokens needs a value")?.parse()?
            }
            "--out" => out = args.next().map(PathBuf::from),
            "--no-verify-hash" => verify_hash = false,
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let dir = dir.unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(DEFAULT_MODEL_DIR)
    });
    if !dir.is_dir() {
        return Err(format!("no model directory at {dir:?}").into());
    }

    eprintln!(
        "cost ledger | verify_hash={verify_hash} context={context} warmup={warmup} \
         ledger_tokens={ledger_tokens}"
    );
    eprintln!("model dir: {}", dir.display());

    let t_open = Instant::now();
    let model = GravityGlmGpu::open_dir(&dir, verify_hash)?;
    let open_ms = t_open.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "opened in {open_ms:.0} ms | layers={} hidden={} experts_per_tok={} vocab={}",
        model.arch.n_layers,
        model.arch.hidden,
        model.arch.num_experts_per_tok,
        model.arch.vocab_size
    );

    let vocab = model.arch.vocab_size as u64;
    let stream = |n: usize| -> Vec<u32> {
        (0..n)
            .map(|i| ((i as u64 * 2_654_435_761) % vocab) as u32)
            .collect()
    };
    let total = context + warmup + ledger_tokens;
    let tokens = stream(total);

    // Prefill: not ledgered (the gate asks for decode-token attribution).
    let t_prefill = Instant::now();
    model.forward(&tokens[..context])?;
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    eprintln!("prefill {context} tokens in {prefill_ms:.0} ms");

    // Warm-up decode tokens without the ledger (residency curve, not cost).
    for (i, &t) in tokens[context..context + warmup].iter().enumerate() {
        let t0 = Instant::now();
        let _ = model.forward_at(&[t], context + i)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "  warmup {:>3}/{}  {ms:>8.1} ms  {:.4} tok/s",
            i + 1,
            warmup,
            1000.0 / ms
        );
    }

    let mut token_reports = Vec::with_capacity(ledger_tokens);
    for (i, &t) in tokens[context + warmup..].iter().enumerate() {
        let abs_pos = context + warmup + i;
        let wall = Instant::now();
        let (logits, _trace, report) = model.forward_at_with_ledger(&[t], abs_pos)?;
        let wall_ms = wall.elapsed().as_secs_f64() * 1e3;
        assert_eq!(logits.len(), model.arch.vocab_size);

        let report = report.ok_or("ledger enabled but no report returned")?;
        eprintln!(
            "  ledger {:>3}/{}  wall {:>8.1} ms  attributed {:.1}%  unattributed_us={}  \
             cbs={} disp={} syncs={} active_bytes={:.3} GB",
            i + 1,
            ledger_tokens,
            wall_ms,
            report.attributed_fraction * 100.0,
            report.unattributed_us,
            report.counters.command_buffers_submitted,
            report.counters.dispatches_encoded,
            report.counters.synchronization_points,
            report.counters.active_bytes_read as f64 / 1e9,
        );
        // Print bucket table for the operator.
        eprintln!("  buckets (us, exclusive):");
        let mut keys: Vec<_> = report.buckets_us.keys().cloned().collect();
        keys.sort();
        for k in keys {
            let us = report.buckets_us[&k].as_u64().unwrap_or(0);
            if us == 0 {
                continue;
            }
            let pct = 100.0 * us as f64 / report.wall_us.max(1) as f64;
            eprintln!("    {k:<32} {us:>12}  ({pct:5.1}%)");
        }
        eprintln!(
            "    {:<32} {:>12}  ({:5.1}%)",
            "unattributed",
            report.unattributed_us,
            100.0 * report.unattributed_us as f64 / report.wall_us.max(1) as f64
        );

        token_reports.push(serde_json::json!({
            "decode_token_index": i + 1,
            "absolute_position": abs_pos,
            "external_wall_ms": wall_ms,
            "ledger": report.to_json_value(),
        }));
    }

    let cache = model.cache_stats();
    let receipt = serde_json::json!({
        "schema": "hawking.gravity.per_token_cost_ledger_run.v1",
        "gate": "BASE_RUNTIME_MAXIMIZED",
        "deliverable": "per-token cost ledger",
        "verify_hash": verify_hash,
        "model_dir": dir.to_string_lossy(),
        "architecture": {
            "layers": model.arch.n_layers,
            "hidden": model.arch.hidden,
            "routed_experts": model.arch.n_routed_experts,
            "experts_per_tok": model.arch.num_experts_per_tok,
            "vocab": model.arch.vocab_size,
        },
        "geometry_active_routed_bytes_sealed": SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES,
        "open_ms": open_ms,
        "prefill_ms": prefill_ms,
        "context_tokens": context,
        "warmup_decode_tokens": warmup,
        "ledger_tokens": ledger_tokens,
        "gpu_weight_cache": {
            "budget_bytes": cache.budget_bytes,
            "resident_bytes": cache.resident_bytes,
            "high_water_bytes": cache.high_water_bytes,
            "entries": cache.entries,
            "evictions": cache.evictions,
        },
        "tokens": token_reports,
        "notes": [
            "Exclusive stack timing: nested metal/verify/transfer steal time from parent semantic scopes so sum(buckets)+unattributed ≈ wall.",
            "Routed experts and the shared expert are co-batched into three command buffers (gate/up/down); metal_* owns that GPU time, shared_experts owns only the CPU residual add.",
            "Attention (including DSA IndexShare) runs on the CPU today; its matvecs still go through the GPU PQ path and land in metal_*.",
            "An unattributed remainder is a finding — do not redistribute it into neighbours.",
            "This binary does not claim timings it did not measure; empty buckets mean that work was not observed on the instrumented path.",
        ],
    });

    let text = serde_json::to_string_pretty(&receipt)? + "\n";
    match out {
        Some(p) => {
            if let Some(parent) = p.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            std::fs::write(&p, &text)?;
            eprintln!("wrote {}", p.display());
        }
        None => print!("{text}"),
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gravity_glm_cost_ledger measures the Metal runtime; it only runs on macOS");
}
