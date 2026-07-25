//! BASE_TRUE_TPS for the GLM-5.2 GPU-resident `.gravity` path.
//!
//! Companion to `gravity_tps.rs` (Llama). Two things do not carry over from
//! the dense case: device-resident bytes grow with the run instead of being
//! fixed at load (only touched experts are ever uploaded), and content
//! affects cost, since different tokens can route to different experts.
//! Both are called out in the receipt rather than assumed away.
//!
//! Defaults are deliberately small -- this is a first honest measurement on
//! an 83 GB, 78-layer MoE artifact, not a scoreboard run. Scale up with
//! `--context` once the small numbers are in.
//!
//!     cargo run --release -p hawking-core --example gravity_glm_tps -- \
//!         --context 16 --decode 8 --out receipt.json

use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_MODEL_DIR: &str = "Library/Application Support/Hawking/Models/GLM-5.2/\
    b4734de4facf877f85769a911abafc5283eab3d9/General-R0";

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hawking_core::gravity_glm::gpu::GravityGlmGpu;

    let mut dir: Option<PathBuf> = None;
    let mut contexts: Vec<usize> = Vec::new();
    let mut decode = 4usize;
    let mut out: Option<PathBuf> = None;
    let mut verify_hash = true;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = args.next().map(PathBuf::from),
            "--context" => contexts.push(args.next().ok_or("--context needs a value")?.parse()?),
            "--decode" => decode = args.next().ok_or("--decode needs a value")?.parse()?,
            "--out" => out = args.next().map(PathBuf::from),
            "--no-verify-hash" => verify_hash = false,
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    if contexts.is_empty() {
        contexts = vec![8, 16];
    }
    let dir = dir.unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(DEFAULT_MODEL_DIR)
    });
    if !dir.is_dir() {
        return Err(format!("no model directory at {dir:?}").into());
    }

    eprintln!("opening (indexing shard headers, decoding nothing)...");
    let t_open = Instant::now();
    let model = GravityGlmGpu::open_dir(&dir, verify_hash)?;
    let open_ms = t_open.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "opened in {open_ms:.0} ms | layers={} hidden={} experts={} vocab={}",
        model.arch.n_layers,
        model.arch.hidden,
        model.arch.n_routed_experts,
        model.arch.vocab_size
    );

    // Deterministic pseudo-token stream inside the vocabulary, continued
    // (not restarted) from prefill into decode so a run is reproducible.
    let vocab = model.arch.vocab_size as u64;
    let stream = |n: usize| -> Vec<u32> {
        (0..n)
            .map(|i| ((i as u64 * 2_654_435_761) % vocab) as u32)
            .collect()
    };

    let mut rows = Vec::new();
    for &context in &contexts {
        let tokens = stream(context + decode);

        let t_prefill = Instant::now();
        model.forward(&tokens[..context])?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;

        let mut decode_ms_each = Vec::with_capacity(decode);
        let mut logits = Vec::new();
        for (i, &t) in tokens[context..].iter().enumerate() {
            let t0 = Instant::now();
            let (l, _) = model.forward_at(&[t], context + i)?;
            decode_ms_each.push(t0.elapsed().as_secs_f64() * 1e3);
            logits = l;
        }
        assert_eq!(logits.len(), model.arch.vocab_size);

        let decode_ms: f64 = decode_ms_each.iter().sum();
        let mut sorted = decode_ms_each.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        rows.push(serde_json::json!({
            "context_tokens": context,
            "decode_tokens": decode,
            "prefill_ms": prefill_ms,
            "prefill_tps": context as f64 / (prefill_ms / 1e3),
            "decode_ms": decode_ms,
            "base_true_decode_tps": decode as f64 / (decode_ms / 1e3),
            "decode_ms_per_token_median": sorted.get(sorted.len() / 2).copied().unwrap_or(0.0),
            "decode_ms_per_token_min": sorted.first().copied().unwrap_or(0.0),
            "decode_ms_per_token_max": sorted.last().copied().unwrap_or(0.0),
            "decode_ms_per_token_all": decode_ms_each,
        }));
        let last = rows.last().unwrap();
        eprintln!(
            "ctx {context:>4}  prefill {:>7.2} tok/s ({prefill_ms:>8.0} ms)  decode {:>7.2} tok/s  \
             ({:.1} ms/tok median)",
            last["prefill_tps"].as_f64().unwrap(),
            last["base_true_decode_tps"].as_f64().unwrap(),
            last["decode_ms_per_token_median"].as_f64().unwrap(),
        );
    }

    let receipt = serde_json::json!({
        "schema": "hawking.gravity.glm_base_tps.v1",
        "scoreboard": "BASE_TRUE_TPS",
        "note": "measured, not modelled; no acceleration of any kind is enabled on this path. \
                 GLM's routed MoE means device-resident bytes grow with the run rather than \
                 being fixed at load, and cost is mildly content-dependent, unlike the dense \
                 Llama instrument gravity_tps.rs measures.",
        "model_dir": dir.to_string_lossy(),
        "architecture": {
            "layers": model.arch.n_layers,
            "hidden": model.arch.hidden,
            "routed_experts": model.arch.n_routed_experts,
            "experts_per_tok": model.arch.num_experts_per_tok,
            "vocab": model.arch.vocab_size,
        },
        "open_ms": open_ms,
        "measurements": rows,
    });
    let text = serde_json::to_string_pretty(&receipt)? + "\n";
    match out {
        Some(p) => std::fs::write(&p, &text)?,
        None => print!("{text}"),
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gravity_glm_tps measures the Metal runtime; it only runs on macOS");
}
