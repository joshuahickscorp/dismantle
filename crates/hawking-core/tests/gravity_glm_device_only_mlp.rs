//! Lane D — ordinary three-batch device-only SiLU on the resident MLP.
//!
//! Flag `HAWKING_GLM_GPU_DEVICE_ONLY_MLP=1` (default off). Reuses the existing
//! `gravity_silu_mul_f32` kernel; does **not** enable expert-wave.
//!
//! Acceptance (same process, warm fixture, real Metal):
//! - Numeric Parity V2.1 continuous **and** discrete pass
//! - device waits/token ≤ baseline (`forward_resident_counted`)
//! - physical command buffers/token differ (candidate executed)
//! - p50 and p95 wall improve, or report honest negative and leave flag off
//! - mlp gate/up download + activation upload transfer counters are zero
//! - causal mutation: flag off or SiLU poison must fail the hit/parity gates
//!
//! Reference oracle for SiLU is a **separate** device dispatch scored against
//! FP64 authority — not the host path (non-circular).
//!
//! ```text
//! cargo test -p hawking-core --test gravity_glm_device_only_mlp -- --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::time::Instant;

use hawking_core::cost_ledger;
use hawking_core::gravity_glm::gpu::GravityGlmGpu;
use hawking_core::gravity_glm_resident::{
    device_only_mlp_fallbacks, device_only_mlp_hits, gpu_device_only_mlp_enabled,
    reset_device_only_mlp_probe, GPU_DEVICE_ONLY_MLP_ENV, GPU_DEVICE_ONLY_MLP_POISON_ENV,
};
use hawking_core::metal::{MetalContext, TokenCommandBuffer};
use hawking_core::numeric_parity::{
    format_score_line, silu_mul_f32_host, silu_mul_f64_authority, score_pair, Bounds, SCHEMA,
};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gravity_glm")
}

fn require_metal() -> Option<MetalContext> {
    match MetalContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("shader") && !msg.contains("compile"),
                "Metal is present but the shader failed to compile -- this is a real \
                 failure, not a skip: {msg}"
            );
            // Gate profile re-run: set HAWKING_REQUIRE_METAL=1 so a sandboxed
            // empty-GPU environment cannot green-skip the acceptance suite.
            if std::env::var_os("HAWKING_REQUIRE_METAL").is_some() {
                panic!(
                    "HAWKING_REQUIRE_METAL is set but no Metal device is visible ({e}). \
                     Re-run under the gate profile (sandbox=off)."
                );
            }
            eprintln!("skip: no Metal device ({e})");
            None
        }
    }
}

fn open_resident(ctx: MetalContext) -> GravityGlmGpu {
    GravityGlmGpu::open_dir_with_budget_resident(
        ctx,
        &fixtures_dir(),
        true,
        256 * 1024 * 1024,
        true,
    )
    .expect("resident open")
}

fn prompt() -> Vec<u32> {
    #[derive(serde::Deserialize)]
    struct Ref {
        tokens: Vec<u32>,
    }
    let r: Ref = serde_json::from_slice(
        &std::fs::read(fixtures_dir().join("ref_glm.json")).expect("ref_glm"),
    )
    .expect("parse");
    r.tokens
}

fn top1(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .min_by(|(i, a), (j, b)| {
            b.partial_cmp(a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(i.cmp(j))
        })
        .map(|(i, _)| i as u32)
        .expect("non-empty")
}

fn percentile_nearest(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let n = sorted.len();
    let rank = ((p / 100.0) * n as f64).ceil().max(1.0) as usize;
    sorted[rank.min(n) - 1]
}

fn dispatch_silu_device_ref(
    ctx: &MetalContext,
    gate: &[f32],
    up: &[f32],
) -> Result<Vec<f32>, String> {
    assert_eq!(gate.len(), up.len());
    let n = gate.len() as u32;
    let gate_buf = ctx
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(gate))
        .map_err(|e| e.to_string())?;
    let up_buf = ctx
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(up))
        .map_err(|e| e.to_string())?;
    let out_buf = ctx
        .new_buffer_checked(gate.len() * 4)
        .map_err(|e| e.to_string())?;
    const TG: u32 = 256;
    let mut tcb = TokenCommandBuffer::new(ctx);
    tcb.dispatch_threads(
        "gravity_silu_mul_f32",
        (n.div_ceil(TG) * TG, 1, 1),
        (TG, 1, 1),
        |enc| {
            enc.set_buffer(0, Some(&gate_buf), 0);
            enc.set_buffer(1, Some(&up_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
        },
    )
    .map_err(|e| e.to_string())?;
    tcb.commit_and_wait().map_err(|e| e.to_string())?;
    let ptr = out_buf.contents() as *const f32;
    Ok(unsafe { std::slice::from_raw_parts(ptr, gate.len()) }.to_vec())
}

fn with_env(key: &str, value: Option<&str>, f: impl FnOnce()) {
    let prev = std::env::var_os(key);
    match value {
        Some(v) => std::env::set_var(key, v),
        None => std::env::remove_var(key),
    }
    f();
    match prev {
        Some(v) => std::env::set_var(key, v),
        None => std::env::remove_var(key),
    }
}

/// Separate device-SiLU reference vs FP64 (non-circular). Host is also scored
/// for information only.
#[test]
fn device_silu_reference_v21_against_f64() {
    let Some(ctx) = require_metal() else {
        return;
    };
    eprintln!("numeric parity schema={SCHEMA}");
    let bounds = Bounds::continuous_only();
    let n = 257usize;
    let gate: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 - 0.5).collect();
    let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.25).collect();
    let g64: Vec<f64> = gate.iter().map(|&v| v as f64).collect();
    let u64v: Vec<f64> = up.iter().map(|&v| v as f64).collect();
    let reference = silu_mul_f64_authority(&g64, &u64v).expect("f64");
    let host = silu_mul_f32_host(&gate, &up).expect("host");
    let device = dispatch_silu_device_ref(&ctx, &gate, &up).expect("device ref");
    let paired = score_pair(&host, &device, &reference, &bounds);
    eprintln!("{}", format_score_line(&paired.host));
    eprintln!("{}", format_score_line(&paired.device));
    assert!(
        paired.pass && paired.device.pass,
        "device SiLU reference must pass V2.1 continuous: host_fail={:?} device_fail={:?}",
        paired.host.failures,
        paired.device.failures
    );
    // Continuous pass must be true — not only discrete.
    assert!(
        paired.device.pass,
        "V2.1 continuous pass=false for device SiLU reference"
    );
}

#[test]
fn device_only_mlp_flag_defaults_off() {
    with_env(GPU_DEVICE_ONLY_MLP_ENV, None, || {
        assert!(!gpu_device_only_mlp_enabled());
    });
    with_env(GPU_DEVICE_ONLY_MLP_ENV, Some("0"), || {
        assert!(!gpu_device_only_mlp_enabled());
    });
}

/// Full live acceptance: baseline vs candidate, physical counters, wall, causal
/// mutation. Does not promote; prints a promote/negative verdict.
#[test]
fn device_only_mlp_live_acceptance() {
    let Some(ctx) = require_metal() else {
        return;
    };

    // Expert-wave and other sealed-negative paths stay off for this lane.
    std::env::remove_var("HAWKING_GLM_GPU_EXPERT_WAVE");
    std::env::remove_var("HAWKING_GLM_GPU_EXPERT_WAVE_CONCURRENT");
    std::env::remove_var(GPU_DEVICE_ONLY_MLP_POISON_ENV);

    let tokens = prompt();
    assert!(!tokens.is_empty());

    // ── Warm immutable resources (weights/pipelines) with flag off ─────────
    std::env::remove_var(GPU_DEVICE_ONLY_MLP_ENV);
    let baseline_model = open_resident(MetalContext::new().expect("baseline ctx"));
    let candidate_model = open_resident(ctx);
    assert!(baseline_model.resident_state_enabled());
    assert!(candidate_model.resident_state_enabled());

    // Warm both sessions once (discard timings).
    let _ = baseline_model
        .forward_resident_counted(&tokens)
        .expect("baseline warm");
    with_env(GPU_DEVICE_ONLY_MLP_ENV, Some("1"), || {
        reset_device_only_mlp_probe();
        let _ = candidate_model
            .forward_resident_counted(&tokens)
            .expect("candidate warm");
        assert!(
            device_only_mlp_hits() > 0,
            "warm candidate must enter device-only MLP (hits={})",
            device_only_mlp_hits()
        );
    });

    // ── Physical ledger run: one measured token each, fresh session state ───
    // Baseline
    std::env::remove_var(GPU_DEVICE_ONLY_MLP_ENV);
    reset_device_only_mlp_probe();
    cost_ledger::set_enabled(true);
    let _ = cost_ledger::end_token();
    assert!(cost_ledger::begin_token());
    let t0 = Instant::now();
    let (base_logits, base_trace, base_waits) = baseline_model
        .forward_resident_counted(&tokens)
        .expect("baseline measured");
    let base_wall_us = t0.elapsed().as_micros() as u64;
    let base_report = cost_ledger::end_token().expect("baseline ledger");
    cost_ledger::set_enabled(false);
    let base_hits = device_only_mlp_hits();
    assert_eq!(base_hits, 0, "baseline must not hit device-only MLP");

    // Candidate
    with_env(GPU_DEVICE_ONLY_MLP_ENV, Some("1"), || {
        reset_device_only_mlp_probe();
        cost_ledger::set_enabled(true);
        let _ = cost_ledger::end_token();
        assert!(cost_ledger::begin_token());
        let t0 = Instant::now();
        let (cand_logits, cand_trace, cand_waits) = candidate_model
            .forward_resident_counted(&tokens)
            .expect("candidate measured");
        let cand_wall_us = t0.elapsed().as_micros() as u64;
        let cand_report = cost_ledger::end_token().expect("candidate ledger");
        cost_ledger::set_enabled(false);

        let cand_hits = device_only_mlp_hits();
        let cand_fallbacks = device_only_mlp_fallbacks();
        assert!(
            cand_hits > 0,
            "candidate must execute device-only MLP (hits={cand_hits} fallbacks={cand_fallbacks})"
        );
        assert_eq!(
            cand_fallbacks, 0,
            "candidate must not fall back to host SiLU on this fixture"
        );

        // Transfer proof: gate/up download and activation upload are zero.
        assert_eq!(
            cand_report.counters.mlp_gate_up_download_bytes, 0,
            "candidate gate/up download bytes must be zero"
        );
        assert_eq!(
            cand_report.counters.mlp_gate_up_download_transfers, 0,
            "candidate gate/up download transfers must be zero"
        );
        assert_eq!(
            cand_report.counters.mlp_activation_upload_bytes, 0,
            "candidate activation upload bytes must be zero"
        );
        assert_eq!(
            cand_report.counters.mlp_activation_upload_transfers, 0,
            "candidate activation upload transfers must be zero"
        );
        assert!(
            cand_report.counters.device_only_mlp_hits > 0,
            "ledger device_only_mlp_hits must be positive"
        );
        // Baseline must have recorded host intermediate materialization.
        assert!(
            base_report.counters.mlp_gate_up_download_bytes > 0,
            "baseline must record gate/up download (got {})",
            base_report.counters.mlp_gate_up_download_bytes
        );
        assert!(
            base_report.counters.mlp_activation_upload_bytes > 0,
            "baseline must record activation upload (got {})",
            base_report.counters.mlp_activation_upload_bytes
        );

        // Waits: no regression.
        assert!(
            cand_waits <= base_waits,
            "wait regression: candidate {cand_waits} > baseline {base_waits}"
        );

        // Command buffers must differ — identical means candidate never ran
        // (failure 2: invalid topology / cached no-op).
        let base_cbs = base_report.counters.command_buffers_submitted;
        let cand_cbs = cand_report.counters.command_buffers_submitted;
        assert_ne!(
            cand_cbs, base_cbs,
            "command buffer counts identical ({base_cbs}) — candidate may not have executed"
        );

        // Discrete decisions exact.
        assert_eq!(
            cand_trace.expert_choices, base_trace.expert_choices,
            "expert choices must match"
        );
        assert_eq!(
            cand_trace.final_topk, base_trace.final_topk,
            "DSA top-k must match"
        );
        assert_eq!(
            top1(&cand_logits),
            top1(&base_logits),
            "greedy argmax must match"
        );

        // V2.1 continuous on logits vs FP64-widened baseline-as-proxy is not
        // available without a full f64 forward; score candidate vs baseline
        // residuals through continuous_only bounds using baseline as the
        // host arm and an f64 lift of baseline as authority (host path is the
        // production oracle for this fixture; device SiLU is already validated
        // against f64 above). Continuous pass must be true.
        let ref64: Vec<f64> = base_logits.iter().map(|&v| v as f64).collect();
        let bounds = Bounds::logits();
        let paired = score_pair(&base_logits, &cand_logits, &ref64, &bounds);
        eprintln!("V2.1 logits {}", format_score_line(&paired.host));
        eprintln!("V2.1 logits {}", format_score_line(&paired.device));
        eprintln!(
            "V2.1 continuous pass={} discrete host_pass={} device_pass={} pair_pass={}",
            paired.device.pass, paired.host.pass, paired.device.pass, paired.pass
        );
        assert!(
            paired.pass,
            "Numeric Parity V2.1 must pass continuous+discrete; continuous pass={}",
            paired.device.pass
        );
        assert!(
            paired.device.pass,
            "continuous pass=false is a hard failure (do not assert around it)"
        );

        eprintln!(
            "ledger baseline: waits={base_waits} cbs={base_cbs} \
             gate_up_dl={} act_ul={} hits={} wall_us={base_wall_us}",
            base_report.counters.mlp_gate_up_download_bytes,
            base_report.counters.mlp_activation_upload_bytes,
            base_report.counters.device_only_mlp_hits,
        );
        eprintln!(
            "ledger candidate: waits={cand_waits} cbs={cand_cbs} \
             gate_up_dl={} act_ul={} hits={} wall_us={cand_wall_us}",
            cand_report.counters.mlp_gate_up_download_bytes,
            cand_report.counters.mlp_activation_upload_bytes,
            cand_report.counters.device_only_mlp_hits,
        );

        // Stash for timing phase via thread-local-like env print; recompute timing below.
        let _ = (base_report, cand_report);
    });

    // ── Wall clock: warm, interleaved, reset-by-fresh-forward position ─────
    // Each sample uses the same prompt on a model that already holds warm
    // weights. We re-open sessions? forward_resident_counted appends; for a
    // fair single-token compare we use the same multi-token prompt each time
    // on separate models that we reset by constructing new sessions.
    //
    // GravityGlmGpu does not expose session reset publicly; measure full
    // prompt forwards as the fixture token path (same work each sample).
    const WARMUPS: usize = 8;
    const SAMPLES: usize = 40;

    let mut base_samples = Vec::with_capacity(SAMPLES);
    let mut cand_samples = Vec::with_capacity(SAMPLES);
    let mut base_wait_samples = Vec::with_capacity(SAMPLES);
    let mut cand_wait_samples = Vec::with_capacity(SAMPLES);

    // Fresh models for timing so session growth does not pollute.
    let base_timing = open_resident(MetalContext::new().expect("base timing ctx"));
    let cand_timing = open_resident(MetalContext::new().expect("cand timing ctx"));

    std::env::remove_var(GPU_DEVICE_ONLY_MLP_ENV);
    for _ in 0..WARMUPS {
        let _ = base_timing.forward_resident_counted(&tokens).expect("bw");
    }
    with_env(GPU_DEVICE_ONLY_MLP_ENV, Some("1"), || {
        for _ in 0..WARMUPS {
            let _ = cand_timing.forward_resident_counted(&tokens).expect("cw");
        }
    });

    // Interleaved measured samples. Use single-token prompts after warm so
    // each sample is one decode step on a growing session — still same work
    // shape for both modes when we keep separate models with identical
    // warmup counts.
    //
    // Better: reopen each sample. Expensive but correct for tiny fixture.
    for i in 0..SAMPLES {
        // Baseline sample
        std::env::remove_var(GPU_DEVICE_ONLY_MLP_ENV);
        let m = open_resident(MetalContext::new().expect("b sample ctx"));
        // one warm forward discarded
        let _ = m.forward_resident_counted(&tokens).expect("b pre");
        let t0 = Instant::now();
        let (_, _, w) = m.forward_resident_counted(&tokens).expect("b meas");
        base_samples.push(t0.elapsed().as_secs_f64() * 1e3);
        base_wait_samples.push(w);

        // Candidate sample
        with_env(GPU_DEVICE_ONLY_MLP_ENV, Some("1"), || {
            reset_device_only_mlp_probe();
            let m = open_resident(MetalContext::new().expect("c sample ctx"));
            let _ = m.forward_resident_counted(&tokens).expect("c pre");
            let t0 = Instant::now();
            let (_, _, w) = m.forward_resident_counted(&tokens).expect("c meas");
            assert!(device_only_mlp_hits() > 0, "sample {i}: no device-only hits");
            cand_samples.push(t0.elapsed().as_secs_f64() * 1e3);
            cand_wait_samples.push(w);
        });
    }

    let mut base_sorted = base_samples.clone();
    let mut cand_sorted = cand_samples.clone();
    base_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cand_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let base_p50 = percentile_nearest(&base_sorted, 50.0);
    let base_p95 = percentile_nearest(&base_sorted, 95.0);
    let cand_p50 = percentile_nearest(&cand_sorted, 50.0);
    let cand_p95 = percentile_nearest(&cand_sorted, 95.0);

    let base_waits_mean =
        base_wait_samples.iter().map(|&w| w as f64).sum::<f64>() / base_wait_samples.len() as f64;
    let cand_waits_mean =
        cand_wait_samples.iter().map(|&w| w as f64).sum::<f64>() / cand_wait_samples.len() as f64;

    eprintln!("=== device-only MLP live acceptance ===");
    eprintln!(
        "waits/token  baseline_mean={base_waits_mean:.1} candidate_mean={cand_waits_mean:.1}"
    );
    eprintln!("p50 ms/token baseline={base_p50:.4} candidate={cand_p50:.4}");
    eprintln!("p95 ms/token baseline={base_p95:.4} candidate={cand_p95:.4}");
    eprintln!(
        "raw base samples (first 5): {:?}",
        &base_samples[..5.min(base_samples.len())]
    );
    eprintln!(
        "raw cand samples (first 5): {:?}",
        &cand_samples[..5.min(cand_samples.len())]
    );

    // ── Causal mutation: flag off must not record hits ─────────────────────
    std::env::remove_var(GPU_DEVICE_ONLY_MLP_ENV);
    reset_device_only_mlp_probe();
    let m = open_resident(MetalContext::new().expect("mutation ctx"));
    let _ = m.forward_resident_counted(&tokens).expect("mutation off");
    assert_eq!(
        device_only_mlp_hits(),
        0,
        "flag off must not count device-only hits"
    );

    // Poison SiLU: with flag on + poison, continuous parity must fail.
    let mut poison_failed = false;
    with_env(GPU_DEVICE_ONLY_MLP_ENV, Some("1"), || {
        with_env(GPU_DEVICE_ONLY_MLP_POISON_ENV, Some("1"), || {
            reset_device_only_mlp_probe();
            let base = open_resident(MetalContext::new().expect("poison base"));
            std::env::remove_var(GPU_DEVICE_ONLY_MLP_ENV);
            std::env::remove_var(GPU_DEVICE_ONLY_MLP_POISON_ENV);
            let (b_logits, _, _) = base.forward_resident_counted(&tokens).expect("pb");
            std::env::set_var(GPU_DEVICE_ONLY_MLP_ENV, "1");
            std::env::set_var(GPU_DEVICE_ONLY_MLP_POISON_ENV, "1");
            let poison = open_resident(MetalContext::new().expect("poison cand"));
            let (p_logits, _, _) = poison.forward_resident_counted(&tokens).expect("pp");
            assert!(
                device_only_mlp_hits() > 0,
                "poison path must still hit device-only MLP"
            );
            let ref64: Vec<f64> = b_logits.iter().map(|&v| v as f64).collect();
            let paired = score_pair(&b_logits, &p_logits, &ref64, &Bounds::logits());
            if !paired.pass || top1(&p_logits) != top1(&b_logits) {
                poison_failed = true;
            }
            eprintln!(
                "poison causal: pair_pass={} base_tok={} poison_tok={}",
                paired.pass,
                top1(&b_logits),
                top1(&p_logits)
            );
        });
    });
    std::env::remove_var(GPU_DEVICE_ONLY_MLP_POISON_ENV);
    std::env::remove_var(GPU_DEVICE_ONLY_MLP_ENV);
    assert!(
        poison_failed,
        "corrupting device SiLU must make parity/token identity fail (causal mutation)"
    );

    // Final wall-clock verdict — improve or honest negative.
    let p50_win = cand_p50 < base_p50;
    let p95_win = cand_p95 < base_p95;
    if p50_win && p95_win {
        eprintln!(
            "VERDICT: promote candidate (p50 {base_p50:.4}->{cand_p50:.4}, \
             p95 {base_p95:.4}->{cand_p95:.4}); flag still default-off until \
             an authorized default flip"
        );
    } else {
        eprintln!(
            "VERDICT: NEGATIVE — leave flag off. p50 {base_p50:.4}->{cand_p50:.4} win={p50_win}; \
             p95 {base_p95:.4}->{cand_p95:.4} win={p95_win}. \
             Correctness and transfer/wait/CB gates held; wall clock did not improve both tails."
        );
        // Do not fail the test on a truthful negative wall result — the lane
        // accepts an honest negative. Hard gates above already asserted.
    }
}

/// Mutation: if the suite required hits while the flag is off, it must fail.
#[test]
fn device_only_mlp_causal_flag_off_has_zero_hits() {
    let Some(ctx) = require_metal() else {
        return;
    };
    std::env::remove_var(GPU_DEVICE_ONLY_MLP_ENV);
    std::env::remove_var("HAWKING_GLM_GPU_EXPERT_WAVE");
    reset_device_only_mlp_probe();
    let model = open_resident(ctx);
    let tokens = prompt();
    let _ = model.forward_resident_counted(&tokens).expect("forward");
    assert_eq!(
        device_only_mlp_hits(),
        0,
        "with flag off, any test that requires hits>0 must fail — hits stayed zero"
    );
}
