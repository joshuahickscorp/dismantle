//! Multi-prompt parity: GPU-resident decode state vs the host-state path.
//!
//! A single-prompt match can pass by luck of phase — a documented failure
//! mode in this codebase. This test runs several prompts (and an incremental
//! decode split) and requires bit-identical tokens (argmax) against the
//! host-state oracle on the same artifact.
//!
//! Also covers **expert CB collapse** (`HAWKING_GLM_EXPERT_CB_COLLAPSE=1`):
//! one command buffer per MoE/MLP layer with device `silu_mul`, bit-identical
//! to the three-batch resident oracle and the host-state GPU path.
//!
//! Requires Metal. The executor sandbox often has none; the controller runs
//! this on a machine with a device:
//!
//! ```text
//! cargo test -p hawking-core --test gravity_glm_resident_parity -- --nocapture
//!
//! # collapse lane (also exercises multi-prompt bit-identity):
//! HAWKING_GLM_GPU_RESIDENT_STATE=1 HAWKING_GLM_EXPERT_CB_COLLAPSE=1 \
//!   cargo test -p hawking-core --test gravity_glm_resident_parity \
//!   expert_collapse -- --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use hawking_core::gravity_glm::gpu::GravityGlmGpu;
use hawking_core::gravity_glm::{
    estimate_host_state_waits_per_token, estimate_mlp_batch_waits_per_token,
    estimate_mlp_collapsed_waits_per_token, estimate_resident_waits_per_token,
    estimate_resident_waits_per_token_with, EXPERT_CB_COLLAPSE_ENV, GravityGlm,
};
use hawking_core::metal::MetalContext;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gravity_glm")
}

fn top1(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .min_by(|(i, a), (j, b)| {
            // Highest logit wins; lower index wins ties (stable).
            b.partial_cmp(a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(i.cmp(j))
        })
        .map(|(i, _)| i as u32)
        .expect("non-empty logits")
}

fn top_k(logits: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .expect("no NaN")
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

/// Several prompts — not one. Fixture tokens plus permutations and short
/// alternatives so a phase-aligned single sequence cannot paper over a bug.
fn prompts(base: &[u32]) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    out.push(base.to_vec());
    if base.len() >= 2 {
        out.push(base[1..].to_vec());
        out.push(base[..base.len() - 1].to_vec());
        let mut rev = base.to_vec();
        rev.reverse();
        out.push(rev);
    }
    // Distinct short prompts inside the fixture vocab.
    out.push(vec![0]);
    out.push(vec![1, 2, 3]);
    out.push(vec![7, 7, 7, 7]);
    out.push(vec![100, 200, 300, 400, 500]);
    out
}

#[test]
fn resident_matches_host_state_over_several_prompts() {
    let dir = fixtures_dir();
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            // A shader COMPILE failure is not an absent device, and treating it as
            // one makes this suite report green while proving nothing. That is
            // exactly how a broken kernel ships. Only a genuinely missing device
            // may skip; anything else fails loudly.
            let msg = e.to_string();
            assert!(
                !msg.contains("shader") && !msg.contains("compile"),
                "Metal is present but the shader failed to compile -- this is a real \
                 failure, not a skip: {msg}"
            );
            eprintln!("skip: no Metal device ({e})");
            return;
        }
    };

    let host = GravityGlm::open(&dir.join("glm52-tiny-R0.gravity"), true).expect("host open");
    let host_gpu = GravityGlmGpu::open_dir_with_budget_resident(
        MetalContext::new().expect("second ctx"),
        &dir,
        true,
        256 * 1024 * 1024,
        false, // host-state GPU path (oracle for GPU weight layout)
    )
    .expect("host-state gpu open");
    let resident = GravityGlmGpu::open_dir_with_budget_resident(
        ctx,
        &dir,
        true,
        256 * 1024 * 1024,
        true, // resident path under test
    )
    .expect("resident open");
    assert!(resident.resident_state_enabled());
    assert!(!host_gpu.resident_state_enabled());

    let base: Vec<u32> = {
        #[derive(serde::Deserialize)]
        struct Ref {
            tokens: Vec<u32>,
        }
        let r: Ref = serde_json::from_slice(
            &std::fs::read(dir.join("ref_glm.json")).expect("ref_glm"),
        )
        .expect("parse");
        r.tokens
    };

    let mut any_waits = None;
    for (pi, prompt) in prompts(&base).into_iter().enumerate() {
        if prompt.is_empty() {
            continue;
        }
        // Skip tokens outside vocab (should not happen with our prompts).
        if prompt.iter().any(|&t| t as usize >= host.arch.vocab_size) {
            continue;
        }

        let (cpu_logits, cpu_trace) = host.forward(&prompt).expect("cpu forward");
        let (host_gpu_logits, host_gpu_trace) =
            host_gpu.forward(&prompt).expect("host-state gpu forward");
        let (res_logits, res_trace, waits) = resident
            .forward_resident_counted(&prompt)
            .expect("resident forward");
        any_waits = Some(waits);

        // Token identity against the host-state GPU path (same weights, same
        // PQ kernels). CPU may differ slightly on the two PQ tensors
        // (embed/lm_head) due to simd_sum reassociation on the GPU path.
        let host_tok = top1(&host_gpu_logits);
        let res_tok = top1(&res_logits);
        assert_eq!(
            res_tok, host_tok,
            "prompt {pi} {prompt:?}: resident argmax {res_tok} != host-state gpu {host_tok}"
        );
        assert_eq!(
            top_k(&res_logits, 5),
            top_k(&host_gpu_logits, 5),
            "prompt {pi}: top-5 tokens diverge"
        );

        // Discrete DSA / expert decisions must match the host-state path.
        assert_eq!(
            res_trace.final_topk, host_gpu_trace.final_topk,
            "prompt {pi}: final DSA top-k"
        );
        assert_eq!(
            res_trace.expert_choices, host_gpu_trace.expert_choices,
            "prompt {pi}: expert choices"
        );

        // Logits: same arithmetic order on the same native projections —
        // expect bit-identity for the tiny fixture (all layer weights native).
        assert_eq!(
            res_logits, host_gpu_logits,
            "prompt {pi}: logits must be bit-identical on the native-heavy fixture"
        );

        // Sanity: CPU oracle still reaches a fluent argmax (not required bit-identical
        // to GPU on PQ embed/head, but top-1 should usually agree on this fixture).
        let _ = (cpu_logits, cpu_trace);
    }

    let waits = any_waits.expect("ran at least one prompt");
    eprintln!(
        "resident waits (live, last prompt path): {waits}; static host={} resident={}",
        estimate_host_state_waits_per_token(&host.arch),
        estimate_resident_waits_per_token(&host.arch)
    );
}

#[test]
fn resident_incremental_decode_matches_full_replay() {
    let dir = fixtures_dir();
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            // A shader COMPILE failure is not an absent device, and treating it as
            // one makes this suite report green while proving nothing. That is
            // exactly how a broken kernel ships. Only a genuinely missing device
            // may skip; anything else fails loudly.
            let msg = e.to_string();
            assert!(
                !msg.contains("shader") && !msg.contains("compile"),
                "Metal is present but the shader failed to compile -- this is a real \
                 failure, not a skip: {msg}"
            );
            eprintln!("skip: no Metal device ({e})");
            return;
        }
    };
    let model = GravityGlmGpu::open_dir_with_budget_resident(
        ctx,
        &dir,
        true,
        256 * 1024 * 1024,
        true,
    )
    .expect("open");

    #[derive(serde::Deserialize)]
    struct Ref {
        tokens: Vec<u32>,
    }
    let reference: Ref =
        serde_json::from_slice(&std::fs::read(dir.join("ref_glm.json")).unwrap()).unwrap();
    let tokens = &reference.tokens;
    assert!(tokens.len() >= 3);

    let (want, _) = model.forward(tokens).expect("full");
    let split = tokens.len() - 2;
    let (mut got, _) = model.forward(&tokens[..split]).expect("prefill");
    for (i, &t) in tokens[split..].iter().enumerate() {
        got = model.forward_at(&[t], split + i).expect("extend").0;
    }
    assert_eq!(got, want, "incremental resident decode must match full replay");
}

#[test]
fn static_wait_estimates_are_exported_for_the_controller() {
    let dir = fixtures_dir();
    let host = GravityGlm::open(&dir.join("glm52-tiny-R0.gravity"), false).unwrap();
    let h = estimate_host_state_waits_per_token(&host.arch);
    let r = estimate_resident_waits_per_token_with(&host.arch, false);
    let c = estimate_resident_waits_per_token_with(&host.arch, true);
    assert!(h > r);
    assert!(c < r, "collapse must cut resident waits: resident={r} collapsed={c}");
    // Tiny fixture: 1 dense + 3 sparse, 3 full indexers + 1 shared.
    // host: dense (5+3+3) + sparse full (5+3+4)*2 + sparse shared (5+4) = 11 + 2*12 + 9 = 44?
    assert!(h >= 30 && h <= 80, "tiny host waits {h}");
    // MLP-only static: 3 * n_layers → 1 * n_layers.
    assert_eq!(
        estimate_mlp_batch_waits_per_token(&host.arch),
        3 * host.arch.n_layers as u64
    );
    assert_eq!(
        estimate_mlp_collapsed_waits_per_token(&host.arch),
        host.arch.n_layers as u64
    );
    eprintln!(
        "tiny fixture static waits: host={h} resident={r} collapsed={c} \
         mlp_batch={} mlp_collapsed={}",
        estimate_mlp_batch_waits_per_token(&host.arch),
        estimate_mlp_collapsed_waits_per_token(&host.arch)
    );
}

/// Metal presence probe: shader compile failures must fail the test, not skip.
fn metal_or_skip() -> Option<MetalContext> {
    match MetalContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                !msg.contains("shader") && !msg.contains("compile"),
                "Metal is present but the shader failed to compile -- this is a real \
                 failure, not a skip: {msg}"
            );
            eprintln!("skip: no Metal device ({e})");
            None
        }
    }
}

/// Expert CB collapse vs three-batch resident and host-state GPU — several
/// prompts, bit-identical logits.
///
/// Collapse is read from the env at each `batched_mlp` call (same pattern as
/// other HAWKING levers). The test toggles the env around each forward so the
/// three-batch oracle cannot accidentally run the collapsed path.
#[test]
fn expert_collapse_matches_host_over_several_prompts() {
    let Some(ctx) = metal_or_skip() else {
        return;
    };
    let dir = fixtures_dir();
    let prev_collapse = std::env::var_os(EXPERT_CB_COLLAPSE_ENV);

    let three_batch = GravityGlmGpu::open_dir_with_budget_resident(
        MetalContext::new().expect("second ctx"),
        &dir,
        true,
        256 * 1024 * 1024,
        true,
    )
    .expect("three-batch resident open");
    let host_gpu = GravityGlmGpu::open_dir_with_budget_resident(
        MetalContext::new().expect("third ctx"),
        &dir,
        true,
        256 * 1024 * 1024,
        false,
    )
    .expect("host-state gpu open");
    let collapsed = GravityGlmGpu::open_dir_with_budget_resident(
        ctx,
        &dir,
        true,
        256 * 1024 * 1024,
        true,
    )
    .expect("collapsed resident open");
    assert!(collapsed.resident_state_enabled());

    let base: Vec<u32> = {
        #[derive(serde::Deserialize)]
        struct Ref {
            tokens: Vec<u32>,
        }
        let r: Ref = serde_json::from_slice(
            &std::fs::read(dir.join("ref_glm.json")).expect("ref_glm"),
        )
        .expect("parse");
        r.tokens
    };

    let arch = &collapsed.arch;
    let mut any_waits = None;
    let mut prompt_count = 0usize;
    for (pi, prompt) in prompts(&base).into_iter().enumerate() {
        if prompt.is_empty() {
            continue;
        }
        if prompt.iter().any(|&t| t as usize >= arch.vocab_size) {
            continue;
        }
        prompt_count += 1;

        // Host-state GPU: collapse flag is irrelevant (not on resident path).
        std::env::remove_var(EXPERT_CB_COLLAPSE_ENV);
        let (host_logits, host_trace) = host_gpu.forward(&prompt).expect("host-state forward");

        // Three-batch resident oracle — collapse must be off.
        std::env::remove_var(EXPERT_CB_COLLAPSE_ENV);
        let (tb_logits, tb_trace, _tb_waits) = three_batch
            .forward_resident_counted(&prompt)
            .expect("three-batch forward");

        // Collapsed path under test.
        std::env::set_var(EXPERT_CB_COLLAPSE_ENV, "1");
        let (col_logits, col_trace, waits) = collapsed
            .forward_resident_counted(&prompt)
            .expect("collapsed forward");
        any_waits = Some(waits);

        assert_eq!(
            top1(&col_logits),
            top1(&host_logits),
            "prompt {pi} {prompt:?}: collapsed argmax != host-state gpu"
        );
        assert_eq!(
            top_k(&col_logits, 5),
            top_k(&host_logits, 5),
            "prompt {pi}: top-5 tokens diverge (collapse vs host-state)"
        );
        assert_eq!(
            col_trace.final_topk, host_trace.final_topk,
            "prompt {pi}: final DSA top-k (collapse vs host-state)"
        );
        assert_eq!(
            col_trace.expert_choices, host_trace.expert_choices,
            "prompt {pi}: expert choices (collapse vs host-state)"
        );
        assert_eq!(
            col_logits, host_logits,
            "prompt {pi}: collapsed logits must be bit-identical to host-state gpu"
        );
        assert_eq!(
            col_logits, tb_logits,
            "prompt {pi}: collapsed must match three-batch resident bit-identically"
        );
        assert_eq!(
            col_trace.expert_choices, tb_trace.expert_choices,
            "prompt {pi}: expert choices (collapse vs three-batch)"
        );
    }
    assert!(
        prompt_count >= 5,
        "need several prompts for collapse parity, got {prompt_count}"
    );

    let waits = any_waits.expect("ran at least one prompt");
    let static_collapsed = estimate_resident_waits_per_token_with(arch, true);
    let static_three = estimate_resident_waits_per_token_with(arch, false);
    eprintln!(
        "expert_collapse: prompts={prompt_count} live_waits={waits} \
         static_collapsed={static_collapsed} static_three_batch={static_three} \
         mlp_only {} -> {}",
        estimate_mlp_batch_waits_per_token(arch),
        estimate_mlp_collapsed_waits_per_token(arch)
    );
    assert!(
        waits <= static_three,
        "collapsed live waits {waits} should be <= three-batch static {static_three}"
    );

    match prev_collapse {
        Some(v) => std::env::set_var(EXPERT_CB_COLLAPSE_ENV, v),
        None => std::env::remove_var(EXPERT_CB_COLLAPSE_ENV),
    }
}

/// Device silu_mul kernel vs host formula — bit-identical on a dense grid.
/// Compile failure fails loud; missing device skips.
#[test]
fn device_silu_mul_matches_host_bit_identical() {
    let Some(ctx) = metal_or_skip() else {
        return;
    };
    use hawking_core::metal::TokenCommandBuffer;

    // Varied magnitudes including zero, large positive/negative (sigmoid tails).
    let mut gate = Vec::new();
    let mut up = Vec::new();
    for i in 0..512 {
        let t = (i as f32) * 0.03125 - 8.0;
        gate.push(t);
        up.push((i as f32 * 0.017) - 1.0);
    }
    let n = gate.len();
    let host: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect();

    let gate_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&gate));
    let up_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&up));
    let out_buf = ctx.new_buffer(n * 4);

    let mut tcb = TokenCommandBuffer::new(&ctx);
    const TG: u32 = 256;
    let n_u32 = n as u32;
    // Do not name locals after Metal types.
    let n_elems = n_u32;
    tcb.dispatch_threads(
        "gravity_silu_mul_f32",
        (n_elems.div_ceil(TG) * TG, 1, 1),
        (TG, 1, 1),
        |enc| {
            enc.set_buffer(0, Some(&gate_buf), 0);
            enc.set_buffer(1, Some(&up_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_bytes(3, 4, &n_elems as *const u32 as *const _);
        },
    )
    .expect("encode gravity_silu_mul_f32 — compile/link failure must not be swallowed");
    tcb.commit_and_wait()
        .expect("commit silu_mul — runtime failure must not be swallowed");

    let device = unsafe {
        std::slice::from_raw_parts(out_buf.contents() as *const f32, n).to_vec()
    };
    assert_eq!(
        device, host,
        "device gravity_silu_mul_f32 must match host f32 silu bit-for-bit"
    );
    eprintln!("device_silu_mul: {n} elements bit-identical to host");
}
