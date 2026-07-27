//! Complete-token live compact MLA parity on a direct-u8 PQ fixture.
//!
//! The ordinary checked-in fixture intentionally uses the historical R0
//! codec, which the absorbed K/V kernels reject. The controller supplies a
//! bounded R4 fixture through `HAWKING_GLM_COMPACT_FIXTURE_DIR` when explicitly
//! validating this default-off candidate.

#![cfg(target_os = "macos")]

use std::path::PathBuf;

use hawking_core::gravity_glm::gpu::GravityGlmGpu;
use hawking_core::gravity_glm::{
    GPU_COMPACT_MLA_ENV, GPU_LM_HEAD_ENV, GPU_LM_HEAD_FULL_LOGITS_ENV,
};
use hawking_core::metal::MetalContext;
use hawking_core::numeric_parity::{score_pair, Bounds};

fn prompts(base: &[u32]) -> Vec<Vec<u32>> {
    let mut prompts = vec![base.to_vec(), vec![7], vec![9, 7]];
    if base.len() > 1 {
        let mut reversed = base.to_vec();
        reversed.reverse();
        prompts.push(reversed);
    }
    prompts
}

#[test]
fn compact_mla_complete_tokens_match_expanded_v21_and_exact_decisions() {
    let Some(dir) = std::env::var_os("HAWKING_GLM_COMPACT_FIXTURE_DIR").map(PathBuf::from) else {
        eprintln!("skip: set HAWKING_GLM_COMPACT_FIXTURE_DIR to a bounded direct-u8 PQ fixture");
        return;
    };
    let Ok(expanded_ctx) = MetalContext::new() else {
        eprintln!("skip: no Metal device");
        return;
    };
    let compact_ctx = MetalContext::new().expect("second Metal context");

    let prior_compact = std::env::var_os(GPU_COMPACT_MLA_ENV);
    let prior_head = std::env::var_os(GPU_LM_HEAD_ENV);
    let prior_full_logits = std::env::var_os(GPU_LM_HEAD_FULL_LOGITS_ENV);
    std::env::remove_var(GPU_LM_HEAD_ENV);
    std::env::remove_var(GPU_LM_HEAD_FULL_LOGITS_ENV);

    std::env::remove_var(GPU_COMPACT_MLA_ENV);
    let expanded = GravityGlmGpu::open_dir_with_budget_resident(
        expanded_ctx,
        &dir,
        true,
        512 * 1024 * 1024,
        true,
    )
    .expect("expanded resident fixture");
    std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
    let compact = GravityGlmGpu::open_dir_with_budget_resident(
        compact_ctx,
        &dir,
        true,
        512 * 1024 * 1024,
        true,
    )
    .expect("compact resident fixture");
    std::env::remove_var(GPU_COMPACT_MLA_ENV);

    #[derive(serde::Deserialize)]
    struct Reference {
        tokens: Vec<u32>,
    }
    let reference: Reference = serde_json::from_slice(
        &std::fs::read(dir.join("ref_glm.json")).expect("compact ref_glm.json"),
    )
    .expect("parse compact reference");
    #[derive(serde::Deserialize)]
    struct Authority {
        tokens: Vec<u32>,
        logits: Vec<f64>,
    }
    let authorities: Vec<Authority> = serde_json::from_slice(
        &std::fs::read(dir.join("ref_logits_f64.json"))
            .expect("explicit FP64 complete-token authorities"),
    )
    .expect("parse FP64 complete-token authorities");

    for (case, prompt) in prompts(&reference.tokens).into_iter().enumerate() {
        let (expanded_logits, expanded_trace) =
            expanded.forward(&prompt).expect("expanded forward");
        let (compact_logits, compact_trace) = compact.forward(&prompt).expect("compact forward");
        let authority = &authorities
            .iter()
            .find(|authority| authority.tokens == prompt)
            .unwrap_or_else(|| panic!("missing FP64 authority for prompt {prompt:?}"))
            .logits;
        let pair = score_pair(
            &expanded_logits,
            &compact_logits,
            authority,
            &Bounds::logits(),
        );
        eprintln!(
            "compact MLA case {case}: host rel_l2={:.3e} meaningful={:.3e}; \
             device rel_l2={:.3e} meaningful={:.3e}; greedy={} top5={}",
            pair.host.continuous.relative_l2,
            pair.host.continuous.max_meaningful_rel,
            pair.device.continuous.relative_l2,
            pair.device.continuous.max_meaningful_rel,
            pair.device.discrete.greedy_match,
            pair.device.discrete.top_k_exact_match
        );
        assert!(
            pair.pass,
            "case {case} prompt {prompt:?}: compact complete-token V2.1 {pair:#?}"
        );
        assert_eq!(
            compact_trace.final_topk, expanded_trace.final_topk,
            "case {case}: exact DSA selection"
        );
        assert_eq!(
            compact_trace.expert_choices, expanded_trace.expert_choices,
            "case {case}: exact expert choices"
        );
    }

    match prior_compact {
        Some(value) => std::env::set_var(GPU_COMPACT_MLA_ENV, value),
        None => std::env::remove_var(GPU_COMPACT_MLA_ENV),
    }
    match prior_head {
        Some(value) => std::env::set_var(GPU_LM_HEAD_ENV, value),
        None => std::env::remove_var(GPU_LM_HEAD_ENV),
    }
    match prior_full_logits {
        Some(value) => std::env::set_var(GPU_LM_HEAD_FULL_LOGITS_ENV, value),
        None => std::env::remove_var(GPU_LM_HEAD_FULL_LOGITS_ENV),
    }
}
