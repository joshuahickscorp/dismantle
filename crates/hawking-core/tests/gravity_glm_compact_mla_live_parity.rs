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
    GPU_COMPACT_MLA_ENV, GPU_DEVICE_DSA_ENV, GPU_LM_HEAD_ENV, GPU_LM_HEAD_FULL_LOGITS_ENV,
};
use hawking_core::metal::MetalContext;
use hawking_core::numeric_parity::{score_pair, Bounds};

fn invalid_compact_geometry_fixture(source: &std::path::Path) -> tempfile::TempDir {
    let invalid = tempfile::tempdir().expect("temporary invalid compact fixture");
    for entry in std::fs::read_dir(source).expect("read compact fixture") {
        let entry = entry.expect("fixture directory entry");
        if entry.file_type().expect("fixture entry type").is_file() {
            std::fs::copy(entry.path(), invalid.path().join(entry.file_name()))
                .expect("copy compact fixture file");
        }
    }

    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(invalid.path().join("model.gravity.index.json"))
            .expect("copied gravity index"),
    )
    .expect("parse copied gravity index");
    let kv_name = "model.layers.0.self_attn.kv_b_proj.weight";
    let shard_name = index["weight_map"][kv_name]
        .as_str()
        .expect("KV owning shard");
    let shard_path = invalid.path().join(shard_name);
    let mut shard = std::fs::read(&shard_path).expect("read copied compact shard");
    let header_len = u64::from_le_bytes(shard[12..20].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(&shard[20..20 + header_len]).expect("parse shard header");
    let descriptor = header["tensors"]
        .as_array()
        .expect("shard tensors")
        .iter()
        .find(|tensor| tensor["name"].as_str() == Some(kv_name))
        .expect("KV descriptor");
    let payload =
        20 + header_len + descriptor["offset"].as_u64().expect("KV payload offset") as usize;

    // Keep the PQ header structurally self-consistent (D=S*sub and
    // cols=nchunk*D), but make it unsupported by the exact D32 compact
    // kernels. The descriptor SHA is intentionally left stale: with
    // verify_hash=true, seeing the geometry error proves preflight ran before
    // the ordinary complete-payload verification/load.
    shard[payload + 8..payload + 10].copy_from_slice(&16u16.to_le_bytes()); // D
    shard[payload + 12..payload + 14].copy_from_slice(&16u16.to_le_bytes()); // sub
    shard[payload + 24..payload + 28].copy_from_slice(&2u32.to_le_bytes()); // nchunk
    std::fs::write(shard_path, shard).expect("write invalid compact shard");
    invalid
}

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
    let device_dsa_ctx = MetalContext::new().expect("device DSA Metal context");
    let invalid_ctx = MetalContext::new().expect("invalid-admission Metal context");
    let misconfigured_ctx = MetalContext::new().expect("misconfigured DSA Metal context");

    let prior_compact = std::env::var_os(GPU_COMPACT_MLA_ENV);
    let prior_device_dsa = std::env::var_os(GPU_DEVICE_DSA_ENV);
    let prior_head = std::env::var_os(GPU_LM_HEAD_ENV);
    let prior_full_logits = std::env::var_os(GPU_LM_HEAD_FULL_LOGITS_ENV);
    std::env::remove_var(GPU_DEVICE_DSA_ENV);
    std::env::remove_var(GPU_LM_HEAD_ENV);
    std::env::remove_var(GPU_LM_HEAD_FULL_LOGITS_ENV);

    std::env::set_var(GPU_DEVICE_DSA_ENV, "1");
    let mode_error = match GravityGlmGpu::open_dir_with_budget_resident(
        misconfigured_ctx,
        &dir,
        true,
        512 * 1024 * 1024,
        true,
    ) {
        Ok(_) => panic!("device DSA was admitted without compact MLA"),
        Err(error) => error,
    };
    assert!(
        mode_error
            .to_string()
            .contains("requires resident state and"),
        "device DSA mode coupling did not fail closed: {mode_error}"
    );
    std::env::remove_var(GPU_DEVICE_DSA_ENV);

    let invalid = invalid_compact_geometry_fixture(&dir);
    std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
    let invalid_error = match GravityGlmGpu::open_dir_with_budget_resident(
        invalid_ctx,
        invalid.path(),
        true,
        512 * 1024 * 1024,
        true,
    ) {
        Ok(_) => panic!("D16 compact KV geometry was admitted"),
        Err(error) => error,
    };
    assert!(
        invalid_error.to_string().contains("dim=16")
            && invalid_error.to_string().contains("unsupported"),
        "invalid compact geometry did not fail in header preflight: {invalid_error}"
    );

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
    std::env::set_var(GPU_COMPACT_MLA_ENV, "1");
    std::env::set_var(GPU_DEVICE_DSA_ENV, "1");
    let compact_device_dsa = GravityGlmGpu::open_dir_with_budget_resident(
        device_dsa_ctx,
        &dir,
        true,
        512 * 1024 * 1024,
        true,
    )
    .expect("compact resident device DSA fixture");
    std::env::remove_var(GPU_DEVICE_DSA_ENV);
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
        let compact_waits = compact
            .last_resident_waits()
            .expect("compact resident wait count");
        let (device_dsa_logits, device_dsa_trace) = compact_device_dsa
            .forward(&prompt)
            .expect("compact device DSA forward");
        let device_dsa_waits = compact_device_dsa
            .last_resident_waits()
            .expect("device DSA resident wait count");
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
        let device_dsa_pair = score_pair(
            &expanded_logits,
            &device_dsa_logits,
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
        eprintln!(
            "device DSA case {case}: rel_l2={:.3e} meaningful={:.3e}; \
             greedy={} top5={}; waits host-rank={} device-rank={}",
            device_dsa_pair.device.continuous.relative_l2,
            device_dsa_pair.device.continuous.max_meaningful_rel,
            device_dsa_pair.device.discrete.greedy_match,
            device_dsa_pair.device.discrete.top_k_exact_match,
            compact_waits,
            device_dsa_waits
        );
        assert!(
            pair.pass,
            "case {case} prompt {prompt:?}: compact complete-token V2.1 {pair:#?}"
        );
        assert!(
            device_dsa_pair.pass,
            "case {case} prompt {prompt:?}: device DSA complete-token V2.1 {device_dsa_pair:#?}"
        );
        assert_eq!(
            compact_trace.final_topk, expanded_trace.final_topk,
            "case {case}: exact DSA selection"
        );
        assert_eq!(
            compact_trace.expert_choices, expanded_trace.expert_choices,
            "case {case}: exact expert choices"
        );
        assert_eq!(
            device_dsa_trace.final_topk, expanded_trace.final_topk,
            "case {case}: exact device DSA selection"
        );
        assert_eq!(
            device_dsa_trace.expert_choices, expanded_trace.expert_choices,
            "case {case}: exact device DSA expert choices"
        );
        assert_eq!(
            compact_waits.saturating_sub(device_dsa_waits),
            prompt.len() as u64,
            "case {case}: one full-indexer rank drain must be removed per token"
        );
    }

    match prior_compact {
        Some(value) => std::env::set_var(GPU_COMPACT_MLA_ENV, value),
        None => std::env::remove_var(GPU_COMPACT_MLA_ENV),
    }
    match prior_device_dsa {
        Some(value) => std::env::set_var(GPU_DEVICE_DSA_ENV, value),
        None => std::env::remove_var(GPU_DEVICE_DSA_ENV),
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
