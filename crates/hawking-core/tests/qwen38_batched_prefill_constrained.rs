//! Constrained-prefill adversarial check: the grammar route must use the same
//! cold batched prompt path without changing generated token identity.

#![cfg(target_os = "macos")]

use hawking_core::json_constrain::{JsonConstraint, JsonVocabIndex};
use hawking_core::model::qwen38_hybrid_decode::{
    generate_constrained, load_qwen38_tokenizer, Qwen38HybridDecodeSession, Qwen38HybridWeights,
};
use std::sync::Arc;

const ARTIFACT: &str = "/Users/scammermike/noetic/NOETIC_PARENT_A";
const TOKENIZER: &str = "/Users/scammermike/noetic/NOETIC_PARENT_A/tokenizer.json";
const MAX_NEW: usize = 24;
const MAX_SEQ: usize = 2048;

fn fusion_env() {
    std::env::set_var("HAWKING_QWEN38_FUSE_ADD_RMSNORM", "1");
    std::env::set_var("HAWKING_QWEN38_FUSE_GQA_QKV", "1");
    std::env::set_var("HAWKING_QWEN38_FUSE_DN_INPROJ", "1");
    std::env::set_var("HAWKING_QWEN38_FUSE_MLP", "swiglu");
    std::env::set_var("HAWKING_QWEN38_IGNORE_EOS", "1");
}

fn artifact_present() -> bool {
    std::path::Path::new(ARTIFACT)
        .join("catalog.hq38m20")
        .is_file()
        && std::path::Path::new(TOKENIZER).is_file()
}

#[test]
fn batched_prefill_constrained_matches_sequential_cold_route() {
    if !artifact_present() {
        eprintln!("skipping constrained prefill: artifact missing at {ARTIFACT}");
        return;
    }
    if hawking_core::metal::MetalContext::new().is_err() {
        eprintln!("skipping constrained prefill: no Metal GPU");
        return;
    }
    fusion_env();
    let weights = match Qwen38HybridWeights::load(ARTIFACT) {
        Ok(weights) => Arc::new(weights),
        Err(err) => {
            eprintln!("skipping constrained prefill: load failed: {err}");
            return;
        }
    };
    let tokenizer = load_qwen38_tokenizer(TOKENIZER).expect("tokenizer");
    // A tiny prompt cannot satisfy a 4x dispatch gate: one batched chunk still
    // has fixed setup cost. Keep the generated suffix short, but make the
    // prompt long enough to exercise the optimization's intended regime.
    let prompt_text = "Return one JSON object. ".repeat(64);
    let prompt = tokenizer.encode(&prompt_text, false).expect("encode");
    assert!(!prompt.is_empty(), "prompt tokenized empty");
    let vocab_size = tokenizer.vocab_size();
    let vocab = JsonVocabIndex::build(vocab_size, |id| {
        tokenizer.decode(&[id], false).unwrap_or_default()
    });

    std::env::set_var("HAWKING_QWEN38_BATCH_PREFILL", "0");
    let mut sequential = Qwen38HybridDecodeSession::attach(Arc::clone(&weights), MAX_SEQ)
        .expect("attach sequential");
    let mut seq_constraint = JsonConstraint::new();
    let (seq, _) = generate_constrained(
        &mut sequential,
        &tokenizer,
        &vocab,
        &mut seq_constraint,
        &prompt,
        MAX_NEW,
        0,
        None,
    )
    .expect("sequential constrained generation");
    assert!(!seq.batched_prefill);

    std::env::set_var("HAWKING_QWEN38_BATCH_PREFILL", "1");
    let mut batched =
        Qwen38HybridDecodeSession::attach(Arc::clone(&weights), MAX_SEQ).expect("attach batched");
    let mut bat_constraint = JsonConstraint::new();
    let (bat, _) = generate_constrained(
        &mut batched,
        &tokenizer,
        &vocab,
        &mut bat_constraint,
        &prompt,
        MAX_NEW,
        0,
        None,
    )
    .expect("batched constrained generation");
    // Print the measurement. The assertions below are unchanged; a passing test
    // is a verdict, and G005 needs the numbers behind it.
    println!(
        "PARITY prompt_tokens={} seq_dispatches={} bat_dispatches={} ratio={:.2} \
seq_prefill_ns={} bat_prefill_ns={} speedup={:.3} seq_tokens={:?} bat_tokens={:?} \
identical={} seq_stop={} bat_stop={}",
        prompt.len(),
        seq.prefill_dispatches,
        bat.prefill_dispatches,
        seq.prefill_dispatches as f64 / bat.prefill_dispatches.max(1) as f64,
        seq.prefill_wall_ns,
        bat.prefill_wall_ns,
        seq.prefill_wall_ns as f64 / bat.prefill_wall_ns.max(1) as f64,
        seq.new_tokens(),
        bat.new_tokens(),
        seq.new_tokens() == bat.new_tokens(),
        seq.stop_reason,
        bat.stop_reason,
    );

    assert!(bat.batched_prefill);
    assert!(bat.prefill_dispatches > 0);
    assert!(bat.prefill_dispatches < seq.prefill_dispatches / 4);
    assert_eq!(seq.new_tokens(), bat.new_tokens());
    assert_eq!(seq.stop_reason, bat.stop_reason);

    std::env::remove_var("HAWKING_QWEN38_BATCH_PREFILL");
}
