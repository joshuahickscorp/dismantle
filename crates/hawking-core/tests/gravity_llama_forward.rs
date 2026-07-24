//! Gate S0.8: the runtime executes a complete token out of a `.gravity`
//! artifact and reproduces the frozen numpy-oracle logits.
//!
//! The reference in `tests/fixtures/gravity_llama/` was produced by
//! `tools/condense/gravity_llama_reference.py` reading the same container
//! through the same codec, so this grades the runtime on what the artifact
//! encodes -- not on how close a 0.877-BPW artifact lands to the BF16
//! parent it never claimed to be.
//!
//! The artifact itself is 135 MB and lives outside the repo. Override the
//! path with `HAWKING_GRAVITY_LLAMA_ARTIFACT`.

use std::path::PathBuf;

use hawking_core::gravity_llama::GravityLlama;

const DEFAULT_ARTIFACT: &str =
    "Library/Application Support/Hawking/CampaignS08/llama32-1b-R0.v2.gravity";

fn artifact_path() -> Option<PathBuf> {
    let p = match std::env::var_os("HAWKING_GRAVITY_LLAMA_ARTIFACT") {
        Some(v) => PathBuf::from(v),
        None => PathBuf::from(std::env::var_os("HOME")?).join(DEFAULT_ARTIFACT),
    };
    p.is_file().then_some(p)
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gravity_llama")
}

#[derive(serde::Deserialize)]
struct Reference {
    tokens: Vec<u32>,
    argmax: u32,
    top5: Vec<u32>,
    logits_head: Vec<f32>,
    tied_head: bool,
}

fn top_k(logits: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .expect("no NaN in logits")
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

#[test]
fn gravity_llama_forward_matches_frozen_oracle() {
    let Some(art) = artifact_path() else {
        eprintln!("skipping gravity_llama_forward: no llama32-1b .gravity artifact on disk");
        return;
    };
    let dir = fixtures_dir();
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(dir.join("ref_3tok.json")).expect("read ref_3tok"))
            .expect("parse ref_3tok");
    let want: Vec<f32> = std::fs::read(dir.join("ref_logits_3tok.f32"))
        .expect("read ref logits")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let model = GravityLlama::open(&art, true).expect("open .gravity artifact");
    assert_eq!(
        model.tied_head, reference.tied_head,
        "head tying disagrees with the oracle"
    );
    assert_eq!(
        model.arch.vocab_size,
        want.len(),
        "vocab_size vs reference logit count"
    );

    let got = model.forward(&reference.tokens).expect("forward");
    assert_eq!(got.len(), want.len(), "logit count");

    // Worst violator by tolerance margin, so the reported culprit is the
    // true worst offender rather than merely the largest raw difference.
    let (mut worst_i, mut worst_margin, mut worst_diff, mut worst_tol) = (0usize, f32::NEG_INFINITY, 0f32, 0f32);
    for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (a - b).abs();
        // Measured worst on the sealed R0 artifact is 2.6e-5 absolute; this
        // leaves ~40x headroom for f32 reassociation without going so slack
        // that a real regression slips through.
        let tol = 1e-3 + 1e-4 * b.abs();
        if diff - tol > worst_margin {
            worst_margin = diff - tol;
            worst_i = i;
            worst_diff = diff;
            worst_tol = tol;
        }
    }
    assert!(
        worst_margin <= 0.0,
        "logit {worst_i}: got {}, want {}, diff {worst_diff} > tol {worst_tol}",
        got[worst_i],
        want[worst_i]
    );
    let max_abs = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("gravity_llama_forward: max |logit diff| vs oracle = {max_abs:.6e}");

    for (i, &w) in reference.logits_head.iter().enumerate() {
        let diff = (got[i] - w).abs();
        assert!(
            diff <= 1e-3 + 1e-4 * w.abs(),
            "logits_head[{i}]: got {}, want {w}, diff {diff}",
            got[i]
        );
    }

    let got_top5 = top_k(&got, 5);
    assert_eq!(
        got_top5[0], reference.argmax,
        "argmax: got {}, want {}",
        got_top5[0], reference.argmax
    );
    assert_eq!(got_top5, reference.top5, "top-5");
}
