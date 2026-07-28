//! Shared helpers for hawking-core integration tests.
//!
//! Metal buffer helpers (`ctx`, `new_f32_buf`, `read_f32_buf`) are macOS-only
//! and are only pulled in via `mod common;` from `#![cfg(target_os = "macos")]`
//! test binaries.
//!
//! Path / argmax helpers are available on every platform so non-Metal engine
//! parity tests can share them too.
//!
//! Different tolerances, tie-breaks, path roots, env overrides, and skip
//! messaging stay local to the files that need them — do not unify those.
#![allow(dead_code)]

use std::path::PathBuf;

// ── path helpers ────────────────────────────────────────────────────────────

/// Default Qwen2.5-3B Q4_K_M GGUF path used by multiseq / w4a8 parity tests.
pub fn weights_path_qwen() -> PathBuf {
    PathBuf::from("../../models/qwen2.5-3b-instruct-q4_k_m.gguf")
}

/// Default DeepSeek-V2-Lite Q4 GGUF path used by v1.1 phase tests.
pub fn weights_path_deepseek() -> PathBuf {
    PathBuf::from("../../models/deepseek-v2-lite-q4.gguf")
}

// ── argmax ──────────────────────────────────────────────────────────────────

/// Argmax over `f32` logits with **first-index-wins** on ties (strict `>`).
///
/// Callers that need last-wins (`Iterator::max_by` / `partial_cmp`) or a
/// `usize` return keep a local copy — those differ in tie-break / type.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i as u32;
        }
    }
    best
}

// ── metal / numeric helpers (macOS) ─────────────────────────────────────────

/// Deterministic pseudo-random `f32` vector in `[-1, 1)` from a fixed seed.
pub fn fixed_f32(n: usize, seed: u64) -> Vec<f32> {
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;
    let mut rng = Pcg64Mcg::new(seed as u128);
    (0..n).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect()
}

/// Maximum absolute element-wise difference between two equal-length slices.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Standard fp16 parity tolerance (Metal kernel vs CPU reference).
pub const ATOL: f32 = 1e-3;

#[cfg(target_os = "macos")]
use hawking_core::metal::{MetalContext, PinnedBuffer};
#[cfg(target_os = "macos")]
use once_cell::sync::Lazy;

/// Process-wide lazily-initialized Metal context (one device per test binary).
#[cfg(target_os = "macos")]
pub fn ctx() -> &'static MetalContext {
    static CTX: Lazy<MetalContext> =
        Lazy::new(|| MetalContext::new().expect("Metal device required"));
    &CTX
}

/// Upload an `f32` slice into a pinned Metal buffer.
#[cfg(target_os = "macos")]
pub fn new_f32_buf(ctx: &MetalContext, data: &[f32]) -> PinnedBuffer {
    ctx.new_buffer_with_bytes(bytemuck::cast_slice(data))
}

/// Read `n` `f32`s back out of a pinned Metal buffer.
#[cfg(target_os = "macos")]
pub fn read_f32_buf(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
    let ptr = buf.contents() as *const f32;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}
