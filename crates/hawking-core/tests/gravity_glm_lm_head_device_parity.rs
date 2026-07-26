//! Multi-vector parity: device-resident native.bf16 lm_head vs host widen+matvec.
//!
//! The flagship head is `native.bf16` [154880, 6144] and is the second-largest
//! per-token cost bucket. This suite does not need the sealed artifact — it
//! builds synthetic matrices and checks **bit-identical** logits (and argmax)
//! against the host oracle over **several** activation vectors. A single-vector
//! pass is a documented failure mode in this codebase.
//!
//! Requires Metal. The executor sandbox often has none; the controller runs:
//!
//! ```text
//! cargo test -p hawking-core --test gravity_glm_lm_head_device_parity -- --nocapture
//! ```
//!
//! Flag for the full model path (not required for this unit suite):
//! `HAWKING_GLM_GPU_LM_HEAD=1` (with optional `HAWKING_GLM_GPU_RESIDENT_STATE=1`).

#![cfg(target_os = "macos")]

use hawking_core::gravity::matvec_bf16_host;
use hawking_core::kernels::argmax_f32;
use hawking_core::metal::MetalContext;

fn top1(logits: &[f32]) -> u32 {
    argmax_f32(logits)
}

/// Several independent "prompts" (activation vectors). Phase-aligned luck on
/// one vector must not pass the suite.
fn activations(cols: usize) -> Vec<Vec<f32>> {
    vec![
        (0..cols).map(|c| (c as f32) * 0.01 - 0.3).collect(),
        (0..cols)
            .map(|c| ((c * 5 + 3) % 17) as f32 * 0.05 - 0.4)
            .collect(),
        vec![1.0; cols],
        vec![0.0; cols],
        (0..cols)
            .map(|c| if c % 3 == 0 { 0.5 } else { -0.25 })
            .collect(),
        (0..cols).map(|c| ((c as f32).sin()) * 0.1).collect(),
        (0..cols)
            .map(|c| if c < cols / 2 { 0.125 } else { -0.0625 })
            .collect(),
        // Second shape: different seed pattern
        (0..cols)
            .map(|c| ((c * 13 + 7) % 31) as f32 * 0.02 - 0.3)
            .collect(),
    ]
}

fn make_bf16_matrix(rows: usize, cols: usize, salt: u32) -> Vec<u8> {
    let mut bits = Vec::with_capacity(rows * cols * 2);
    for i in 0..(rows * cols) {
        // Keep values in the finite normal-ish bf16 range; avoid NaN/Inf.
        let u = (((i as u32).wrapping_mul(37).wrapping_add(salt)) % 0x7F00) as u16;
        bits.extend_from_slice(&u.to_le_bytes());
    }
    bits
}

#[test]
fn device_bf16_lm_head_matches_host_over_several_vectors() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(e) => {
            // A shader COMPILE failure is not an absent device. Only a
            // genuinely missing device may skip; compile errors fail loudly.
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

    // Two matrix shapes so one accidental dimension alignment cannot hide bugs.
    let shapes = [(64usize, 32usize), (257usize, 17usize), (16usize, 64usize)];

    for &(rows, cols) in &shapes {
        let weight = make_bf16_matrix(rows, cols, (rows * cols) as u32);
        let w_buf = ctx
            .new_buffer_with_bytes_checked(&weight)
            .expect("upload weight");

        for (vi, x) in activations(cols).into_iter().enumerate() {
            let host = matvec_bf16_host(&weight, cols, &x).expect("host oracle");
            let device = hawking_core::gravity_glm::gpu::dispatch_gemv_native_bf16_seq(
                &ctx, &w_buf, rows as u32, cols as u32, &x,
            )
            .unwrap_or_else(|e| {
                // Kernel missing / compile-time name failure must not look green.
                panic!(
                    "device gemv failed (rows={rows} cols={cols} vec={vi}): {e} — \
                     if this is a shader compile issue it is a hard fail"
                );
            });

            assert_eq!(
                device.len(),
                host.len(),
                "rows={rows} cols={cols} vec={vi}: length"
            );
            assert_eq!(
                device, host,
                "rows={rows} cols={cols} vec={vi}: logits must be bit-identical \
                 (sequential f32 accumulate after bf16 widen)"
            );
            assert_eq!(
                top1(&device),
                top1(&host),
                "rows={rows} cols={cols} vec={vi}: argmax"
            );
        }
        eprintln!("ok shape [{rows}, {cols}] over {} vectors", activations(cols).len());
    }
}
