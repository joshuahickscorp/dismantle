//! CP6e -- the first interleaved-layout kernel, and the correction that forced it.
//!
//! `CP6_SCOPE.md` called the norms and elementwise kernels on the chunked path
//! "trivially K-parallel", claiming a K-wide launch was the whole change. Trying
//! to build the chunked MLP refuted that. The multi-position matmul kernels
//! require INTERLEAVED activations (`input[col * K + k]`) because their inner
//! loop reads K contiguous floats at `(col + i) * K` -- one coalesced load
//! feeding all R rows. A per-position kernel bound at an offset addresses a
//! BLOCKED layout, so it cannot be reused. Every elementwise and norm kernel on
//! the chunked path needs a strided variant.
//!
//! This proves the first one. It needs NO resident: a bare MetalContext, two
//! buffers, seconds to run -- so it costs nothing to keep as a standing gate.
//!
//! Two checks, because equality alone would pass a kernel that ignored k:
//!
//!   EQUIVALENCE  interleaved(K) de-interleaved == K separate blocked runs
//!   ISOLATION    perturbing position j changes ONLY position j's output
//!
//! ISOLATION is the negative control. A kernel that reduced over the whole
//! interleaved buffer instead of one position's stride would still be
//! self-consistent; it would fail here and only here.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run()
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::metal::MetalContext;
    use std::process;

    const HIDDEN: usize = 5120;
    const TG: u32 = 256;
    const EPS: f32 = 1e-6;

    // Deterministic, no rand dependency: a plain LCG so the fixture is pinned.
    fn fixture(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect()
    }

    fn as_bytes(v: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
    }

    fn read_f32(buf: &metal::Buffer, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) }.to_vec()
    }

    /// One position through the PRODUCTION per-position kernel. This is the
    /// reference: whatever it computes is by definition correct, because it is
    /// what the real runtime already runs.
    fn blocked(ctx: &MetalContext, x: &[f32], w: &[f32]) -> Vec<f32> {
        let xb = ctx.new_buffer_with_bytes(as_bytes(x));
        let wb = ctx.new_buffer_with_bytes(as_bytes(w));
        let ob = ctx.new_buffer(HIDDEN * 4);
        let hidden = HIDDEN as u32;
        ctx.dispatch_threads("qwen80_residual_rmsnorm_tg", (TG, 1, 1), (TG, 1, 1), |e| {
            e.set_buffer(0, Some(&xb), 0);
            e.set_buffer(1, Some(&wb), 0);
            e.set_buffer(2, Some(&ob), 0);
            e.set_bytes(3, 4, &hidden as *const u32 as *const _);
            e.set_bytes(4, 4, &EPS as *const f32 as *const _);
            e.set_threadgroup_memory_length(0, (TG as u64) * 4);
        })
        .expect("blocked rmsnorm");
        read_f32(&ob, HIDDEN)
    }

    /// K positions in ONE dispatch through the interleaved kernel.
    fn interleaved(ctx: &MetalContext, xi: &[f32], w: &[f32], chunk: usize) -> Vec<f32> {
        let xb = ctx.new_buffer_with_bytes(as_bytes(xi));
        let wb = ctx.new_buffer_with_bytes(as_bytes(w));
        let ob = ctx.new_buffer(HIDDEN * chunk * 4);
        let hidden = HIDDEN as u32;
        let k = chunk as u32;
        ctx.dispatch_threads(
            "qwen38_residual_rmsnorm_tg_interleaved",
            (TG * k, 1, 1),
            (TG, 1, 1),
            |e| {
                e.set_buffer(0, Some(&xb), 0);
                e.set_buffer(1, Some(&wb), 0);
                e.set_buffer(2, Some(&ob), 0);
                e.set_bytes(3, 4, &hidden as *const u32 as *const _);
                e.set_bytes(4, 4, &EPS as *const f32 as *const _);
                e.set_bytes(5, 4, &k as *const u32 as *const _);
                e.set_threadgroup_memory_length(0, (TG as u64) * 4);
            },
        )
        .expect("interleaved rmsnorm");
        read_f32(&ob, HIDDEN * chunk)
    }

    pub fn run() {
        let ctx = match MetalContext::new() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no Metal context: {e}");
                process::exit(1);
            }
        };
        let w = fixture(HIDDEN, 0xBEEF);
        let mut failures = 0usize;

        for &chunk in &[1usize, 2, 4, 8] {
            // Per-position fixtures, then the interleaved packing of the same bytes.
            let pos: Vec<Vec<f32>> = (0..chunk).map(|k| fixture(HIDDEN, 0x5EED + k as u64)).collect();
            let mut xi = vec![0f32; HIDDEN * chunk];
            for (k, p) in pos.iter().enumerate() {
                for c in 0..HIDDEN {
                    xi[c * chunk + k] = p[c];
                }
            }

            let got = interleaved(&ctx, &xi, &w, chunk);
            let mut max_abs = 0f64;
            for (k, p) in pos.iter().enumerate() {
                let want = blocked(&ctx, p, &w);
                for c in 0..HIDDEN {
                    let d = (got[c * chunk + k] as f64 - want[c] as f64).abs();
                    if d > max_abs {
                        max_abs = d;
                    }
                }
            }
            // Same arithmetic, same order, same threadgroup width: this should be
            // bit-identical, not merely close. Any drift means the reduction
            // changed shape, which is exactly what we are testing for.
            let equiv = max_abs == 0.0;
            if !equiv {
                failures += 1;
            }
            println!(
                "  K={chunk}  EQUIVALENCE {}  max_abs {max_abs:.3e}",
                if equiv { "PASS" } else { "FAIL" }
            );

            // ISOLATION: move ONE position's input, confirm the others are byte-frozen.
            if chunk > 1 {
                let target = chunk / 2;
                let mut xp = xi.clone();
                for c in 0..HIDDEN {
                    xp[c * chunk + target] += 1.0;
                }
                let got2 = interleaved(&ctx, &xp, &w, chunk);
                let mut bled = 0usize;
                let mut moved = false;
                for c in 0..HIDDEN {
                    for k in 0..chunk {
                        let changed = got2[c * chunk + k] != got[c * chunk + k];
                        if k == target {
                            moved |= changed;
                        } else if changed {
                            bled += 1;
                        }
                    }
                }
                let isolated = bled == 0 && moved;
                if !isolated {
                    failures += 1;
                }
                println!(
                    "  K={chunk}  ISOLATION   {}  bled {bled} of {} (target moved: {moved})",
                    if isolated { "PASS" } else { "FAIL" },
                    HIDDEN * (chunk - 1)
                );
            }
        }

        if failures > 0 {
            eprintln!("{failures} check(s) FAILED");
            process::exit(1);
        }
        println!("all checks PASS");
    }
}
