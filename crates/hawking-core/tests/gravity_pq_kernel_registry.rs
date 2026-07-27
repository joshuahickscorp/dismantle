//! Static coverage for the additive gravity-pq kernel registry. No Metal
//! device is required; the device parity suite lives beside this test.

use half::f16;
use hawking_core::gravity::{
    parse_pq_header, pq_matvec, pq_matvec_f64_authority, PqHeader, PqMetalKernelVariant,
};
use hawking_core::metal::SHADER_GRAVITY_PQ;
use hawking_core::numeric_parity::{score_against_f64, Bounds};

fn primary_header(bits: u16) -> PqHeader {
    PqHeader {
        d: 32,
        s: 1,
        sub: 32,
        card: if bits == 8 { 256 } else { 128 },
        rows: 17,
        cols: 192,
        nchunk: 6,
        seed: 0,
        bits,
        rotate: 0,
        n_codebooks: 1,
    }
}

fn push_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn tiny_bits8_payload() -> (Vec<u8>, Vec<f32>) {
    let h = primary_header(8);
    let mut out = Vec::new();
    out.extend_from_slice(b"GLM52CPK");
    push_u16(&mut out, h.d);
    push_u16(&mut out, h.s);
    push_u16(&mut out, h.sub);
    push_u16(&mut out, h.card);
    push_u32(&mut out, h.rows);
    push_u32(&mut out, h.cols);
    push_u32(&mut out, h.nchunk);
    push_u32(&mut out, h.seed);
    push_u16(&mut out, h.bits);
    out.push(h.rotate);
    out.push(h.n_codebooks);
    out.resize(64, 0);
    for code in 0..h.card as usize {
        for j in 0..h.sub as usize {
            let v = (((code * 13 + j * 7) % 257) as f32 - 128.0) / 64.0;
            push_u16(&mut out, f16::from_f32(v).to_bits());
        }
    }
    for row in 0..h.rows as usize {
        for chunk in 0..h.nchunk as usize {
            out.push(((row * 37 + chunk * 19 + row * chunk) & 255) as u8);
        }
    }
    let x = (0..h.cols as usize)
        .map(|i| ((i as f32 + 0.25) * 0.03125).sin() + (i % 11) as f32 * 0.0078125)
        .collect();
    (out, x)
}

#[test]
fn registry_is_explicit_unique_and_keeps_generic_first() {
    assert_eq!(PqMetalKernelVariant::ALL[0], PqMetalKernelVariant::Generic);
    assert_eq!(
        PqMetalKernelVariant::Generic.kernel_name(),
        "gravity_pq_matvec"
    );
    let names: std::collections::HashSet<_> = PqMetalKernelVariant::ALL
        .iter()
        .map(|v| v.as_str())
        .collect();
    assert_eq!(names.len(), PqMetalKernelVariant::ALL.len());
    for variant in PqMetalKernelVariant::ALL {
        assert_eq!(
            variant.as_str().parse::<PqMetalKernelVariant>().unwrap(),
            variant
        );
    }
}

#[test]
fn primary_bits8_geometry_admits_all_candidates_but_packed_bits_do_not() {
    let bits8 = primary_header(8);
    assert!(PqMetalKernelVariant::ALL.iter().all(|v| v.supports(&bits8)));

    let bits7 = primary_header(7);
    assert!(PqMetalKernelVariant::Generic.supports(&bits7));
    assert!(PqMetalKernelVariant::ALL[1..]
        .iter()
        .all(|v| !v.supports(&bits7)));
}

#[test]
fn shader_registers_direct_vector_and_deterministic_2d_reduction() {
    for symbol in [
        "kernel void gravity_pq_matvec_bits8_direct",
        "kernel void gravity_pq_matvec_bits8_vec4",
        "kernel void gravity_pq_matvec_bits8_2d",
        "kernel void gravity_pq_reduce_2d",
    ] {
        assert!(SHADER_GRAVITY_PQ.contains(symbol), "missing {symbol}");
    }
    assert!(SHADER_GRAVITY_PQ.contains("uint(codes[flat])"));
    assert!(SHADER_GRAVITY_PQ.contains("partials[row * splits + split] = acc"));
    assert!(SHADER_GRAVITY_PQ.contains("for (uint split = 0u; split < splits; ++split)"));
    assert!(
        !SHADER_GRAVITY_PQ.contains("atomic_fetch_add"),
        "2D reduction must not use nondeterministic atomics"
    );
}

#[test]
fn pq_fp64_authority_scores_host_candidate_under_v21() {
    let (payload, x) = tiny_bits8_payload();
    let h = parse_pq_header(&payload).expect("header");
    assert_eq!((h.d, h.s, h.sub, h.card, h.bits), (32, 1, 32, 256, 8));
    let host = pq_matvec(&payload, &x).expect("host f32");
    let authority = pq_matvec_f64_authority(&payload, &x).expect("f64 authority");
    let score = score_against_f64(&host, &authority, &Bounds::continuous_only(), "host-f32");
    assert!(score.pass, "V2.1 host score failed: {:?}", score.failures);
}
