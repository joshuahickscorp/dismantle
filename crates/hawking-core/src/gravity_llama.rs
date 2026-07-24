//! Llama-family forward pass executed directly out of a `.gravity` shard.
//!
//! This is the production counterpart of the numpy oracle in
//! `tools/condense/gravity_llama_reference.py`, and it is graded against
//! that oracle's frozen logits (`tests/fixtures/gravity_llama/`) rather
//! than against the BF16 parent. A sub-bit artifact is lossy by
//! construction; "correct" means the runtime computes what the *artifact*
//! encodes, so the oracle reads the same container through the same codec
//! and the two must agree.
//!
//! No dense reconstruction anywhere: every projection is a matvec over the
//! packed `gravity-pq` payload, and the embedding row is decoded from its
//! own chunk codes rather than by materializing the `[vocab, hidden]`
//! matrix. No source weights are consulted.

use std::collections::HashMap;
use std::path::Path;

use crate::attn::mha_decode_step;
use crate::gravity::{GravityShard, PqTensor};
use crate::kernels::{add_inplace, rmsnorm, rope_inplace_scaled, silu_mul, Llama3RopeScaling};
use crate::{Error, Result};

/// The architecture fields the forward pass needs, read from the shard
/// header's `architecture` object. Absent or malformed fields are an
/// error: a runtime that guesses `rope_theta` produces plausible garbage.
#[derive(Debug, Clone)]
pub struct GravityLlamaArch {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub rope_scaling: Option<Llama3RopeScaling>,
}

fn arch_u64(v: &serde_json::Value, key: &str) -> Result<u64> {
    v.get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| Error::Gravity(format!("architecture.{key} missing or not an integer")))
}

fn arch_f64(v: &serde_json::Value, key: &str) -> Result<f64> {
    v.get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| Error::Gravity(format!("architecture.{key} missing or not a number")))
}

impl GravityLlamaArch {
    pub fn from_header(extra: &serde_json::Value) -> Result<GravityLlamaArch> {
        let a = extra
            .get("architecture")
            .ok_or_else(|| Error::Gravity("shard header has no `architecture`".into()))?;

        let model_type = a.get("model_type").and_then(serde_json::Value::as_str);
        if model_type != Some("llama") {
            return Err(Error::Gravity(format!(
                "gravity_llama: architecture.model_type is {model_type:?}, expected \"llama\""
            )));
        }

        let hidden = arch_u64(a, "hidden_size")? as usize;
        let n_heads = arch_u64(a, "num_attention_heads")? as usize;
        // `head_dim` is explicit in recent configs; older ones imply it.
        let head_dim = match a.get("head_dim").and_then(serde_json::Value::as_u64) {
            Some(hd) => hd as usize,
            None => {
                if n_heads == 0 || hidden % n_heads != 0 {
                    return Err(Error::Gravity(format!(
                        "cannot infer head_dim from hidden_size {hidden} / num_attention_heads {n_heads}"
                    )));
                }
                hidden / n_heads
            }
        };

        // Only `rope_type == "llama3"` is a scaling this decoder implements.
        // Any other declared rope_type is refused rather than silently
        // executed as unscaled RoPE, which would be a wrong model that runs.
        let rope_scaling = match a.get("rope_scaling") {
            None | Some(serde_json::Value::Null) => None,
            Some(rs) => {
                let ty = rs.get("rope_type").and_then(serde_json::Value::as_str);
                if ty != Some("llama3") {
                    return Err(Error::Gravity(format!(
                        "unsupported architecture.rope_scaling.rope_type {ty:?}"
                    )));
                }
                Some(Llama3RopeScaling {
                    factor: arch_f64(rs, "factor")? as f32,
                    low_freq_factor: arch_f64(rs, "low_freq_factor")? as f32,
                    high_freq_factor: arch_f64(rs, "high_freq_factor")? as f32,
                    original_max_position_embeddings: arch_u64(
                        rs,
                        "original_max_position_embeddings",
                    )? as u32,
                })
            }
        };

        Ok(GravityLlamaArch {
            n_layers: arch_u64(a, "num_hidden_layers")? as usize,
            hidden,
            n_heads,
            n_kv_heads: arch_u64(a, "num_key_value_heads")? as usize,
            head_dim,
            vocab_size: arch_u64(a, "vocab_size")? as usize,
            rope_theta: arch_f64(a, "rope_theta")? as f32,
            rms_norm_eps: arch_f64(a, "rms_norm_eps")? as f32,
            rope_scaling,
        })
    }
}

/// One tensor as the forward pass consumes it: either a packed PQ payload
/// decoded once, or a natively-carried dense tensor widened to f32.
enum Weight {
    Pq(PqTensor),
    Dense(Vec<f32>),
}

/// A `.gravity` shard loaded as an executable Llama model.
pub struct GravityLlama {
    pub arch: GravityLlamaArch,
    weights: HashMap<String, Weight>,
    /// `lm_head.weight` when the artifact carries one, otherwise the tied
    /// `model.embed_tokens.weight`.
    head_name: String,
    pub tied_head: bool,
}

/// Widen a `native.<dtype>` payload to f32. Mirrors the oracle's dtype
/// handling; an unknown dtype is an error rather than a reinterpretation.
fn widen_native(codec: &str, blob: &[u8]) -> Result<Vec<f32>> {
    let dtype = codec.split_once('.').map(|(_, d)| d).unwrap_or("");
    match dtype {
        "bf16" => {
            if blob.len() % 2 != 0 {
                return Err(Error::Gravity(format!(
                    "native.bf16 payload length {} is not even",
                    blob.len()
                )));
            }
            Ok(blob
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect())
        }
        "f16" => {
            if blob.len() % 2 != 0 {
                return Err(Error::Gravity(format!(
                    "native.f16 payload length {} is not even",
                    blob.len()
                )));
            }
            Ok(blob
                .chunks_exact(2)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect())
        }
        "f32" => {
            if blob.len() % 4 != 0 {
                return Err(Error::Gravity(format!(
                    "native.f32 payload length {} is not a multiple of 4",
                    blob.len()
                )));
            }
            Ok(blob
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        other => Err(Error::Gravity(format!(
            "unsupported native tensor dtype {other:?} (codec {codec:?})"
        ))),
    }
}

impl GravityLlama {
    /// Load every tensor in the shard. `verify_hash` checks each payload
    /// against the descriptor's SHA-256 on the way in — on by default for
    /// anything that matters, since a silently corrupt weight produces
    /// confident wrong logits.
    pub fn open(path: &Path, verify_hash: bool) -> Result<GravityLlama> {
        let shard = GravityShard::open(path)?;
        let arch = GravityLlamaArch::from_header(&shard.extra)?;

        let names: Vec<String> = shard.tensor_names().map(str::to_string).collect();
        let mut weights = HashMap::with_capacity(names.len());
        for name in &names {
            let codec = shard
                .descriptor(name)
                .expect("name came from tensor_names")
                .codec
                .clone();
            let blob = shard.read_tensor(name, verify_hash)?;
            let w = if codec == "gravity-pq" {
                Weight::Pq(PqTensor::from_payload(&blob)?)
            } else if codec.starts_with("native.") {
                Weight::Dense(widen_native(&codec, &blob)?)
            } else {
                return Err(Error::Gravity(format!(
                    "tensor {name}: unsupported codec {codec:?}"
                )));
            };
            weights.insert(name.clone(), w);
        }

        let tied_head = !weights.contains_key("lm_head.weight");
        let head_name = if tied_head {
            "model.embed_tokens.weight".to_string()
        } else {
            "lm_head.weight".to_string()
        };
        if !weights.contains_key(&head_name) {
            return Err(Error::Gravity(format!(
                "artifact has neither lm_head.weight nor model.embed_tokens.weight"
            )));
        }

        Ok(GravityLlama {
            arch,
            weights,
            head_name,
            tied_head,
        })
    }

    fn weight(&self, name: &str) -> Result<&Weight> {
        self.weights
            .get(name)
            .ok_or_else(|| Error::Gravity(format!("artifact has no tensor {name:?}")))
    }

    fn dense(&self, name: &str) -> Result<&[f32]> {
        match self.weight(name)? {
            Weight::Dense(v) => Ok(v),
            Weight::Pq(_) => Err(Error::Gravity(format!(
                "tensor {name:?} is packed; expected a natively-carried dense tensor"
            ))),
        }
    }

    fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        match self.weight(name)? {
            Weight::Pq(t) => t.matvec(x),
            // A natively-carried 2D tensor is stored row-major [rows, cols].
            Weight::Dense(w) => {
                if x.is_empty() || w.len() % x.len() != 0 {
                    return Err(Error::Gravity(format!(
                        "tensor {name:?}: {} values is not a whole number of {}-wide rows",
                        w.len(),
                        x.len()
                    )));
                }
                let cols = x.len();
                Ok(w.chunks_exact(cols)
                    .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                    .collect())
            }
        }
    }

    /// One row of a weight matrix — the embedding lookup path.
    fn row(&self, name: &str, index: usize) -> Result<Vec<f32>> {
        match self.weight(name)? {
            Weight::Pq(t) => t.row(index),
            Weight::Dense(w) => {
                let cols = self.arch.hidden;
                let start = index * cols;
                if start + cols > w.len() {
                    return Err(Error::Gravity(format!(
                        "tensor {name:?}: row {index} out of range"
                    )));
                }
                Ok(w[start..start + cols].to_vec())
            }
        }
    }

    /// Run `tokens` through the model from an empty KV cache and return the
    /// logits after the final token — `vocab_size` values.
    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(Error::Gravity("forward: no tokens".into()));
        }
        let a = &self.arch;
        let kv_width = a.n_kv_heads * a.head_dim;

        // KV cache per layer, appended one position per token, laid out
        // [pos][kv_head][head_dim] to match `mha_decode_step`.
        let mut k_cache: Vec<Vec<f32>> = vec![Vec::new(); a.n_layers];
        let mut v_cache: Vec<Vec<f32>> = vec![Vec::new(); a.n_layers];

        let mut scratch = vec![0f32; a.hidden];
        let mut logits = Vec::new();

        for (pos, &token) in tokens.iter().enumerate() {
            if token as usize >= a.vocab_size {
                return Err(Error::Gravity(format!(
                    "token {token} out of range for vocab_size {}",
                    a.vocab_size
                )));
            }
            let mut x = self.row("model.embed_tokens.weight", token as usize)?;
            if x.len() != a.hidden {
                return Err(Error::Gravity(format!(
                    "embedding row is {} wide, expected hidden_size {}",
                    x.len(),
                    a.hidden
                )));
            }

            for layer in 0..a.n_layers {
                let p = format!("model.layers.{layer}.");

                rmsnorm(
                    &x,
                    self.dense(&format!("{p}input_layernorm.weight"))?,
                    a.rms_norm_eps,
                    &mut scratch,
                );
                let mut q = self.matvec(&format!("{p}self_attn.q_proj.weight"), &scratch)?;
                let mut k = self.matvec(&format!("{p}self_attn.k_proj.weight"), &scratch)?;
                let v = self.matvec(&format!("{p}self_attn.v_proj.weight"), &scratch)?;

                for h in 0..a.n_heads {
                    rope_inplace_scaled(
                        &mut q[h * a.head_dim..(h + 1) * a.head_dim],
                        pos as u32,
                        a.rope_theta,
                        a.rope_scaling,
                    );
                }
                for h in 0..a.n_kv_heads {
                    rope_inplace_scaled(
                        &mut k[h * a.head_dim..(h + 1) * a.head_dim],
                        pos as u32,
                        a.rope_theta,
                        a.rope_scaling,
                    );
                }

                k_cache[layer].extend_from_slice(&k);
                v_cache[layer].extend_from_slice(&v);
                let seq_len = k_cache[layer].len() / kv_width;

                let mut attn = vec![0f32; a.n_heads * a.head_dim];
                mha_decode_step(
                    &q,
                    &k_cache[layer],
                    &v_cache[layer],
                    a.n_heads,
                    a.n_kv_heads,
                    a.head_dim,
                    seq_len,
                    &mut attn,
                )?;

                let o = self.matvec(&format!("{p}self_attn.o_proj.weight"), &attn)?;
                add_inplace(&mut x, &o);

                rmsnorm(
                    &x,
                    self.dense(&format!("{p}post_attention_layernorm.weight"))?,
                    a.rms_norm_eps,
                    &mut scratch,
                );
                let gate = self.matvec(&format!("{p}mlp.gate_proj.weight"), &scratch)?;
                let up = self.matvec(&format!("{p}mlp.up_proj.weight"), &scratch)?;
                let mut act = vec![0f32; gate.len()];
                silu_mul(&gate, &up, &mut act);
                let down = self.matvec(&format!("{p}mlp.down_proj.weight"), &act)?;
                add_inplace(&mut x, &down);
            }

            rmsnorm(
                &x,
                self.dense("model.norm.weight")?,
                a.rms_norm_eps,
                &mut scratch,
            );
            logits = self.matvec(&self.head_name.clone(), &scratch)?;
        }

        Ok(logits)
    }
}

// ---------------------------------------------------------------------
// Resident GPU path.
// ---------------------------------------------------------------------

/// The same model with its packed weights resident in device memory.
///
/// The distinction that matters for throughput is *resident*:
/// [`crate::gravity::pq_matvec_metal`] uploads codebooks and codes on every
/// call, which is right for a parity test and useless for a runtime, where
/// each tensor is read once per token forever. Here every payload is
/// uploaded once at load, verbatim — the kernel reads `half` codebooks and
/// walks the packed index stream itself, so nothing is widened or unpacked
/// on the way in and device bytes equal artifact bytes.
#[cfg(target_os = "macos")]
pub mod gpu {
    use super::*;
    use crate::gravity::{pq_row, pq_sections, parse_pq_header, PqHeader};
    use crate::metal::{MetalContext, TokenCommandBuffer};
    use metal::Buffer;
    use std::cell::RefCell;
    use std::time::Instant;

    /// Mirror of `GravityPQParams` in `shaders/gravity_pq.metal`: eight
    /// `uint`s in declaration order, `#[repr(C)]` so a pointer cast is a
    /// valid `set_bytes` payload.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    struct PqParams {
        dim: u32,
        subspaces: u32,
        sub: u32,
        card: u32,
        rows: u32,
        cols: u32,
        nchunk: u32,
        bits: u32,
    }

    impl PqParams {
        fn from_header(h: &PqHeader) -> PqParams {
            PqParams {
                dim: h.d as u32,
                subspaces: h.s as u32,
                sub: h.sub as u32,
                card: h.card as u32,
                rows: h.rows,
                cols: h.cols,
                nchunk: h.nchunk,
                bits: h.bits as u32,
            }
        }
    }

    /// One packed tensor resident on the device.
    struct GpuPq {
        codebooks: Buffer,
        codes: Buffer,
        params: PqParams,
    }

    enum GpuWeight {
        Pq(GpuPq),
        Dense(Vec<f32>),
    }

    /// Reusable device-side activation vectors, allocated once per role and
    /// reused for every token thereafter. Keyed by role rather than by size:
    /// two roles that happen to share a width (`k` and `v`, `gate` and `up`)
    /// must still be two buffers, and a size key would silently alias them
    /// into one.
    struct BufPool {
        buffers: RefCell<HashMap<&'static str, Buffer>>,
    }

    impl BufPool {
        fn new() -> BufPool {
            BufPool {
                buffers: RefCell::new(HashMap::new()),
            }
        }

        fn get(&self, ctx: &MetalContext, role: &'static str, elems: usize) -> Buffer {
            self.buffers
                .borrow_mut()
                .entry(role)
                .or_insert_with(|| ctx.new_buffer(elems * std::mem::size_of::<f32>()))
                .clone()
        }
    }

    fn write_f32(buf: &Buffer, src: &[f32]) {
        // Safety: shared-storage buffer sized for at least `src.len()` f32s
        // by construction (`BufPool::get` is called with that element count).
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), buf.contents() as *mut f32, src.len());
        }
    }

    fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
        // Safety: same sizing contract as `write_f32`.
        unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) }.to_vec()
    }

    /// What one `forward` call cost, so throughput reporting never has to
    /// infer timings from a wall clock wrapped around the whole thing.
    ///
    /// `per_token_ms` is what separates prefill from decode: both phases run
    /// the same code here (one token at a time against a growing cache), so
    /// the only honest way to report them separately is to time each token
    /// and split the series, rather than to run two passes and pretend the
    /// second started cold.
    #[derive(Debug, Clone, Default)]
    pub struct ForwardStats {
        pub tokens: usize,
        pub command_buffers: usize,
        pub dispatches: usize,
        pub first_token_ms: f64,
        pub total_ms: f64,
        pub per_token_ms: Vec<f64>,
    }

    pub struct GravityLlamaGpu<'ctx> {
        ctx: &'ctx MetalContext,
        pub arch: GravityLlamaArch,
        weights: HashMap<String, GpuWeight>,
        /// Raw `gravity-pq` payload of the embedding table, kept for row
        /// lookup. One row per token is decoded from it directly; the same
        /// tensor is also resident on the device for the tied LM head.
        embed_payload: Vec<u8>,
        head_name: String,
        pub tied_head: bool,
        pub load_ms: f64,
        pub device_bytes: usize,
        pool: BufPool,
    }

    impl<'ctx> GravityLlamaGpu<'ctx> {
        pub fn open(
            ctx: &'ctx MetalContext,
            path: &Path,
            verify_hash: bool,
        ) -> Result<GravityLlamaGpu<'ctx>> {
            let t0 = Instant::now();
            let shard = GravityShard::open(path)?;
            let arch = GravityLlamaArch::from_header(&shard.extra)?;

            let names: Vec<String> = shard.tensor_names().map(str::to_string).collect();
            let mut weights = HashMap::with_capacity(names.len());
            let mut embed_payload = Vec::new();
            let mut device_bytes = 0usize;
            for name in &names {
                let codec = shard
                    .descriptor(name)
                    .expect("name came from tensor_names")
                    .codec
                    .clone();
                let blob = shard.read_tensor(name, verify_hash)?;
                if codec == "gravity-pq" {
                    let h = parse_pq_header(&blob)?;
                    let (cb, codes) = pq_sections(&blob)?;
                    // Four bytes of tail padding so the kernel's whole-word
                    // read at the last index's byte offset stays in bounds.
                    let mut codes_padded = Vec::with_capacity(codes.len() + 4);
                    codes_padded.extend_from_slice(codes);
                    codes_padded.extend_from_slice(&[0u8; 4]);
                    device_bytes += cb.len() + codes_padded.len();
                    weights.insert(
                        name.clone(),
                        GpuWeight::Pq(GpuPq {
                            codebooks: ctx.new_buffer_with_bytes_checked(cb)?,
                            codes: ctx.new_buffer_with_bytes_checked(&codes_padded)?,
                            params: PqParams::from_header(&h),
                        }),
                    );
                    if name == "model.embed_tokens.weight" {
                        embed_payload = blob;
                    }
                } else if codec.starts_with("native.") {
                    weights.insert(name.clone(), GpuWeight::Dense(widen_native(&codec, &blob)?));
                } else {
                    return Err(Error::Gravity(format!(
                        "tensor {name}: unsupported codec {codec:?}"
                    )));
                }
            }

            let tied_head = !weights.contains_key("lm_head.weight");
            let head_name = if tied_head {
                "model.embed_tokens.weight".to_string()
            } else {
                "lm_head.weight".to_string()
            };
            if embed_payload.is_empty() {
                return Err(Error::Gravity(
                    "artifact has no packed model.embed_tokens.weight to look rows up in".into(),
                ));
            }

            Ok(GravityLlamaGpu {
                ctx,
                arch,
                weights,
                embed_payload,
                head_name,
                tied_head,
                load_ms: t0.elapsed().as_secs_f64() * 1e3,
                device_bytes,
                pool: BufPool::new(),
            })
        }

        fn pq(&self, name: &str) -> Result<&GpuPq> {
            match self
                .weights
                .get(name)
                .ok_or_else(|| Error::Gravity(format!("artifact has no tensor {name:?}")))?
            {
                GpuWeight::Pq(t) => Ok(t),
                GpuWeight::Dense(_) => Err(Error::Gravity(format!(
                    "tensor {name:?} is dense; expected a packed projection"
                ))),
            }
        }

        fn dense(&self, name: &str) -> Result<&[f32]> {
            match self
                .weights
                .get(name)
                .ok_or_else(|| Error::Gravity(format!("artifact has no tensor {name:?}")))?
            {
                GpuWeight::Dense(v) => Ok(v),
                GpuWeight::Pq(_) => Err(Error::Gravity(format!(
                    "tensor {name:?} is packed; expected a natively-carried dense tensor"
                ))),
            }
        }

        /// Encode one packed matvec into an open command buffer. Dispatches
        /// encoded into the same buffer must write disjoint outputs.
        fn encode_matvec(
            &self,
            tcb: &mut TokenCommandBuffer<'_>,
            name: &str,
            x: &Buffer,
            y: &Buffer,
        ) -> Result<()> {
            let w = self.pq(name)?;
            // One SIMD group (32 lanes) per output row, 8 SIMD groups (256
            // threads) per threadgroup; the kernel guards the tail.
            const TG: u32 = 256;
            let n_tg = w.params.rows.div_ceil(8);
            let params = w.params;
            tcb.dispatch_threads("gravity_pq_matvec", (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(&w.codebooks), 0);
                enc.set_buffer(1, Some(&w.codes), 0);
                enc.set_buffer(2, Some(x), 0);
                enc.set_buffer(3, Some(y), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of::<PqParams>() as u64,
                    &params as *const PqParams as *const _,
                );
            })
        }

        /// Run `tokens` from an empty KV cache; returns the logits after the
        /// final token, plus what the run cost.
        pub fn forward(&self, tokens: &[u32]) -> Result<(Vec<f32>, ForwardStats)> {
            if tokens.is_empty() {
                return Err(Error::Gravity("forward: no tokens".into()));
            }
            let a = &self.arch;
            let kv_width = a.n_kv_heads * a.head_dim;
            let inter = self.pq("model.layers.0.mlp.gate_proj.weight")?.params.rows as usize;

            let x_buf = self.pool.get(self.ctx, "x", a.hidden);
            let q_buf = self.pool.get(self.ctx, "q", a.n_heads * a.head_dim);
            let k_buf = self.pool.get(self.ctx, "k", kv_width);
            let v_buf = self.pool.get(self.ctx, "v", kv_width);
            let gate_buf = self.pool.get(self.ctx, "gate", inter);
            let up_buf = self.pool.get(self.ctx, "up", inter);
            let act_buf = self.pool.get(self.ctx, "act", inter);
            let o_buf = self.pool.get(self.ctx, "o", a.hidden);
            let logits_buf = self.pool.get(self.ctx, "logits", a.vocab_size);

            let mut k_cache: Vec<Vec<f32>> = vec![Vec::new(); a.n_layers];
            let mut v_cache: Vec<Vec<f32>> = vec![Vec::new(); a.n_layers];
            let mut scratch = vec![0f32; a.hidden];
            let mut logits = Vec::new();
            let mut stats = ForwardStats {
                tokens: tokens.len(),
                ..Default::default()
            };

            let t_start = Instant::now();
            let mut t_token = t_start;
            for (pos, &token) in tokens.iter().enumerate() {
                if token as usize >= a.vocab_size {
                    return Err(Error::Gravity(format!(
                        "token {token} out of range for vocab_size {}",
                        a.vocab_size
                    )));
                }
                let mut x = pq_row(&self.embed_payload, token as usize)?;

                for layer in 0..a.n_layers {
                    let p = format!("model.layers.{layer}.");

                    rmsnorm(
                        &x,
                        self.dense(&format!("{p}input_layernorm.weight"))?,
                        a.rms_norm_eps,
                        &mut scratch,
                    );
                    write_f32(&x_buf, &scratch);

                    // Q/K/V read the same input and write disjoint outputs,
                    // so they share one command buffer.
                    let mut tcb = TokenCommandBuffer::new(self.ctx);
                    self.encode_matvec(&mut tcb, &format!("{p}self_attn.q_proj.weight"), &x_buf, &q_buf)?;
                    self.encode_matvec(&mut tcb, &format!("{p}self_attn.k_proj.weight"), &x_buf, &k_buf)?;
                    self.encode_matvec(&mut tcb, &format!("{p}self_attn.v_proj.weight"), &x_buf, &v_buf)?;
                    stats.dispatches += tcb.dispatch_count();
                    tcb.commit_and_wait()?;
                    stats.command_buffers += 1;

                    let mut q = read_f32(&q_buf, a.n_heads * a.head_dim);
                    let mut k = read_f32(&k_buf, kv_width);
                    let v = read_f32(&v_buf, kv_width);

                    for h in 0..a.n_heads {
                        rope_inplace_scaled(
                            &mut q[h * a.head_dim..(h + 1) * a.head_dim],
                            pos as u32,
                            a.rope_theta,
                            a.rope_scaling,
                        );
                    }
                    for h in 0..a.n_kv_heads {
                        rope_inplace_scaled(
                            &mut k[h * a.head_dim..(h + 1) * a.head_dim],
                            pos as u32,
                            a.rope_theta,
                            a.rope_scaling,
                        );
                    }
                    k_cache[layer].extend_from_slice(&k);
                    v_cache[layer].extend_from_slice(&v);
                    let seq_len = k_cache[layer].len() / kv_width;

                    let mut attn = vec![0f32; a.n_heads * a.head_dim];
                    mha_decode_step(
                        &q,
                        &k_cache[layer],
                        &v_cache[layer],
                        a.n_heads,
                        a.n_kv_heads,
                        a.head_dim,
                        seq_len,
                        &mut attn,
                    )?;

                    write_f32(&q_buf, &attn);
                    let mut tcb = TokenCommandBuffer::new(self.ctx);
                    self.encode_matvec(&mut tcb, &format!("{p}self_attn.o_proj.weight"), &q_buf, &o_buf)?;
                    stats.dispatches += tcb.dispatch_count();
                    tcb.commit_and_wait()?;
                    stats.command_buffers += 1;
                    add_inplace(&mut x, &read_f32(&o_buf, a.hidden));

                    rmsnorm(
                        &x,
                        self.dense(&format!("{p}post_attention_layernorm.weight"))?,
                        a.rms_norm_eps,
                        &mut scratch,
                    );
                    write_f32(&x_buf, &scratch);

                    // gate and up likewise read one input and write disjoint
                    // outputs.
                    let mut tcb = TokenCommandBuffer::new(self.ctx);
                    self.encode_matvec(&mut tcb, &format!("{p}mlp.gate_proj.weight"), &x_buf, &gate_buf)?;
                    self.encode_matvec(&mut tcb, &format!("{p}mlp.up_proj.weight"), &x_buf, &up_buf)?;
                    stats.dispatches += tcb.dispatch_count();
                    tcb.commit_and_wait()?;
                    stats.command_buffers += 1;

                    let gate = read_f32(&gate_buf, inter);
                    let up = read_f32(&up_buf, inter);
                    let mut act = vec![0f32; inter];
                    silu_mul(&gate, &up, &mut act);
                    write_f32(&act_buf, &act);

                    let mut tcb = TokenCommandBuffer::new(self.ctx);
                    self.encode_matvec(&mut tcb, &format!("{p}mlp.down_proj.weight"), &act_buf, &o_buf)?;
                    stats.dispatches += tcb.dispatch_count();
                    tcb.commit_and_wait()?;
                    stats.command_buffers += 1;
                    add_inplace(&mut x, &read_f32(&o_buf, a.hidden));
                }

                rmsnorm(
                    &x,
                    self.dense("model.norm.weight")?,
                    a.rms_norm_eps,
                    &mut scratch,
                );
                write_f32(&x_buf, &scratch);
                let mut tcb = TokenCommandBuffer::new(self.ctx);
                self.encode_matvec(&mut tcb, &self.head_name.clone(), &x_buf, &logits_buf)?;
                stats.dispatches += tcb.dispatch_count();
                tcb.commit_and_wait()?;
                stats.command_buffers += 1;
                logits = read_f32(&logits_buf, a.vocab_size);

                let now = Instant::now();
                stats
                    .per_token_ms
                    .push(now.duration_since(t_token).as_secs_f64() * 1e3);
                t_token = now;
                if pos == 0 {
                    stats.first_token_ms = t_start.elapsed().as_secs_f64() * 1e3;
                }
            }
            stats.total_ms = t_start.elapsed().as_secs_f64() * 1e3;
            Ok((logits, stats))
        }
    }
}
