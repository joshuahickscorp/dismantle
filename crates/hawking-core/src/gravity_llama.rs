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
