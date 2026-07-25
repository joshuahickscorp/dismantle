//! GLM-5.2 (`glm_moe_dsa`) forward pass executed directly out of `.gravity`.
//!
//! The production counterpart of `tools/condense/glm52_reference.py`, and
//! graded against it. Everything here that looks like an odd choice is a
//! choice the reference makes, and the two agree or this is wrong:
//!
//!   - RoPE is *interleaved*, and the rotated halves are **concatenated**,
//!     not scattered back to their even and odd source positions.
//!   - The router selects on sigmoid scores **plus** a correction bias, then
//!     weights with the **uncorrected** scores (`noaux_tc`).
//!   - Expert groups are chosen by the sum of each group's best two
//!     corrected scores, and only experts inside the chosen groups compete.
//!   - An IndexShare layer **reuses** the previous full layer's DSA
//!     selection. Recomputing it there is a different model.
//!   - Indexer scores are ReLU'd before the per-head weighted sum.
//!
//! Ties are broken the way `np.argsort(kind="stable")` breaks them -- lower
//! index first -- because a differently-broken tie silently selects a
//! different expert or a different key, and nothing downstream would say so.

use std::path::Path;

use crate::gravity::GravityWeights;
use crate::{Error, Result};

/// What the forward pass needs from wherever weights actually live: CPU
/// decode-on-call (`GravityWeights`) or a GPU-resident lazy cache
/// ([`gpu::GpuWeightCache`]). The orchestration below — MLA, DSA, the
/// router, MoE dispatch — is the part that took real work to get bit-exact
/// against the oracle; it is written once against this trait and both
/// backends run the identical logic, rather than risking two copies
/// drifting apart.
pub trait WeightAccess {
    fn dense(&self, name: &str) -> Result<Vec<f32>>;
    fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>>;
    fn row(&self, name: &str, index: usize, cols: usize) -> Result<Vec<f32>>;

    /// `calls.len()` independent matvecs, dispatched together where the
    /// backend can. The default just loops `matvec` -- correct for any
    /// backend, but only the GPU one overrides it to batch every call into
    /// one command buffer instead of paying a synchronous round trip per
    /// matvec. That round trip is what a routed MoE layer's experts are
    /// dominated by: each is independent of the others, so nothing about
    /// correctness requires paying for eight of them one at a time.
    fn matvec_batch(&self, calls: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>> {
        calls.iter().map(|&(name, x)| self.matvec(name, x)).collect()
    }
}

impl WeightAccess for GravityWeights {
    fn dense(&self, name: &str) -> Result<Vec<f32>> {
        GravityWeights::dense(self, name)
    }
    fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
        GravityWeights::matvec(self, name, x)
    }
    fn row(&self, name: &str, index: usize, cols: usize) -> Result<Vec<f32>> {
        GravityWeights::row(self, name, index, cols)
    }
}

fn cfg_u64(v: &serde_json::Value, key: &str) -> Result<u64> {
    v.get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| Error::Gravity(format!("architecture.{key} missing or not an integer")))
}

fn cfg_f64(v: &serde_json::Value, key: &str) -> Result<f64> {
    v.get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| Error::Gravity(format!("architecture.{key} missing or not a number")))
}

fn cfg_strings(v: &serde_json::Value, key: &str) -> Result<Vec<String>> {
    v.get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::Gravity(format!("architecture.{key} missing or not an array")))?
        .iter()
        .map(|e| {
            e.as_str()
                .map(str::to_string)
                .ok_or_else(|| Error::Gravity(format!("architecture.{key} has a non-string entry")))
        })
        .collect()
}

/// The GLM configuration the forward pass needs. Every field is read from
/// the artifact header; none is defaulted, because a guessed `index_topk`
/// or `topk_group` produces a model that runs and is wrong.
#[derive(Debug, Clone)]
pub struct GlmArch {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub n_routed_experts: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Per layer: `"full"` computes a DSA index, `"shared"` reuses the
    /// previous full layer's.
    pub indexer_types: Vec<String>,
    /// Per layer: `"dense"` or `"sparse"`.
    pub mlp_layer_types: Vec<String>,
}

impl GlmArch {
    pub fn from_header(extra: &serde_json::Value) -> Result<GlmArch> {
        let a = extra
            .get("architecture")
            .ok_or_else(|| Error::Gravity("shard header has no `architecture`".into()))?;
        let model_type = a.get("model_type").and_then(serde_json::Value::as_str);
        if model_type != Some("glm_moe_dsa") {
            return Err(Error::Gravity(format!(
                "gravity_glm: architecture.model_type is {model_type:?}, expected \"glm_moe_dsa\""
            )));
        }
        let rope_theta = a
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .and_then(serde_json::Value::as_f64)
            .or_else(|| a.get("rope_theta").and_then(serde_json::Value::as_f64))
            .ok_or_else(|| Error::Gravity("architecture rope_theta missing".into()))?;

        let arch = GlmArch {
            n_layers: cfg_u64(a, "num_hidden_layers")? as usize,
            hidden: cfg_u64(a, "hidden_size")? as usize,
            n_heads: cfg_u64(a, "num_attention_heads")? as usize,
            q_lora_rank: cfg_u64(a, "q_lora_rank")? as usize,
            kv_lora_rank: cfg_u64(a, "kv_lora_rank")? as usize,
            qk_nope_head_dim: cfg_u64(a, "qk_nope_head_dim")? as usize,
            qk_rope_head_dim: cfg_u64(a, "qk_rope_head_dim")? as usize,
            v_head_dim: cfg_u64(a, "v_head_dim")? as usize,
            index_n_heads: cfg_u64(a, "index_n_heads")? as usize,
            index_head_dim: cfg_u64(a, "index_head_dim")? as usize,
            index_topk: cfg_u64(a, "index_topk")? as usize,
            n_routed_experts: cfg_u64(a, "n_routed_experts")? as usize,
            n_group: cfg_u64(a, "n_group")? as usize,
            topk_group: cfg_u64(a, "topk_group")? as usize,
            num_experts_per_tok: cfg_u64(a, "num_experts_per_tok")? as usize,
            norm_topk_prob: a
                .get("norm_topk_prob")
                .and_then(serde_json::Value::as_bool)
                .ok_or_else(|| Error::Gravity("architecture.norm_topk_prob missing".into()))?,
            routed_scaling_factor: cfg_f64(a, "routed_scaling_factor")? as f32,
            vocab_size: cfg_u64(a, "vocab_size")? as usize,
            rms_norm_eps: cfg_f64(a, "rms_norm_eps")? as f32,
            rope_theta: rope_theta as f32,
            indexer_types: cfg_strings(a, "indexer_types")?,
            mlp_layer_types: cfg_strings(a, "mlp_layer_types")?,
        };
        if arch.indexer_types.len() != arch.n_layers || arch.mlp_layer_types.len() != arch.n_layers {
            return Err(Error::Gravity(format!(
                "layer schedules are {} / {} long but the model has {} layers",
                arch.indexer_types.len(),
                arch.mlp_layer_types.len(),
                arch.n_layers
            )));
        }
        if arch.indexer_types.first().map(String::as_str) == Some("shared") {
            return Err(Error::Gravity(
                "layer 0 is an IndexShare layer with no previous index to share".into(),
            ));
        }
        if arch.n_group == 0 || arch.n_routed_experts % arch.n_group != 0 {
            return Err(Error::Gravity(format!(
                "{} routed experts do not divide into {} groups",
                arch.n_routed_experts, arch.n_group
            )));
        }
        Ok(arch)
    }

    pub fn qk_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

fn rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * inv * w).collect()
}

/// Affine LayerNorm, used only by the DSA indexer's key projection.
fn layernorm(x: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    (0..x.len())
        .map(|i| (x[i] - mean) * inv * weight[i] + bias[i])
        .collect()
}

fn silu_mul(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
        .collect()
}

/// Descending top-k with `np.argsort(kind="stable")` tie-breaking: equal
/// values keep ascending index order. NaN never wins, matching the way the
/// reference's `-inf` masking removes a candidate entirely.
fn topk_desc(values: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

/// GLM's interleaved RoPE for one `rotary_dim`-wide vector.
///
/// The trap: the rotated first and second components are **concatenated**,
/// so output element `i` is not input element `i` rotated. `cos`/`sin` are
/// `rotary_dim/2` long.
fn rope_interleaved(v: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
    let half = v.len() / 2;
    let mut out = vec![0f32; v.len()];
    for i in 0..half {
        let first = v[2 * i];
        let second = v[2 * i + 1];
        out[i] = first * cos[i] - second * sin[i];
        out[half + i] = second * cos[i] + first * sin[i];
    }
    out
}

/// Per-layer attention state. GLM caches the assembled `qk_dim` keys and
/// `v_head_dim` values per head, plus the DSA indexer's own key stream.
#[derive(Default)]
struct LayerCache {
    /// `[pos][head][qk_dim]` flattened.
    keys: Vec<f32>,
    /// `[pos][head][v_head_dim]` flattened.
    values: Vec<f32>,
    /// `[pos][index_head_dim]` flattened. Only full-indexer layers fill it.
    index_keys: Vec<f32>,
}

/// The growing per-layer caches for one generation. Held across calls by
/// whichever caller needs incremental decode ([`gpu::GravityGlmGpu`]); a
/// one-shot caller ([`GravityGlm::forward`]) makes a fresh one and throws it
/// away.
struct GlmSession {
    caches: Vec<LayerCache>,
}

impl GlmSession {
    fn new(arch: &GlmArch) -> GlmSession {
        GlmSession {
            caches: (0..arch.n_layers).map(|_| LayerCache::default()).collect(),
        }
    }

    /// Drop everything back to an empty cache, for a caller starting a new
    /// request on a model it keeps resident between requests.
    fn reset(&mut self) {
        for c in &mut self.caches {
            *c = LayerCache::default();
        }
    }
}

/// A `.gravity` shard loaded as an executable GLM-5.2 model.
pub struct GravityGlm {
    pub arch: GlmArch,
    weights: GravityWeights,
}

/// Everything one token's forward produced that a caller may want to check
/// without re-running it.
#[derive(Debug, Clone, Default)]
pub struct GlmTrace {
    /// DSA keys selected by the last layer that computed an index.
    pub final_topk: Vec<usize>,
    /// Per sparse layer, the experts the router chose for this token.
    pub expert_choices: Vec<Vec<usize>>,
}

impl GravityGlm {
    /// Open a single-shard `.gravity` artifact — the semantic fixture, which
    /// is small enough to decode eagerly.
    pub fn open(path: &Path, verify_hash: bool) -> Result<GravityGlm> {
        let weights = GravityWeights::open(path, verify_hash)?;
        let arch = GlmArch::from_header(&weights.header)?;
        Ok(GravityGlm { arch, weights })
    }

    /// Open a multi-shard flagship model — every `model-*.gravity` file
    /// under `dir`, indexed but not decoded. Only 8 of 256 experts activate
    /// per layer, so eager decode of every shard would waste ~32x the
    /// necessary work and, at this scale, exceed physical memory; see
    /// [`crate::gravity::GravityWeights::open_dir`].
    pub fn open_dir(dir: &Path, verify_hash: bool) -> Result<GravityGlm> {
        let weights = GravityWeights::open_dir(dir, verify_hash)?;
        let arch = GlmArch::from_header(&weights.header)?;
        Ok(GravityGlm { arch, weights })
    }

    /// Run `tokens` from an empty cache and return the logits after the last
    /// one, plus a trace of what the routers and the indexer chose.
    pub fn forward(&self, tokens: &[u32]) -> Result<(Vec<f32>, GlmTrace)> {
        let mut session = GlmSession::new(&self.arch);
        forward_impl(&self.weights, &self.arch, &mut session, tokens, 0)
    }
}

/// RoPE cos/sin for one position: `qk_rope_head_dim/2` of each. The
/// reference builds a `rotary_dim`-wide table by concatenating the
/// frequencies with themselves and then takes the first half, which is
/// exactly these values.
fn rope_cos_sin(arch: &GlmArch, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let rot = arch.qk_rope_head_dim;
    let half = rot / 2;
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for i in 0..half {
        // f32 throughout: the reference forces float32 here, and matching
        // it in f64 would drift from the thing being reproduced.
        let inv_freq = 1.0f32 / arch.rope_theta.powf(2.0 * i as f32 / rot as f32);
        let theta = pos as f32 * inv_freq;
        cos[i] = theta.cos();
        sin[i] = theta.sin();
    }
    (cos, sin)
}

/// The DSA indexer: which cached keys this token is allowed to attend to.
#[allow(clippy::too_many_arguments)]
fn indexer_topk(
    weights: &dyn WeightAccess,
    arch: &GlmArch,
    prefix: &str,
    hidden: &[f32],
    q_resid: &[f32],
    cache: &mut LayerCache,
    pos: usize,
    cos: &[f32],
    sin: &[f32],
) -> Result<Vec<usize>> {
    let a = arch;
    let (ih, idim, rot) = (a.index_n_heads, a.index_head_dim, a.qk_rope_head_dim);
    let idx = format!("{prefix}.indexer");

    let q = weights.matvec(&format!("{idx}.wq_b.weight"), q_resid)?;
    if q.len() != ih * idim {
        return Err(Error::Gravity(format!(
            "indexer wq_b produced {} values, expected {ih} heads * {idim}",
            q.len()
        )));
    }
    let k_raw = weights.matvec(&format!("{idx}.wk.weight"), hidden)?;
    let k = layernorm(
        &k_raw,
        &weights.dense(&format!("{idx}.k_norm.weight"))?,
        &weights.dense(&format!("{idx}.k_norm.bias"))?,
        1e-6,
    );

    // Rotate the leading `rot` dims, keep the tail unrotated.
    let mut k_full = rope_interleaved(&k[..rot], cos, sin);
    k_full.extend_from_slice(&k[rot..]);
    cache.index_keys.extend_from_slice(&k_full);
    let n_keys = cache.index_keys.len() / idim;

    let mut q_full = vec![0f32; ih * idim];
    for h in 0..ih {
        let src = &q[h * idim..(h + 1) * idim];
        let rotated = rope_interleaved(&src[..rot], cos, sin);
        q_full[h * idim..h * idim + rot].copy_from_slice(&rotated);
        q_full[h * idim + rot..(h + 1) * idim].copy_from_slice(&src[rot..]);
    }

    let head_scale = (ih as f32).powf(-0.5);
    let mut head_weights = weights.matvec(&format!("{idx}.weights_proj.weight"), hidden)?;
    for w in head_weights.iter_mut() {
        *w *= head_scale;
    }

    let dim_scale = (idim as f32).powf(-0.5);
    let mut index_scores = vec![0f32; n_keys];
    for (t, score) in index_scores.iter_mut().enumerate() {
        let key = &cache.index_keys[t * idim..(t + 1) * idim];
        let mut acc = 0f32;
        for h in 0..ih {
            let qh = &q_full[h * idim..(h + 1) * idim];
            let dot: f32 = qh.iter().zip(key).map(|(a, b)| a * b).sum();
            // ReLU before the weighted sum: a head that dislikes a key
            // contributes nothing rather than voting against it.
            acc += head_weights[h] * (dot * dim_scale).max(0.0);
        }
        *score = acc;
    }
    // Causal: this token cannot index a key from the future. Positions
    // beyond `pos` only exist during batched prefill.
    for (t, score) in index_scores.iter_mut().enumerate() {
        if t > pos {
            *score = f32::NEG_INFINITY;
        }
    }
    Ok(topk_desc(&index_scores, a.index_topk.min(n_keys)))
}

/// GLM's `noaux_tc` router: select on corrected scores inside the
/// winning expert groups, weight with the uncorrected ones.
fn router(
    weights: &dyn WeightAccess,
    arch: &GlmArch,
    prefix: &str,
    hidden: &[f32],
) -> Result<(Vec<usize>, Vec<f32>)> {
    let a = arch;
    let logits = weights.matvec(&format!("{prefix}.gate.weight"), hidden)?;
    let scores: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
    let bias = weights.dense(&format!("{prefix}.gate.e_score_correction_bias"))?;
    if bias.len() != a.n_routed_experts || scores.len() != a.n_routed_experts {
        return Err(Error::Gravity(format!(
            "router shape mismatch: {} logits, {} bias values, {} experts",
            scores.len(),
            bias.len(),
            a.n_routed_experts
        )));
    }
    let corrected: Vec<f32> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();

    // A group's strength is the sum of its best two corrected scores.
    let per_group = a.n_routed_experts / a.n_group;
    let group_scores: Vec<f32> = (0..a.n_group)
        .map(|g| {
            let slice = &corrected[g * per_group..(g + 1) * per_group];
            topk_desc(slice, 2.min(per_group))
                .iter()
                .map(|&i| slice[i])
                .sum()
        })
        .collect();
    let chosen: Vec<usize> = topk_desc(&group_scores, a.topk_group);

    // Only experts inside a chosen group compete.
    let mut choice = vec![f32::NEG_INFINITY; a.n_routed_experts];
    for &g in &chosen {
        for e in g * per_group..(g + 1) * per_group {
            choice[e] = corrected[e];
        }
    }
    let indices = topk_desc(&choice, a.num_experts_per_tok);
    let mut weights_out: Vec<f32> = indices.iter().map(|&i| scores[i]).collect();
    if a.norm_topk_prob {
        let total: f32 = weights_out.iter().sum::<f32>() + 1e-20;
        for w in weights_out.iter_mut() {
            *w /= total;
        }
    }
    for w in weights_out.iter_mut() {
        *w *= a.routed_scaling_factor;
    }
    Ok((indices, weights_out))
}

fn dense_mlp(weights: &dyn WeightAccess, prefix: &str, x: &[f32]) -> Result<Vec<f32>> {
    let gate = weights.matvec(&format!("{prefix}.gate_proj.weight"), x)?;
    let up = weights.matvec(&format!("{prefix}.up_proj.weight"), x)?;
    weights.matvec(&format!("{prefix}.down_proj.weight"), &silu_mul(&gate, &up))
}

/// `dense_mlp` for every prefix in `prefixes` against the same `x`, batched
/// into three round trips (gate, up, down) instead of three per prefix. An
/// MoE layer's experts are mutually independent -- none reads another's
/// output -- so nothing about correctness requires visiting them one at a
/// time; only `down_proj`'s input differs per expert, and even that batches,
/// since `matvec_batch` takes its own `x` per call.
fn batched_mlp(weights: &dyn WeightAccess, prefixes: &[String], x: &[f32]) -> Result<Vec<Vec<f32>>> {
    let gate_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.gate_proj.weight")).collect();
    let up_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.up_proj.weight")).collect();
    let gate_calls: Vec<(&str, &[f32])> = gate_names.iter().map(|n| (n.as_str(), x)).collect();
    let up_calls: Vec<(&str, &[f32])> = up_names.iter().map(|n| (n.as_str(), x)).collect();
    let gates = weights.matvec_batch(&gate_calls)?;
    let ups = weights.matvec_batch(&up_calls)?;

    let hidden: Vec<Vec<f32>> = gates.iter().zip(&ups).map(|(g, u)| silu_mul(g, u)).collect();
    let down_names: Vec<String> = prefixes.iter().map(|p| format!("{p}.down_proj.weight")).collect();
    let down_calls: Vec<(&str, &[f32])> = down_names
        .iter()
        .zip(&hidden)
        .map(|(n, h)| (n.as_str(), h.as_slice()))
        .collect();
    weights.matvec_batch(&down_calls)
}

fn routed_moe(
    weights: &dyn WeightAccess,
    arch: &GlmArch,
    prefix: &str,
    x: &[f32],
) -> Result<(Vec<f32>, Vec<usize>)> {
    let (indices, moe_weights) = router(weights, arch, prefix, x)?;
    // The reference accumulates in ascending expert order, so match it:
    // float addition is not associative and the artifact is graded to
    // 1e-4, not to "close enough". Batching changes when a matvec's bytes
    // cross the bus, never the order results are summed in below.
    let mut order: Vec<usize> = (0..indices.len()).collect();
    order.sort_by_key(|&s| indices[s]);

    let prefixes: Vec<String> = order
        .iter()
        .map(|&slot| format!("{prefix}.experts.{}", indices[slot]))
        .chain(std::iter::once(format!("{prefix}.shared_experts")))
        .collect();
    let mut outs = batched_mlp(weights, &prefixes, x)?;
    let shared = outs.pop().expect("prefixes has the shared expert last");

    let mut routed = vec![0f32; x.len()];
    for (out, &slot) in outs.iter().zip(&order) {
        for (r, o) in routed.iter_mut().zip(out) {
            *r += o * moe_weights[slot];
        }
    }
    for (r, s) in routed.iter_mut().zip(&shared) {
        *r += s;
    }
    Ok((routed, indices))
}

/// Run `tokens` starting at `start_pos` against `session`'s cache and return
/// the logits after the last one, plus a trace of what the routers and the
/// indexer chose. Shared by [`GravityGlm::forward`] (CPU, always starts a
/// fresh session at position 0) and the GPU-resident path, which reuses one
/// session across a whole generation so incremental decode never repeats
/// work a previous call already did.
fn forward_impl(
    weights: &dyn WeightAccess,
    arch: &GlmArch,
    session: &mut GlmSession,
    tokens: &[u32],
    start_pos: usize,
) -> Result<(Vec<f32>, GlmTrace)> {
    if tokens.is_empty() {
        return Err(Error::Gravity("forward: no tokens".into()));
    }
    let a = arch;
    let qk_dim = a.qk_dim();
    let mut logits = Vec::new();
    let mut trace = GlmTrace::default();

    for (i, &token) in tokens.iter().enumerate() {
        let pos = start_pos + i;
        if token as usize >= a.vocab_size {
            return Err(Error::Gravity(format!(
                "token {token} out of range for vocab_size {}",
                a.vocab_size
            )));
        }
        let mut x = weights.row("model.embed_tokens.weight", token as usize, a.hidden)?;
        let (cos, sin) = rope_cos_sin(arch, pos);
        let mut shared_topk: Option<Vec<usize>> = None;
        trace.expert_choices.clear();

        for layer in 0..a.n_layers {
            let p = format!("model.layers.{layer}");
            let attn_p = format!("{p}.self_attn");
            let h = rmsnorm(
                &x,
                &weights.dense(&format!("{p}.input_layernorm.weight"))?,
                a.rms_norm_eps,
            );

            // Queries through the low-rank path.
            let q_a = weights.matvec(&format!("{attn_p}.q_a_proj.weight"), &h)?;
            let q_resid = rmsnorm(
                &q_a,
                &weights.dense(&format!("{attn_p}.q_a_layernorm.weight"))?,
                a.rms_norm_eps,
            );
            let q = weights.matvec(&format!("{attn_p}.q_b_proj.weight"), &q_resid)?;

            // Keys and values through the compressed MLA latent. The
            // rope part is shared across heads (MQA on that slice).
            let compressed = weights.matvec(&format!("{attn_p}.kv_a_proj_with_mqa.weight"), &h)?;
            let k_latent = rmsnorm(
                &compressed[..a.kv_lora_rank],
                &weights.dense(&format!("{attn_p}.kv_a_layernorm.weight"))?,
                a.rms_norm_eps,
            );
            let k_rot = rope_interleaved(&compressed[a.kv_lora_rank..], &cos, &sin);
            let kv = weights.matvec(&format!("{attn_p}.kv_b_proj.weight"), &k_latent)?;

            let per_head_kv = a.qk_nope_head_dim + a.v_head_dim;
            let cache = &mut session.caches[layer];
            for head in 0..a.n_heads {
                let src = &kv[head * per_head_kv..(head + 1) * per_head_kv];
                cache.keys.extend_from_slice(&src[..a.qk_nope_head_dim]);
                cache.keys.extend_from_slice(&k_rot);
                cache.values.extend_from_slice(&src[a.qk_nope_head_dim..]);
            }

            // Assemble this token's queries: unrotated nope half, then
            // the interleaved-rotated rope half.
            let mut queries = vec![0f32; a.n_heads * qk_dim];
            for head in 0..a.n_heads {
                let src = &q[head * qk_dim..(head + 1) * qk_dim];
                let dst = &mut queries[head * qk_dim..(head + 1) * qk_dim];
                dst[..a.qk_nope_head_dim].copy_from_slice(&src[..a.qk_nope_head_dim]);
                dst[a.qk_nope_head_dim..]
                    .copy_from_slice(&rope_interleaved(&src[a.qk_nope_head_dim..], &cos, &sin));
            }

            let topk = match a.indexer_types[layer].as_str() {
                "full" => {
                    let t = indexer_topk(weights, arch, &attn_p, &h, &q_resid, cache, pos, &cos, &sin)?;
                    shared_topk = Some(t.clone());
                    t
                }
                "shared" => shared_topk.clone().ok_or_else(|| {
                    Error::Gravity(format!(
                        "layer {layer} shares an index but no earlier layer computed one"
                    ))
                })?,
                other => {
                    return Err(Error::Gravity(format!(
                        "layer {layer}: unknown indexer type {other:?}"
                    )))
                }
            };

            // Attend only to selected keys at or before this position.
            let n_keys = cache.keys.len() / (a.n_heads * qk_dim);
            let mut allow = vec![false; n_keys];
            for &t in &topk {
                if t <= pos && t < n_keys {
                    allow[t] = true;
                }
            }

            let scale = (qk_dim as f32).powf(-0.5);
            let mut context = vec![0f32; a.n_heads * a.v_head_dim];
            let mut scores = vec![f32::NEG_INFINITY; n_keys];
            for head in 0..a.n_heads {
                let qh = &queries[head * qk_dim..(head + 1) * qk_dim];
                let mut best = f32::NEG_INFINITY;
                for t in 0..n_keys {
                    if !allow[t] {
                        scores[t] = f32::NEG_INFINITY;
                        continue;
                    }
                    let off = (t * a.n_heads + head) * qk_dim;
                    let dot: f32 = qh
                        .iter()
                        .zip(&cache.keys[off..off + qk_dim])
                        .map(|(a, b)| a * b)
                        .sum();
                    scores[t] = dot * scale;
                    best = best.max(scores[t]);
                }
                let mut total = 0f32;
                for s in scores.iter_mut() {
                    *s = if s.is_finite() { (*s - best).exp() } else { 0.0 };
                    total += *s;
                }
                let out = &mut context[head * a.v_head_dim..(head + 1) * a.v_head_dim];
                for (t, &prob) in scores.iter().enumerate() {
                    if prob == 0.0 {
                        continue;
                    }
                    let w = prob / total;
                    let off = (t * a.n_heads + head) * a.v_head_dim;
                    for (o, v) in out.iter_mut().zip(&cache.values[off..off + a.v_head_dim]) {
                        *o += w * v;
                    }
                }
            }

            let attn_out = weights.matvec(&format!("{attn_p}.o_proj.weight"), &context)?;
            for (xv, o) in x.iter_mut().zip(&attn_out) {
                *xv += o;
            }

            let h2 = rmsnorm(
                &x,
                &weights.dense(&format!("{p}.post_attention_layernorm.weight"))?,
                a.rms_norm_eps,
            );
            let mlp_out = match a.mlp_layer_types[layer].as_str() {
                "dense" => dense_mlp(weights, &format!("{p}.mlp"), &h2)?,
                "sparse" => {
                    let (out, experts) = routed_moe(weights, arch, &format!("{p}.mlp"), &h2)?;
                    trace.expert_choices.push(experts);
                    out
                }
                other => {
                    return Err(Error::Gravity(format!(
                        "layer {layer}: unknown MLP type {other:?}"
                    )))
                }
            };
            for (xv, m) in x.iter_mut().zip(&mlp_out) {
                *xv += m;
            }

            if layer + 1 == a.n_layers {
                trace.final_topk = topk;
            }
        }

        let final_hidden = rmsnorm(&x, &weights.dense("model.norm.weight")?, a.rms_norm_eps);
        logits = weights.matvec("lm_head.weight", &final_hidden)?;
    }
    Ok((logits, trace))
}

/// GPU-resident execution: the same [`forward_impl`] orchestration above,
/// against weights lazily uploaded to Metal buffers instead of decoded on
/// the CPU on every call.
#[cfg(target_os = "macos")]
pub mod gpu {
    use super::*;
    use crate::gravity::{matvec_dense, parse_pq_header, pq_sections, widen_native, PqHeader};
    use crate::metal::{MetalContext, TokenCommandBuffer};
    use metal::Buffer;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Mirror of `GravityPQParams` in `shaders/gravity_pq.metal`: eight
    /// `uint`s in declaration order, `#[repr(C)]` so a pointer cast is a
    /// valid `set_bytes` payload. The same layout as
    /// `gravity_llama::gpu`'s copy -- both dispatch the same kernel -- kept
    /// as a second small struct rather than a shared one neither
    /// architecture module privately owns.
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

    /// One tensor resident on the device, or -- for the natively-carried
    /// ones (`lm_head.weight` in the frozen General artifact) -- widened
    /// once and left on the host. A matvec against the vocabulary runs once
    /// per token, not once per layer, so it is not worth a dedicated dense
    /// kernel the way the 30-odd per-layer projections are.
    enum GpuTensor {
        Pq {
            codebooks: Buffer,
            codes: Buffer,
            params: PqParams,
        },
        NativeCpu(Vec<f32>),
    }

    /// A [`WeightAccess`] backend that uploads each `gravity-pq` tensor to
    /// the device the first time it is asked for and keeps it there for the
    /// life of the model. Correct for GLM's MoE sparsity the same way the
    /// CPU `Lazy` source is: a short run touches 8 of 256 experts per layer,
    /// so most of the artifact is never uploaded at all, and nothing pays
    /// twice for an expert two different tokens both happen to route to.
    pub struct GpuWeightCache {
        ctx: MetalContext,
        weights: GravityWeights,
        cache: Mutex<HashMap<String, GpuTensor>>,
    }

    impl GpuWeightCache {
        fn ensure(&self, name: &str) -> Result<()> {
            if self
                .cache
                .lock()
                .expect("gpu weight cache mutex")
                .contains_key(name)
            {
                return Ok(());
            }
            let (codec, blob) = self.weights.raw_payload(name)?;
            let entry = if codec == "gravity-pq" {
                let h = parse_pq_header(&blob)?;
                if h.rotate != 0 {
                    return Err(Error::Gravity(format!(
                        "tensor {name}: rotated gravity-pq artifacts (rotate=1) are not yet \
                         supported on the GPU path"
                    )));
                }
                let (cb, codes) = pq_sections(&blob)?;
                // Four bytes of tail padding so the kernel's whole-word read
                // at the last index's byte offset stays in bounds.
                let mut codes_padded = Vec::with_capacity(codes.len() + 4);
                codes_padded.extend_from_slice(codes);
                codes_padded.extend_from_slice(&[0u8; 4]);
                GpuTensor::Pq {
                    codebooks: self.ctx.new_buffer_with_bytes_checked(cb)?,
                    codes: self.ctx.new_buffer_with_bytes_checked(&codes_padded)?,
                    params: PqParams::from_header(&h),
                }
            } else if codec.starts_with("native.") {
                GpuTensor::NativeCpu(widen_native(&codec, &blob)?)
            } else {
                return Err(Error::Gravity(format!(
                    "tensor {name}: unsupported codec {codec:?}"
                )));
            };
            self.cache
                .lock()
                .expect("gpu weight cache mutex")
                .insert(name.to_string(), entry);
            Ok(())
        }
    }

    impl WeightAccess for GpuWeightCache {
        // Norm weights and biases: small, natively carried, touched every
        // layer -- decoding them on the CPU each call is cheaper than the
        // round trip a GPU read-back would cost.
        fn dense(&self, name: &str) -> Result<Vec<f32>> {
            self.weights.dense(name)
        }

        // The embedding table's row lookup: one row, once per token. Also
        // not worth a device-resident path.
        fn row(&self, name: &str, index: usize, cols: usize) -> Result<Vec<f32>> {
            self.weights.row(name, index, cols)
        }

        fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>> {
            self.ensure(name)?;
            let cache = self.cache.lock().expect("gpu weight cache mutex");
            match cache.get(name).expect("ensure just inserted it") {
                GpuTensor::NativeCpu(w) => matvec_dense(w, x, name),
                GpuTensor::Pq {
                    codebooks,
                    codes,
                    params,
                } => dispatch_pq_matvec(&self.ctx, codebooks, codes, *params, x),
            }
        }

        /// The batching that makes an MoE layer affordable: every call this
        /// token's routed experts need goes into one command buffer instead
        /// of one synchronous round trip apiece. A command buffer's fixed
        /// submission/wait cost is what a straight per-matvec `dispatch`
        /// pays 8-9x over for a layer's worth of experts, regardless of how
        /// cheap the kernel itself is.
        fn matvec_batch(&self, calls: &[(&str, &[f32])]) -> Result<Vec<Vec<f32>>> {
            for &(name, _) in calls {
                self.ensure(name)?;
            }
            let cache = self.cache.lock().expect("gpu weight cache mutex");

            let mut results: Vec<Option<Vec<f32>>> = vec![None; calls.len()];
            let mut gpu_calls: Vec<(usize, &Buffer, &Buffer, PqParams, &[f32])> = Vec::new();
            for (i, &(name, x)) in calls.iter().enumerate() {
                match cache.get(name).expect("ensure just inserted it") {
                    GpuTensor::NativeCpu(w) => results[i] = Some(matvec_dense(w, x, name)?),
                    GpuTensor::Pq {
                        codebooks,
                        codes,
                        params,
                    } => gpu_calls.push((i, codebooks, codes, *params, x)),
                }
            }

            if !gpu_calls.is_empty() {
                let pq_calls: Vec<(&Buffer, &Buffer, PqParams, &[f32])> = gpu_calls
                    .iter()
                    .map(|&(_, cb, co, params, x)| (cb, co, params, x))
                    .collect();
                let outs = dispatch_pq_matvec_batch(&self.ctx, &pq_calls)?;
                for (&(i, ..), y) in gpu_calls.iter().zip(outs) {
                    results[i] = Some(y);
                }
            }

            results
                .into_iter()
                .enumerate()
                .map(|(i, r)| {
                    r.ok_or_else(|| Error::Gravity(format!("matvec_batch: no result for call {i}")))
                })
                .collect()
        }
    }

    /// One `gravity_pq_matvec` dispatch against already-resident codebooks
    /// and codes: upload `x`, run, read `y` back. Same kernel and launch
    /// shape as [`crate::gravity::pq_matvec_metal`], which additionally
    /// uploads the codebooks and codes on every call -- right for a parity
    /// test against a single payload, wasteful for a weight a token
    /// revisits every layer.
    fn dispatch_pq_matvec(
        ctx: &MetalContext,
        codebooks: &Buffer,
        codes: &Buffer,
        params: PqParams,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        if x.len() != params.cols as usize {
            return Err(Error::Gravity(format!(
                "gpu matvec: x.len() {} != cols {}",
                x.len(),
                params.cols
            )));
        }
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let y_buf = ctx.new_buffer(params.rows as usize * std::mem::size_of::<f32>());

        // One SIMD group (32 lanes) per output row, 8 SIMD groups (256
        // threads) per threadgroup; the kernel guards `row >= rows` for the
        // boundary threadgroup.
        const TG: u32 = 256;
        let n_tg = params.rows.div_ceil(8);
        ctx.dispatch_threads("gravity_pq_matvec", (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(codebooks), 0);
            enc.set_buffer(1, Some(codes), 0);
            enc.set_buffer(2, Some(&x_buf), 0);
            enc.set_buffer(3, Some(&y_buf), 0);
            enc.set_bytes(
                4,
                std::mem::size_of::<PqParams>() as u64,
                &params as *const PqParams as *const _,
            );
        })?;

        let y_ptr = y_buf.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(y_ptr, params.rows as usize) }.to_vec())
    }

    /// `calls.len()` independent `gravity_pq_matvec` dispatches into one
    /// [`TokenCommandBuffer`], one `commit_and_wait` for the whole batch.
    /// Every call gets its own `x`/`y` buffer pair -- `down_proj`'s calls
    /// each take a different expert's `silu_mul(gate, up)`, so a shared `x`
    /// would be wrong, not just less general.
    fn dispatch_pq_matvec_batch(
        ctx: &MetalContext,
        calls: &[(&Buffer, &Buffer, PqParams, &[f32])],
    ) -> Result<Vec<Vec<f32>>> {
        if calls.is_empty() {
            return Ok(Vec::new());
        }
        let mut x_bufs = Vec::with_capacity(calls.len());
        let mut y_bufs = Vec::with_capacity(calls.len());
        for &(_, _, params, x) in calls {
            if x.len() != params.cols as usize {
                return Err(Error::Gravity(format!(
                    "gpu matvec_batch: x.len() {} != cols {}",
                    x.len(),
                    params.cols
                )));
            }
            x_bufs.push(ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x)));
            y_bufs.push(ctx.new_buffer(params.rows as usize * std::mem::size_of::<f32>()));
        }

        let mut tcb = TokenCommandBuffer::new(ctx);
        const TG: u32 = 256;
        for (i, &(codebooks, codes, params, _)) in calls.iter().enumerate() {
            let n_tg = params.rows.div_ceil(8);
            tcb.dispatch_threads("gravity_pq_matvec", (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(codebooks), 0);
                enc.set_buffer(1, Some(codes), 0);
                enc.set_buffer(2, Some(&x_bufs[i]), 0);
                enc.set_buffer(3, Some(&y_bufs[i]), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of::<PqParams>() as u64,
                    &params as *const PqParams as *const _,
                );
            })?;
        }
        tcb.commit_and_wait()?;

        Ok(calls
            .iter()
            .zip(&y_bufs)
            .map(|(&(_, _, params, _), y_buf)| {
                let y_ptr = y_buf.contents() as *const f32;
                unsafe { std::slice::from_raw_parts(y_ptr, params.rows as usize) }.to_vec()
            })
            .collect())
    }

    /// A `.gravity` GLM-5.2 model with weights lazily resident on the GPU.
    /// Runs the identical orchestration [`GravityGlm`] does (same
    /// [`forward_impl`]); the only difference is where a matvec's bytes
    /// live and who reads them.
    pub struct GravityGlmGpu {
        pub arch: GlmArch,
        weights: GpuWeightCache,
        session: Mutex<GlmSession>,
    }

    impl GravityGlmGpu {
        /// Open with a context this model owns. An `Engine` must be `Send +
        /// Sync`, and a borrowed context makes that impossible to express,
        /// so the model holds its own -- same reasoning as
        /// [`crate::gravity_llama::gpu::GravityLlamaGpu`].
        pub fn open_dir(dir: &Path, verify_hash: bool) -> Result<GravityGlmGpu> {
            Self::open_dir_with(MetalContext::new()?, dir, verify_hash)
        }

        pub fn open_dir_with(
            ctx: MetalContext,
            dir: &Path,
            verify_hash: bool,
        ) -> Result<GravityGlmGpu> {
            let weights = GravityWeights::open_dir(dir, verify_hash)?;
            let arch = GlmArch::from_header(&weights.header)?;
            let session = Mutex::new(GlmSession::new(&arch));
            Ok(GravityGlmGpu {
                weights: GpuWeightCache {
                    ctx,
                    weights,
                    cache: Mutex::new(HashMap::new()),
                },
                arch,
                session,
            })
        }

        /// Run `tokens` from an empty cache -- the start of a new request on
        /// a model kept resident across many of them.
        pub fn forward(&self, tokens: &[u32]) -> Result<(Vec<f32>, GlmTrace)> {
            let mut session = self.session.lock().expect("glm session mutex");
            session.reset();
            forward_impl(&self.weights, &self.arch, &mut session, tokens, 0)
        }

        /// Continue the current request's cache from `start_pos`: decode,
        /// one new token against whatever `forward` or a previous
        /// `forward_at` already built.
        pub fn forward_at(
            &self,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<(Vec<f32>, GlmTrace)> {
            let mut session = self.session.lock().expect("glm session mutex");
            forward_impl(&self.weights, &self.arch, &mut session, tokens, start_pos)
        }
    }
}

/// `GravityGlmGpu` must be `Send + Sync` to be served behind the `Engine`
/// trait. This fails to compile the moment that stops being true, which is
/// the only way to notice: nothing else in the crate would.
#[cfg(all(test, target_os = "macos"))]
mod gpu_bounds {
    fn _assert_send_sync<T: Send + Sync>() {}
    #[test]
    fn gravity_glm_gpu_is_send_and_sync() {
        _assert_send_sync::<super::gpu::GravityGlmGpu>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference's own selfcheck pins position zero to `[x0,x2,x1,x3]`
    /// for a 4-wide vector -- the concatenated layout, not the scattered
    /// one. If this passes with a scatter implementation, the test is wrong.
    #[test]
    fn interleaved_rope_position_zero_is_the_concatenated_layout() {
        let v = [0f32, 1.0, 2.0, 3.0];
        let got = rope_interleaved(&v, &[1.0, 1.0], &[0.0, 0.0]);
        assert_eq!(got, vec![0.0, 2.0, 1.0, 3.0]);
    }

    #[test]
    fn topk_desc_breaks_ties_toward_the_lower_index() {
        assert_eq!(topk_desc(&[1.0, 3.0, 3.0, 0.0], 2), vec![1, 2]);
        assert_eq!(topk_desc(&[f32::NEG_INFINITY, 0.5], 1), vec![1]);
    }
}
