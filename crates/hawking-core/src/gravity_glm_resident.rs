//! GPU-resident decode state for GLM-5.2.
//!
//! Prerequisite lane for Temporal Gravity command-buffer collapse. The host
//! path keeps the residual stream and per-layer KV / DSA caches as host
//! `Vec<f32>`, so every projection must finish and return a host vector before
//! the next host loop can run (~1,171 `commit_and_wait`s per flagship token).
//!
//! This module keeps those tensors in device (Metal shared) buffers across a
//! token: activations, KV, index keys, router logits / top-k / expert offsets.
//! Discrete decisions (stable top-k, noaux_tc groups, sparse softmax) still use
//! the same host arithmetic as [`crate::gravity_glm::forward_impl`] so token
//! identity is bit-exact against the host-state path; they read device-mapped
//! memory in place rather than owning a separate host cache. Projection outputs
//! are written straight into those buffers and are not copied into host `Vec`s
//! as the cache of record.
//!
//! `lm_head` is once per token. Default: host dense or PQ via
//! [`GpuWeightCache::matvec`]. With [`crate::gravity_glm::GPU_LM_HEAD_ENV`]=1 and
//! a `native.bf16` head, the weight stays device-resident and the projection
//! + greedy argmax + top-k diagnostics run on GPU (no per-token host widen of
//! the 1.90 GB table). Default readback is **token + top-k only**; full logits
//! require `HAWKING_GLM_GPU_LM_HEAD_FULL_LOGITS=1`. The same flag keeps other
//! rank-2 `native.bf16` matvecs (indexer, router) as device bf16.
//!
//! **Expert-wave** (`HAWKING_GLM_GPU_EXPERT_WAVE=1`, default off): opt-in collapse
//! of each MLP layer to one command buffer (`gate + up → SiLU → down` and MoE
//! weighted combine). The default three-`matvec_batch` path is unchanged when
//! the flag is unset (Parity V2.1 item 6).
//!
//! Gated by [`GPU_RESIDENT_STATE_ENV`] (`HAWKING_GLM_GPU_RESIDENT_STATE`), default
//! off, so the host-state path remains the parity oracle.

#![cfg(target_os = "macos")]

use crate::gravity::matvec_dense;
use crate::gravity_glm::gpu::{
    encode_argmax_f32, encode_gemv_native_bf16_seq, encode_sample_topk_f32, GpuTensor,
    GpuWeightCache,
};
use crate::gravity_glm::{
    gpu_expert_wave_enabled, gpu_lm_head_full_logits_enabled, rope_cos_sin, rope_interleaved,
    topk_desc, GlmArch, GlmTrace, WeightAccess, GPU_LM_HEAD_DIAG_TOPK,
    RESIDENT_RUNTIME_INITIAL_KV_CAPACITY_TOKENS,
};
use crate::metal::{MetalContext, TokenCommandBuffer};
use crate::{Error, Result};
use metal::Buffer;
use std::cell::Cell;
use std::sync::Mutex;

// Flag + static wait estimators live on `gravity_glm` so non-Metal unit tests
// can see them: `GPU_RESIDENT_STATE_ENV`, `gpu_resident_state_enabled`,
// `estimate_host_state_waits_per_token`, `estimate_resident_waits_per_token`.

fn write_f32(buf: &Buffer, src: &[f32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), buf.contents() as *mut f32, src.len());
    }
}

fn read_f32(buf: &Buffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n).to_vec() }
}

fn read_u32(buf: &Buffer, n: usize) -> Vec<u32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const u32, n).to_vec() }
}

/// Per-layer device KV / DSA cache.
struct LayerGpuCache {
    keys: Buffer,
    values: Buffer,
    index_keys: Buffer,
    capacity: usize,
}

const MIN_SEQUENCE_CAPACITY: usize = 4;

fn checked_sequence_bytes(elements: usize, element_bytes: usize, what: &str) -> Result<usize> {
    elements.checked_mul(element_bytes).ok_or_else(|| {
        Error::Gravity(format!(
            "{what}: sequence buffer size overflow ({elements} elements x {element_bytes} bytes)"
        ))
    })
}

fn grown_sequence_capacity(current: usize, need: usize) -> Result<usize> {
    if need <= current {
        return Ok(current);
    }
    need.checked_next_power_of_two()
        .map(|cap| cap.max(8))
        .ok_or_else(|| {
            Error::Gravity(format!(
                "resident sequence capacity overflow: current={current}, need={need}"
            ))
        })
}

fn active_sequence_len(position: usize, capacity: usize, owner: &str) -> Result<usize> {
    let need = position.checked_add(1).ok_or_else(|| {
        Error::Gravity(format!(
            "{owner}: position {position} cannot be represented as a sequence length"
        ))
    })?;
    if need > capacity {
        return Err(Error::Gravity(format!(
            "{owner}: position {position} needs {need} elements, capacity is {capacity}"
        )));
    }
    Ok(need)
}

/// Host workspaces whose lengths track the resident sequence capacity.
///
/// Keeping these vectors at their reserved length means index scoring,
/// stable top-k selection, and sparse-attention masking do not allocate
/// sequence-sized temporaries after [`ResidentSession::reserve`] succeeds.
#[derive(Debug)]
struct HostSequenceScratch {
    index_scores: Vec<f32>,
    selection_indices: Vec<usize>,
    attention_allowed: Vec<u8>,
    attention_scores: Vec<f32>,
}

impl HostSequenceScratch {
    fn new(capacity: usize) -> Self {
        Self {
            index_scores: vec![0.0; capacity],
            selection_indices: vec![0; capacity],
            attention_allowed: vec![0; capacity],
            attention_scores: vec![f32::NEG_INFINITY; capacity],
        }
    }

    fn grow_preserving(&mut self, capacity: usize) {
        if capacity <= self.index_scores.len() {
            return;
        }
        self.index_scores.resize(capacity, 0.0);
        self.selection_indices.resize(capacity, 0);
        self.attention_allowed.resize(capacity, 0);
        self.attention_scores.resize(capacity, f32::NEG_INFINITY);
    }
}

/// Sequence-sized DSA/index-selection scratch owned by one resident session.
///
/// `ActPool` is model-global and fixed-size, so sequence-dependent buffers do
/// not belong there. This workspace grows in lockstep with the session's KV
/// caches and is then reused serially by every layer.
struct SequenceScratch {
    index_scores_device: Buffer,
    host: HostSequenceScratch,
    capacity: usize,
    device_score_len: usize,
}

impl SequenceScratch {
    fn new(ctx: &MetalContext, initial_cap: usize) -> Result<Self> {
        let capacity = initial_cap.max(MIN_SEQUENCE_CAPACITY);
        let bytes = checked_sequence_bytes(
            capacity,
            std::mem::size_of::<f32>(),
            "resident index scores",
        )?;
        Ok(Self {
            index_scores_device: ctx.new_buffer_checked(bytes)?,
            host: HostSequenceScratch::new(capacity),
            capacity,
            device_score_len: 0,
        })
    }

    fn reserve(&mut self, ctx: &MetalContext, need: usize) -> Result<()> {
        let capacity = grown_sequence_capacity(self.capacity, need)?;
        if capacity == self.capacity {
            return Ok(());
        }

        let bytes = checked_sequence_bytes(
            capacity,
            std::mem::size_of::<f32>(),
            "resident index scores",
        )?;
        let next = ctx.new_buffer_checked(bytes)?;
        if self.device_score_len > 0 {
            let copy_bytes = checked_sequence_bytes(
                self.device_score_len,
                std::mem::size_of::<f32>(),
                "resident index score copy",
            )?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.index_scores_device.contents() as *const u8,
                    next.contents() as *mut u8,
                    copy_bytes,
                );
            }
        }
        self.host.grow_preserving(capacity);
        self.index_scores_device = next;
        self.capacity = capacity;
        Ok(())
    }

    fn active_len(&self, position: usize) -> Result<usize> {
        active_sequence_len(position, self.capacity, "resident sequence scratch")
    }

    fn store_index_scores(&mut self, len: usize) -> Result<()> {
        if len > self.capacity || len > self.host.index_scores.len() {
            return Err(Error::Gravity(format!(
                "resident index score write needs {len} elements, capacity is {}",
                self.capacity
            )));
        }
        let bytes = checked_sequence_bytes(
            len,
            std::mem::size_of::<f32>(),
            "resident index score write",
        )?;
        if bytes as u64 > self.index_scores_device.length() {
            return Err(Error::Gravity(format!(
                "resident index score write needs {bytes} bytes, device buffer has {}",
                self.index_scores_device.length()
            )));
        }
        write_f32(&self.index_scores_device, &self.host.index_scores[..len]);
        self.device_score_len = len;
        Ok(())
    }
}

/// Reuses the session's O(sequence-length) index workspace for [`topk_desc`].
///
/// The index tie-break makes the comparator a total order even though the
/// backing sort is unstable, preserving the reference's ascending-index
/// result for equal finite scores without allocating a stable-sort merge
/// buffer. The returned O(k) result remains owned, matching the existing
/// resident-path interface.
fn topk_desc_with_scratch(
    values: &[f32],
    k: usize,
    selection_indices: &mut [usize],
) -> Result<Vec<usize>> {
    if selection_indices.len() < values.len() {
        return Err(Error::Gravity(format!(
            "resident top-k selection needs {} indices, scratch has {}",
            values.len(),
            selection_indices.len()
        )));
    }
    let indices = &mut selection_indices[..values.len()];
    for (index, slot) in indices.iter_mut().enumerate() {
        *slot = index;
    }
    indices.sort_unstable_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    Ok(indices[..k.min(indices.len())].to_vec())
}

/// Device-resident working set for one generation.
pub struct ResidentSession {
    layers: Vec<LayerGpuCache>,
    sequence_scratch: SequenceScratch,
    pub seq_len: usize,
    shared_topk: Option<Vec<usize>>,
    waits: Cell<u64>,
}

impl ResidentSession {
    pub fn new(ctx: &MetalContext, arch: &GlmArch, initial_cap: usize) -> Result<Self> {
        let cap = initial_cap.max(MIN_SEQUENCE_CAPACITY);
        let qk = arch.qk_dim();
        let mut layers = Vec::with_capacity(arch.n_layers);
        for _ in 0..arch.n_layers {
            layers.push(LayerGpuCache {
                keys: ctx.new_buffer_checked(cap * arch.n_heads * qk * 4)?,
                values: ctx.new_buffer_checked(cap * arch.n_heads * arch.v_head_dim * 4)?,
                index_keys: ctx.new_buffer_checked(cap * arch.index_head_dim * 4)?,
                capacity: cap,
            });
        }
        Ok(Self {
            layers,
            sequence_scratch: SequenceScratch::new(ctx, cap)?,
            seq_len: 0,
            shared_topk: None,
            waits: Cell::new(0),
        })
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
        self.shared_topk = None;
        self.waits.set(0);
    }

    pub fn waits(&self) -> u64 {
        self.waits.get()
    }

    fn reserve(&mut self, ctx: &MetalContext, arch: &GlmArch, need: usize) -> Result<()> {
        self.sequence_scratch.reserve(ctx, need)?;
        let cur = self.layers.first().map(|l| l.capacity).unwrap_or(0);
        if need <= cur {
            return Ok(());
        }
        let cap = grown_sequence_capacity(cur, need)?;
        let qk = arch.qk_dim();
        for layer in 0..arch.n_layers {
            let old = &self.layers[layer];
            let nk = ctx.new_buffer_checked(cap * arch.n_heads * qk * 4)?;
            let nv = ctx.new_buffer_checked(cap * arch.n_heads * arch.v_head_dim * 4)?;
            let ni = ctx.new_buffer_checked(cap * arch.index_head_dim * 4)?;
            if self.seq_len > 0 {
                let ck = self.seq_len * arch.n_heads * qk * 4;
                let cv = self.seq_len * arch.n_heads * arch.v_head_dim * 4;
                let ci = self.seq_len * arch.index_head_dim * 4;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        old.keys.contents() as *const u8,
                        nk.contents() as *mut u8,
                        ck,
                    );
                    std::ptr::copy_nonoverlapping(
                        old.values.contents() as *const u8,
                        nv.contents() as *mut u8,
                        cv,
                    );
                    std::ptr::copy_nonoverlapping(
                        old.index_keys.contents() as *const u8,
                        ni.contents() as *mut u8,
                        ci,
                    );
                }
            }
            self.layers[layer] = LayerGpuCache {
                keys: nk,
                values: nv,
                index_keys: ni,
                capacity: cap,
            };
        }
        Ok(())
    }
}

/// Activation / scratch pool (device buffers, reused every token).
pub struct ActPool {
    pub x: Buffer,
    h: Buffer,
    q_a: Buffer,
    q_resid: Buffer,
    q: Buffer,
    compressed: Buffer,
    k_latent: Buffer,
    kv: Buffer,
    queries: Buffer,
    context: Buffer,
    o: Buffer,
    // DSA / router scratch kept on device as the cache of record
    idx_q: Buffer,
    idx_k_raw: Buffer,
    idx_head_w: Buffer,
    router_logits: Buffer,
    router_scores: Buffer,
    router_corrected: Buffer,
    expert_idx: Buffer,
    expert_w: Buffer,
    // Expert scratch (sized for future device-side expert chaining; the
    // batched path currently uses matvec_batch into host Vecs for the three
    // co-issued waits that match the host oracle).
    #[allow(dead_code)]
    gate: Buffer,
    #[allow(dead_code)]
    up: Buffer,
    #[allow(dead_code)]
    act: Buffer,
    #[allow(dead_code)]
    down: Buffer,
    final_hidden: Buffer,
    /// Device logits for device-resident lm_head (vocab-sized). Stay on device;
    /// host only reads them under `HAWKING_GLM_GPU_LM_HEAD_FULL_LOGITS=1`.
    logits: Buffer,
    /// Single u32 greedy token from on-device argmax (token-only readback).
    sample_token: Buffer,
    /// Diagnostic top-k indices over device logits (`GPU_LM_HEAD_DIAG_TOPK`).
    head_topk_idx: Buffer,
    /// Diagnostic top-k values over device logits.
    head_topk_val: Buffer,
    #[allow(dead_code)]
    gate_cap: usize,
}

impl ActPool {
    pub fn new(ctx: &MetalContext, arch: &GlmArch) -> Result<Self> {
        let h = arch.hidden;
        let qk = arch.qk_dim();
        let gate_cap = (h * 32).max(4096);
        Ok(Self {
            x: ctx.new_buffer_checked(h * 4)?,
            h: ctx.new_buffer_checked(h * 4)?,
            q_a: ctx.new_buffer_checked(arch.q_lora_rank * 4)?,
            q_resid: ctx.new_buffer_checked(arch.q_lora_rank * 4)?,
            q: ctx.new_buffer_checked(arch.n_heads * qk * 4)?,
            compressed: ctx.new_buffer_checked((arch.kv_lora_rank + arch.qk_rope_head_dim) * 4)?,
            k_latent: ctx.new_buffer_checked(arch.kv_lora_rank * 4)?,
            kv: ctx
                .new_buffer_checked(arch.n_heads * (arch.qk_nope_head_dim + arch.v_head_dim) * 4)?,
            queries: ctx.new_buffer_checked(arch.n_heads * qk * 4)?,
            context: ctx.new_buffer_checked(arch.n_heads * arch.v_head_dim * 4)?,
            o: ctx.new_buffer_checked(h * 4)?,
            idx_q: ctx.new_buffer_checked(arch.index_n_heads * arch.index_head_dim * 4)?,
            idx_k_raw: ctx.new_buffer_checked(arch.index_head_dim * 4)?,
            idx_head_w: ctx.new_buffer_checked(arch.index_n_heads * 4)?,
            router_logits: ctx.new_buffer_checked(arch.n_routed_experts * 4)?,
            router_scores: ctx.new_buffer_checked(arch.n_routed_experts * 4)?,
            router_corrected: ctx.new_buffer_checked(arch.n_routed_experts * 4)?,
            expert_idx: ctx.new_buffer_checked(arch.num_experts_per_tok.max(1) * 4)?,
            expert_w: ctx.new_buffer_checked(arch.num_experts_per_tok.max(1) * 4)?,
            gate: ctx.new_buffer_checked(gate_cap * 4)?,
            up: ctx.new_buffer_checked(gate_cap * 4)?,
            act: ctx.new_buffer_checked(gate_cap * 4)?,
            down: ctx.new_buffer_checked(h * 4)?,
            final_hidden: ctx.new_buffer_checked(h * 4)?,
            logits: ctx.new_buffer_checked(arch.vocab_size * 4)?,
            sample_token: ctx.new_buffer_checked(4)?,
            head_topk_idx: ctx.new_buffer_checked((GPU_LM_HEAD_DIAG_TOPK as usize) * 4)?,
            head_topk_val: ctx.new_buffer_checked((GPU_LM_HEAD_DIAG_TOPK as usize) * 4)?,
            gate_cap,
        })
    }
}

fn commit(tcb: Option<TokenCommandBuffer<'_>>, waits: &Cell<u64>) -> Result<()> {
    if let Some(buf) = tcb {
        // When the cost ledger is recording, commit folds metal_encode /
        // metal_submit / metal_synchronize + GPU timestamps. Off path is
        // the historical uninstrumented flush (single atomic load).
        buf.commit_and_wait()?;
        waits.set(waits.get().saturating_add(1));
    }
    Ok(())
}

fn record_dense_matvec_ops(rows: u64, cols: u64) {
    let fp = rows.saturating_mul(cols).saturating_mul(2);
    crate::cost_ledger::record_source_modelled_operations(fp, 0, 0, 0, fp);
}

fn record_pq_matvec_ops(params: crate::gravity_glm::gpu::PqParams) {
    let rows = params.rows as u64;
    let dense_fp = rows.saturating_mul(params.cols as u64).saturating_mul(2);
    // Kernel source executes one FMA per logical weight plus a 32-lane
    // simd_sum (31 adds/row). pq_index's visible bit arithmetic is a
    // documented 15-op lower bound per row/chunk/subspace lookup.
    let fp = dense_fp.saturating_add(rows.saturating_mul(31));
    let lookups = rows
        .saturating_mul(params.nchunk as u64)
        .saturating_mul(params.subspaces as u64);
    crate::cost_ledger::record_source_modelled_operations(
        fp,
        lookups.saturating_mul(15),
        0,
        0,
        dense_fp,
    );
}

/// Matvec into a device buffer. Host-native weights run on the host into the
/// shared buffer (no wait). PQ and device-resident bf16 encode into `tcb` and
/// need a later commit.
fn matvec_into<'a>(
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    weights: &GpuWeightCache,
    name: &str,
    x: &Buffer,
    x_len: usize,
    y: &Buffer,
) -> Result<()> {
    crate::cost_ledger::record_matvec_call();
    let mut cache = weights.cache.lock().expect("gpu weight cache");
    weights.ensure_many_locked(&mut cache, &[name])?;
    match cache.get(name).expect("ensured") {
        GpuTensor::NativeCpu(w) => {
            // Host-oracle path (flag off): do not change default billing or
            // numerics. Active-byte category partition for native.f32 widen is
            // owned by WeightAccess::matvec when that path is used — still
            // bill here so the resident path is not a blind hole when the
            // ledger is on.
            crate::cost_ledger::record_active_bytes_for(name, (w.len() * 4) as u64);
            record_dense_matvec_ops((w.len() / x_len) as u64, x_len as u64);
            let x_host = read_f32(x, x_len);
            let y_host = matvec_dense(w, &x_host, name)?;
            write_f32(y, &y_host);
            Ok(())
        }
        GpuTensor::NativeGpuBf16 { buf, rows, cols } => {
            if x_len != *cols as usize {
                return Err(Error::Gravity(format!(
                    "resident matvec {name}: x_len {x_len} != cols {cols}"
                )));
            }
            // Bill stored bf16 length (no f32 widen tax).
            crate::cost_ledger::record_active_bytes_for(name, buf.length());
            record_dense_matvec_ops(*rows as u64, *cols as u64);
            let tcb = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
            // MetalEncode charged at TCB commit from dispatch_threads wall.
            encode_gemv_native_bf16_seq(tcb, buf, *rows, *cols, x, y)
        }
        GpuTensor::Pq {
            codebooks,
            codes,
            params,
        } => {
            if x_len != params.cols as usize {
                return Err(Error::Gravity(format!(
                    "resident matvec {name}: x_len {x_len} != cols {}",
                    params.cols
                )));
            }
            crate::cost_ledger::record_active_bytes_for(name, codebooks.length() + codes.length());
            record_pq_matvec_ops(*params);
            let tcb = tcb.get_or_insert_with(|| TokenCommandBuffer::new(ctx));
            const TG: u32 = 256;
            let n_tg = params.rows.div_ceil(8);
            let params = *params;
            let cb = codebooks.clone();
            let co = codes.clone();
            tcb.dispatch_threads("gravity_pq_matvec", (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(&cb), 0);
                enc.set_buffer(1, Some(&co), 0);
                enc.set_buffer(2, Some(x), 0);
                enc.set_buffer(3, Some(y), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
            })?;
            Ok(())
        }
    }
}

fn rmsnorm_into(x: &Buffer, x_len: usize, weight: &[f32], eps: f32, out: &Buffer) {
    let _norm = crate::cost_ledger::Scope::new(crate::cost_ledger::Bucket::Norm);
    crate::cost_ledger::record_source_modelled_operations((4 * x_len) as u64, 0, 0, 1, 0);
    let xv = read_f32(x, x_len);
    let mean_sq = xv.iter().map(|v| v * v).sum::<f32>() / x_len as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    let y: Vec<f32> = xv.iter().zip(weight).map(|(v, w)| v * inv * *w).collect();
    write_f32(out, &y);
}

fn residual_add(x: &Buffer, add: &Buffer, n: usize) {
    let _state = crate::cost_ledger::Scope::new(crate::cost_ledger::Bucket::ResidualAndState);
    crate::cost_ledger::record_source_modelled_operations(n as u64, 0, 0, 0, 0);
    let mut xv = read_f32(x, n);
    let av = read_f32(add, n);
    for (a, b) in xv.iter_mut().zip(&av) {
        *a += *b;
    }
    write_f32(x, &xv);
}

/// One generation step (or prefill) with decode state on device.
pub fn forward_resident(
    weights: &GpuWeightCache,
    arch: &GlmArch,
    session: &mut ResidentSession,
    pool: &ActPool,
    tokens: &[u32],
    start_pos: usize,
) -> Result<(Vec<f32>, GlmTrace, u64)> {
    use crate::cost_ledger::{self, Bucket};

    if tokens.is_empty() {
        return Err(Error::Gravity("forward_resident: no tokens".into()));
    }
    let ctx = &weights.ctx;
    let a = arch;
    let qk = a.qk_dim();
    let required_sequence = start_pos.checked_add(tokens.len()).ok_or_else(|| {
        Error::Gravity(format!(
            "forward_resident: sequence length overflow ({start_pos} + {})",
            tokens.len()
        ))
    })?;
    {
        let _kv = cost_ledger::Scope::new(Bucket::KvUpdate);
        session.reserve(ctx, arch, required_sequence)?;
    }
    let waits_before = session.waits.get();
    let mut logits = Vec::new();
    let mut trace = GlmTrace::default();

    // Same geometry stamp as host forward_impl so live vs geometry is comparable.
    cost_ledger::set_geometry_active_bytes(cost_ledger::geometry_active_bytes(
        a.n_layers,
        a.num_experts_per_tok,
        None,
    ));

    for (step, &token) in tokens.iter().enumerate() {
        let pos = start_pos + step;
        if token as usize >= a.vocab_size {
            return Err(Error::Gravity(format!(
                "token {token} out of range for vocab_size {}",
                a.vocab_size
            )));
        }

        {
            let _embedding = cost_ledger::Scope::new(Bucket::EmbeddingAndPosition);
            let emb = weights.row("model.embed_tokens.weight", token as usize, a.hidden)?;
            write_f32(&pool.x, &emb);
        }
        let (cos, sin) = {
            let _position = cost_ledger::Scope::new(Bucket::EmbeddingAndPosition);
            rope_cos_sin(arch, pos)
        };
        let mut shared_topk = session.shared_topk.clone();
        trace.expert_choices.clear();

        for layer in 0..a.n_layers {
            let p = format!("model.layers.{layer}");
            let attn_p = format!("{p}.self_attn");
            let mut tcb: Option<TokenCommandBuffer<'_>> = None;

            // Attention + IndexShare: projections, DSA indexer, sparse attend,
            // o_proj residual. Nested metal/norm/kv buckets steal exclusive time.
            let topk = {
                let _attn = cost_ledger::Scope::new(Bucket::AttentionAndIndexShare);

                let w_in = weights.dense(&format!("{p}.input_layernorm.weight"))?;
                rmsnorm_into(&pool.x, a.hidden, &w_in, a.rms_norm_eps, &pool.h);

                // Q path
                matvec_into(
                    &mut tcb,
                    ctx,
                    weights,
                    &format!("{attn_p}.q_a_proj.weight"),
                    &pool.h,
                    a.hidden,
                    &pool.q_a,
                )?;
                // KV-a is independent of q_a — co-issue before the wait.
                matvec_into(
                    &mut tcb,
                    ctx,
                    weights,
                    &format!("{attn_p}.kv_a_proj_with_mqa.weight"),
                    &pool.h,
                    a.hidden,
                    &pool.compressed,
                )?;
                commit(tcb.take(), &session.waits)?;

                let w_q = weights.dense(&format!("{attn_p}.q_a_layernorm.weight"))?;
                rmsnorm_into(
                    &pool.q_a,
                    a.q_lora_rank,
                    &w_q,
                    a.rms_norm_eps,
                    &pool.q_resid,
                );

                let compressed = read_f32(&pool.compressed, a.kv_lora_rank + a.qk_rope_head_dim);
                let w_kv = weights.dense(&format!("{attn_p}.kv_a_layernorm.weight"))?;
                let k_latent = {
                    let _norm = cost_ledger::Scope::new(Bucket::Norm);
                    let x = &compressed[..a.kv_lora_rank];
                    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
                    let inv = 1.0 / (mean_sq + a.rms_norm_eps).sqrt();
                    x.iter()
                        .zip(&w_kv)
                        .map(|(v, w)| v * inv * w)
                        .collect::<Vec<_>>()
                };
                write_f32(&pool.k_latent, &k_latent);
                let k_rot = rope_interleaved(&compressed[a.kv_lora_rank..], &cos, &sin);

                matvec_into(
                    &mut tcb,
                    ctx,
                    weights,
                    &format!("{attn_p}.q_b_proj.weight"),
                    &pool.q_resid,
                    a.q_lora_rank,
                    &pool.q,
                )?;
                matvec_into(
                    &mut tcb,
                    ctx,
                    weights,
                    &format!("{attn_p}.kv_b_proj.weight"),
                    &pool.k_latent,
                    a.kv_lora_rank,
                    &pool.kv,
                )?;
                commit(tcb.take(), &session.waits)?;

                // Append MLA K/V into the device cache (cache of record).
                {
                    let _kv = cost_ledger::Scope::new(Bucket::KvUpdate);
                    let kv = read_f32(&pool.kv, a.n_heads * (a.qk_nope_head_dim + a.v_head_dim));
                    let cache = &session.layers[layer];
                    let per = a.qk_nope_head_dim + a.v_head_dim;
                    let mut keys_pos = Vec::with_capacity(a.n_heads * qk);
                    let mut vals_pos = Vec::with_capacity(a.n_heads * a.v_head_dim);
                    for head in 0..a.n_heads {
                        let src = &kv[head * per..(head + 1) * per];
                        keys_pos.extend_from_slice(&src[..a.qk_nope_head_dim]);
                        keys_pos.extend_from_slice(&k_rot);
                        vals_pos.extend_from_slice(&src[a.qk_nope_head_dim..]);
                    }
                    let k_off = pos * a.n_heads * qk;
                    let v_off = pos * a.n_heads * a.v_head_dim;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            keys_pos.as_ptr(),
                            (cache.keys.contents() as *mut f32).add(k_off),
                            keys_pos.len(),
                        );
                        std::ptr::copy_nonoverlapping(
                            vals_pos.as_ptr(),
                            (cache.values.contents() as *mut f32).add(v_off),
                            vals_pos.len(),
                        );
                    }
                }

                // Queries
                {
                    let q = read_f32(&pool.q, a.n_heads * qk);
                    let mut queries = vec![0f32; a.n_heads * qk];
                    cost_ledger::record_allocation((queries.len() * 4) as u64);
                    for head in 0..a.n_heads {
                        let src = &q[head * qk..(head + 1) * qk];
                        let dst = &mut queries[head * qk..(head + 1) * qk];
                        dst[..a.qk_nope_head_dim].copy_from_slice(&src[..a.qk_nope_head_dim]);
                        dst[a.qk_nope_head_dim..].copy_from_slice(&rope_interleaved(
                            &src[a.qk_nope_head_dim..],
                            &cos,
                            &sin,
                        ));
                    }
                    write_f32(&pool.queries, &queries);
                }

                let topk = match a.indexer_types[layer].as_str() {
                    "full" => {
                        let cache = &session.layers[layer];
                        let scratch = &mut session.sequence_scratch;
                        let t = indexer_topk(
                            weights,
                            arch,
                            &attn_p,
                            pool,
                            cache,
                            scratch,
                            pos,
                            &cos,
                            &sin,
                            &mut tcb,
                            ctx,
                            &session.waits,
                        )?;
                        shared_topk = Some(t.clone());
                        session.shared_topk = Some(t.clone());
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

                // Sparse attend over device-resident KV (same math as forward_impl).
                let cache = &session.layers[layer];
                let scratch = &mut session.sequence_scratch;
                let context = sparse_attend(a, pool, cache, scratch, pos, &topk, qk)?;
                write_f32(&pool.context, &context);

                matvec_into(
                    &mut tcb,
                    ctx,
                    weights,
                    &format!("{attn_p}.o_proj.weight"),
                    &pool.context,
                    a.n_heads * a.v_head_dim,
                    &pool.o,
                )?;
                commit(tcb.take(), &session.waits)?;
                residual_add(&pool.x, &pool.o, a.hidden);
                topk
            };

            let w_post = weights.dense(&format!("{p}.post_attention_layernorm.weight"))?;
            rmsnorm_into(&pool.x, a.hidden, &w_post, a.rms_norm_eps, &pool.h);

            match a.mlp_layer_types[layer].as_str() {
                "dense" => {
                    let _dense = cost_ledger::Scope::new(Bucket::DenseExperts);
                    let prefix = format!("{p}.mlp");
                    let out = mlp_one(
                        weights,
                        &prefix,
                        &pool.h,
                        a.hidden,
                        pool,
                        &mut tcb,
                        ctx,
                        &session.waits,
                    )?;
                    write_f32(&pool.o, &out);
                    residual_add(&pool.x, &pool.o, a.hidden);
                }
                "sparse" => {
                    let prefix = format!("{p}.mlp");
                    // Router gate matvec + host select — Routing bucket (matches host).
                    {
                        let _route = cost_ledger::Scope::new(Bucket::Routing);
                        matvec_into(
                            &mut tcb,
                            ctx,
                            weights,
                            &format!("{prefix}.gate.weight"),
                            &pool.h,
                            a.hidden,
                            &pool.router_logits,
                        )?;
                        commit(tcb.take(), &session.waits)?;
                    }

                    let (indices, moe_weights) = router_select(weights, a, &prefix, pool)?;
                    // Residency: expert selection + weights live on device.
                    {
                        let _route_state = cost_ledger::Scope::new(Bucket::Routing);
                        let idx_u: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                idx_u.as_ptr(),
                                pool.expert_idx.contents() as *mut u32,
                                idx_u.len(),
                            );
                        }
                        write_f32(&pool.expert_w, &moe_weights);
                    }
                    trace.expert_choices.push(indices.clone());

                    // Ascending expert order (float-add associativity), then
                    // shared last — same as host `routed_moe` / `batched_mlp`.
                    let mut order: Vec<usize> = (0..indices.len()).collect();
                    order.sort_by_key(|&s| indices[s]);
                    let prefixes: Vec<String> = order
                        .iter()
                        .map(|&slot| format!("{prefix}.experts.{}", indices[slot]))
                        .chain(std::iter::once(format!("{prefix}.shared_experts")))
                        .collect();
                    // Expert-wave (flagged, default off): one CB for gate/up/SiLU/
                    // down/weighted combine. Default three-batch path is unchanged.
                    // RoutedExperts owns co-batch CPU glue; metal_* steals GPU waits.
                    let routed = {
                        let _routed = cost_ledger::Scope::new(Bucket::RoutedExperts);
                        if gpu_expert_wave_enabled() {
                            let scales: Vec<f32> = order
                                .iter()
                                .map(|&slot| moe_weights[slot])
                                .chain(std::iter::once(1.0f32))
                                .collect();
                            moe_device_wave(
                                weights,
                                &prefixes,
                                &scales,
                                &pool.h,
                                a.hidden,
                                &mut tcb,
                                ctx,
                                &session.waits,
                            )?
                        } else {
                            let mut outs = batched_mlp(
                                weights,
                                &prefixes,
                                &pool.h,
                                a.hidden,
                                pool,
                                &mut tcb,
                                ctx,
                                &session.waits,
                            )?;
                            let shared = outs.pop().expect("shared last");
                            let mut routed = {
                                let _r = cost_ledger::Scope::new(Bucket::RoutedExperts);
                                let mut routed = vec![0f32; a.hidden];
                                cost_ledger::record_allocation((routed.len() * 4) as u64);
                                cost_ledger::record_source_modelled_operations(
                                    (2usize
                                        .saturating_mul(routed.len())
                                        .saturating_mul(outs.len()))
                                        as u64,
                                    0,
                                    0,
                                    0,
                                    0,
                                );
                                for (out, &slot) in outs.iter().zip(&order) {
                                    for (r, o) in routed.iter_mut().zip(out) {
                                        *r += o * moe_weights[slot];
                                    }
                                }
                                routed
                            };
                            {
                                let _shared = cost_ledger::Scope::new(Bucket::SharedExperts);
                                cost_ledger::record_source_modelled_operations(
                                    routed.len() as u64,
                                    0,
                                    0,
                                    0,
                                    0,
                                );
                                for (r, s) in routed.iter_mut().zip(&shared) {
                                    *r += *s;
                                }
                            }
                            routed
                        }
                    };
                    write_f32(&pool.o, &routed);
                    residual_add(&pool.x, &pool.o, a.hidden);
                }
                other => {
                    return Err(Error::Gravity(format!(
                        "layer {layer}: unknown MLP type {other:?}"
                    )))
                }
            }

            if layer + 1 == a.n_layers {
                trace.final_topk = topk;
            }
        }

        let w_norm = weights.dense("model.norm.weight")?;
        rmsnorm_into(
            &pool.x,
            a.hidden,
            &w_norm,
            a.rms_norm_eps,
            &pool.final_hidden,
        );
        // lm_head once per token. Device-resident bf16 keeps weight + logits
        // on GPU, runs blockwise gemv + greedy argmax + top-k, and by default
        // reads back only the token + top-k diagnostics (not 154_880 logits).
        // Full logits: HAWKING_GLM_GPU_LM_HEAD_FULL_LOGITS=1. Host dense / PQ
        // keep the prior path (full logits).
        let waits_before_head = session.waits.get();
        {
            let _head = crate::cost_ledger::Scope::new(crate::cost_ledger::Bucket::FinalHead);
            let mut cache = weights.cache.lock().expect("gpu weight cache");
            weights.ensure_many_locked(&mut cache, &["lm_head.weight"])?;
            match cache.get("lm_head.weight").expect("ensured lm_head") {
                GpuTensor::NativeGpuBf16 { buf, rows, cols } => {
                    if a.hidden != *cols as usize {
                        return Err(Error::Gravity(format!(
                            "lm_head device path: hidden {} != cols {cols}",
                            a.hidden
                        )));
                    }
                    if a.vocab_size != *rows as usize {
                        return Err(Error::Gravity(format!(
                            "lm_head device path: vocab {} != rows {rows}",
                            a.vocab_size
                        )));
                    }
                    crate::cost_ledger::record_matvec_call();
                    crate::cost_ledger::record_active_bytes_for("lm_head.weight", buf.length());
                    crate::cost_ledger::record_source_modelled_operations(
                        2u64.saturating_mul(*rows as u64)
                            .saturating_mul(*cols as u64),
                        0,
                        0,
                        0,
                        2u64.saturating_mul(*rows as u64)
                            .saturating_mul(*cols as u64),
                    );
                    let mut tcb = TokenCommandBuffer::new(ctx);
                    encode_gemv_native_bf16_seq(
                        &mut tcb,
                        buf,
                        *rows,
                        *cols,
                        &pool.final_hidden,
                        &pool.logits,
                    )?;
                    {
                        let _sampling = cost_ledger::Scope::new(cost_ledger::Bucket::Sampling);
                        encode_argmax_f32(&mut tcb, &pool.logits, *rows, &pool.sample_token)?;
                        encode_sample_topk_f32(
                            &mut tcb,
                            &pool.logits,
                            *rows,
                            GPU_LM_HEAD_DIAG_TOPK,
                            &pool.head_topk_idx,
                            &pool.head_topk_val,
                        )?;
                        let rounds = GPU_LM_HEAD_DIAG_TOPK as u64 + 1;
                        cost_ledger::record_source_modelled_operations(
                            0,
                            0,
                            rounds
                                .saturating_mul(*rows as u64)
                                .saturating_add(rounds.saturating_mul(255)),
                            0,
                            0,
                        );
                    }
                    tcb.commit_and_wait()?;
                    session.waits.set(session.waits.get().saturating_add(1));

                    // Token + diagnostics only (default). Full vector is opt-in.
                    {
                        let _sampling = cost_ledger::Scope::new(cost_ledger::Bucket::Sampling);
                        let tok = read_u32(&pool.sample_token, 1)[0];
                        let k = GPU_LM_HEAD_DIAG_TOPK as usize;
                        let topk_idx = read_u32(&pool.head_topk_idx, k);
                        let topk_val = read_f32(&pool.head_topk_val, k);
                        crate::cost_ledger::record_transfer(
                            (4 + k * 4 + k * 4) as u64,
                            false,
                            "lm_head_token_diag_download",
                        );
                        trace.sample_token = Some(tok);
                        trace.head_topk_idx = topk_idx;
                        trace.head_topk_val = topk_val;
                    }

                    if gpu_lm_head_full_logits_enabled() {
                        logits = read_f32(&pool.logits, a.vocab_size);
                        crate::cost_ledger::record_transfer(
                            (a.vocab_size * 4) as u64,
                            false,
                            "lm_head_y_download",
                        );
                        trace.head_full_logits_readback = true;
                    } else {
                        // Empty host logits: callers that need a token use
                        // `trace.sample_token`. Continuous-logit parity sets
                        // FULL_LOGITS=1.
                        logits = Vec::new();
                        trace.head_full_logits_readback = false;
                    }
                }
                GpuTensor::NativeCpu(_) | GpuTensor::Pq { .. } => {
                    drop(cache);
                    let hidden = read_f32(&pool.final_hidden, a.hidden);
                    logits = weights.matvec("lm_head.weight", &hidden)?;
                    trace.head_full_logits_readback = true;
                    if session.waits.get() == waits_before_head {
                        let mut cache = weights.cache.lock().expect("gpu weight cache");
                        weights.ensure_many_locked(&mut cache, &["lm_head.weight"])?;
                        if matches!(cache.get("lm_head.weight"), Some(GpuTensor::Pq { .. })) {
                            session.waits.set(session.waits.get().saturating_add(1));
                        }
                    }
                }
            }
        }
    }

    session.seq_len = required_sequence;
    let waits = session.waits.get().saturating_sub(waits_before);
    Ok((logits, trace, waits))
}

fn router_select(
    weights: &GpuWeightCache,
    a: &GlmArch,
    prefix: &str,
    pool: &ActPool,
) -> Result<(Vec<usize>, Vec<f32>)> {
    // Host-side noaux_tc arithmetic after the gate matvec. Nested under any
    // open parent; when none is open this is the exclusive Routing line.
    let _route = crate::cost_ledger::Scope::new(crate::cost_ledger::Bucket::Routing);
    let logits = read_f32(&pool.router_logits, a.n_routed_experts);
    let scores: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
    crate::cost_ledger::record_source_modelled_operations(
        (3 * a.n_routed_experts) as u64,
        0,
        0,
        a.n_routed_experts as u64,
        0,
    );
    crate::cost_ledger::record_allocation((scores.len() * 4) as u64);
    write_f32(&pool.router_scores, &scores);
    let bias = weights.dense(&format!("{prefix}.gate.e_score_correction_bias"))?;
    let corrected: Vec<f32> = scores.iter().zip(&bias).map(|(s, b)| s + b).collect();
    crate::cost_ledger::record_source_modelled_operations(corrected.len() as u64, 0, 0, 0, 0);
    crate::cost_ledger::record_allocation((corrected.len() * 4) as u64);
    write_f32(&pool.router_corrected, &corrected);
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
    let chosen = topk_desc(&group_scores, a.topk_group);
    let mut choice = vec![f32::NEG_INFINITY; a.n_routed_experts];
    for &g in &chosen {
        for e in g * per_group..(g + 1) * per_group {
            choice[e] = corrected[e];
        }
    }
    let indices = topk_desc(&choice, a.num_experts_per_tok);
    let mut weights_out: Vec<f32> = indices.iter().map(|&i| scores[i]).collect();
    crate::cost_ledger::record_allocation(
        ((group_scores.len() + choice.len() + weights_out.len()) * 4) as u64,
    );
    if a.norm_topk_prob {
        let total: f32 = weights_out.iter().sum::<f32>() + 1e-20;
        crate::cost_ledger::record_source_modelled_operations(
            (2 * weights_out.len() + 1) as u64,
            0,
            0,
            0,
            0,
        );
        for w in weights_out.iter_mut() {
            *w /= total;
        }
    }
    crate::cost_ledger::record_source_modelled_operations(weights_out.len() as u64, 0, 0, 0, 0);
    for w in weights_out.iter_mut() {
        *w *= a.routed_scaling_factor;
    }
    Ok((indices, weights_out))
}

fn sparse_attend(
    a: &GlmArch,
    pool: &ActPool,
    cache: &LayerGpuCache,
    scratch: &mut SequenceScratch,
    pos: usize,
    topk: &[usize],
    qk: usize,
) -> Result<Vec<f32>> {
    let n_keys = active_sequence_len(pos, cache.capacity, "resident attention cache")?;
    let scratch_len = scratch.active_len(pos)?;
    if scratch_len != n_keys {
        return Err(Error::Gravity(format!(
            "resident sparse attention capacity mismatch: cache={n_keys}, scratch={scratch_len}"
        )));
    }
    let keys = unsafe {
        std::slice::from_raw_parts(cache.keys.contents() as *const f32, n_keys * a.n_heads * qk)
    };
    let values = unsafe {
        std::slice::from_raw_parts(
            cache.values.contents() as *const f32,
            n_keys * a.n_heads * a.v_head_dim,
        )
    };
    let queries = read_f32(&pool.queries, a.n_heads * qk);
    let HostSequenceScratch {
        attention_allowed,
        attention_scores,
        ..
    } = &mut scratch.host;
    let allow = &mut attention_allowed[..n_keys];
    allow.fill(0);
    for &t in topk {
        if t <= pos && t < n_keys {
            allow[t] = 1;
        }
    }
    let selected = allow.iter().filter(|&&v| v != 0).count() as u64;
    let heads = a.n_heads as u64;
    let per_selected_fp = (2 * qk + 4 + 2 * a.v_head_dim) as u64;
    crate::cost_ledger::record_source_modelled_operations(
        heads
            .saturating_mul(selected)
            .saturating_mul(per_selected_fp),
        0,
        heads.saturating_mul(n_keys as u64),
        heads.saturating_mul(selected),
        0,
    );
    let scale = (qk as f32).powf(-0.5);
    let mut context = vec![0f32; a.n_heads * a.v_head_dim];
    let scores = &mut attention_scores[..n_keys];
    scores.fill(f32::NEG_INFINITY);
    crate::cost_ledger::record_allocation((context.len() * 4) as u64);
    for head in 0..a.n_heads {
        let qh = &queries[head * qk..(head + 1) * qk];
        let mut best = f32::NEG_INFINITY;
        for t in 0..n_keys {
            if allow[t] == 0 {
                scores[t] = f32::NEG_INFINITY;
                continue;
            }
            let off = (t * a.n_heads + head) * qk;
            let dot: f32 = qh
                .iter()
                .zip(&keys[off..off + qk])
                .map(|(x, y)| x * y)
                .sum();
            scores[t] = dot * scale;
            best = best.max(scores[t]);
        }
        let mut total = 0f32;
        for s in scores.iter_mut() {
            *s = if s.is_finite() {
                (*s - best).exp()
            } else {
                0.0
            };
            total += *s;
        }
        let out = &mut context[head * a.v_head_dim..(head + 1) * a.v_head_dim];
        for (t, &prob) in scores.iter().enumerate() {
            if prob == 0.0 {
                continue;
            }
            let w = prob / total;
            let off = (t * a.n_heads + head) * a.v_head_dim;
            for (o, v) in out.iter_mut().zip(&values[off..off + a.v_head_dim]) {
                *o += w * v;
            }
        }
    }
    Ok(context)
}

#[allow(clippy::too_many_arguments)]
fn indexer_topk<'a>(
    weights: &GpuWeightCache,
    arch: &GlmArch,
    attn_p: &str,
    pool: &ActPool,
    cache: &LayerGpuCache,
    scratch: &mut SequenceScratch,
    pos: usize,
    cos: &[f32],
    sin: &[f32],
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<Vec<usize>> {
    let a = arch;
    let (ih, idim, rot) = (a.index_n_heads, a.index_head_dim, a.qk_rope_head_dim);
    let idx = format!("{attn_p}.indexer");

    matvec_into(
        tcb,
        ctx,
        weights,
        &format!("{idx}.wq_b.weight"),
        &pool.q_resid,
        a.q_lora_rank,
        &pool.idx_q,
    )?;
    matvec_into(
        tcb,
        ctx,
        weights,
        &format!("{idx}.wk.weight"),
        &pool.h,
        a.hidden,
        &pool.idx_k_raw,
    )?;
    commit(tcb.take(), waits)?;

    let k_raw = read_f32(&pool.idx_k_raw, idim);
    let kw = weights.dense(&format!("{idx}.k_norm.weight"))?;
    let kb = weights.dense(&format!("{idx}.k_norm.bias"))?;
    let k = {
        let n = k_raw.len() as f32;
        let mean = k_raw.iter().sum::<f32>() / n;
        let var = k_raw.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + 1e-6).sqrt();
        (0..k_raw.len())
            .map(|i| (k_raw[i] - mean) * inv * kw[i] + kb[i])
            .collect::<Vec<_>>()
    };
    let mut k_full = rope_interleaved(&k[..rot], cos, sin);
    k_full.extend_from_slice(&k[rot..]);
    unsafe {
        std::ptr::copy_nonoverlapping(
            k_full.as_ptr(),
            (cache.index_keys.contents() as *mut f32).add(pos * idim),
            idim,
        );
    }

    let q = read_f32(&pool.idx_q, ih * idim);
    let mut q_full = vec![0f32; ih * idim];
    for h in 0..ih {
        let src = &q[h * idim..(h + 1) * idim];
        let rotated = rope_interleaved(&src[..rot], cos, sin);
        q_full[h * idim..h * idim + rot].copy_from_slice(&rotated);
        q_full[h * idim + rot..(h + 1) * idim].copy_from_slice(&src[rot..]);
    }

    matvec_into(
        tcb,
        ctx,
        weights,
        &format!("{idx}.weights_proj.weight"),
        &pool.h,
        a.hidden,
        &pool.idx_head_w,
    )?;
    commit(tcb.take(), waits)?;
    let head_scale = (ih as f32).powf(-0.5);
    let mut head_weights = read_f32(&pool.idx_head_w, ih);
    for w in head_weights.iter_mut() {
        *w *= head_scale;
    }

    let n_keys = active_sequence_len(pos, cache.capacity, "resident index-key cache")?;
    let scratch_len = scratch.active_len(pos)?;
    if scratch_len != n_keys {
        return Err(Error::Gravity(format!(
            "resident indexer capacity mismatch: cache={n_keys}, scratch={scratch_len}"
        )));
    }
    let dim_scale = (idim as f32).powf(-0.5);
    let index_keys = unsafe {
        std::slice::from_raw_parts(cache.index_keys.contents() as *const f32, n_keys * idim)
    };
    crate::cost_ledger::record_source_modelled_operations(
        (n_keys as u64)
            .saturating_mul(ih as u64)
            .saturating_mul((2 * idim + 3) as u64),
        0,
        (n_keys as u64).saturating_mul(ih as u64),
        0,
        0,
    );
    let topk = {
        let HostSequenceScratch {
            index_scores,
            selection_indices,
            ..
        } = &mut scratch.host;
        let index_scores = &mut index_scores[..n_keys];
        index_scores.fill(0.0);
        for (t, score) in index_scores.iter_mut().enumerate() {
            let key = &index_keys[t * idim..(t + 1) * idim];
            let mut acc = 0f32;
            for h in 0..ih {
                let qh = &q_full[h * idim..(h + 1) * idim];
                let dot: f32 = qh.iter().zip(key).map(|(x, y)| x * y).sum();
                acc += head_weights[h] * (dot * dim_scale).max(0.0);
            }
            *score = acc;
        }
        for (t, score) in index_scores.iter_mut().enumerate() {
            if t > pos {
                *score = f32::NEG_INFINITY;
            }
        }
        topk_desc_with_scratch(
            index_scores,
            a.index_topk.min(n_keys),
            &mut selection_indices[..n_keys],
        )?
    };
    scratch.store_index_scores(n_keys)?;
    Ok(topk)
}

#[allow(clippy::too_many_arguments)]
fn mlp_one<'a>(
    weights: &GpuWeightCache,
    prefix: &str,
    x: &Buffer,
    x_len: usize,
    pool: &ActPool,
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<Vec<f32>> {
    // Expert-wave: one CB for dense MLP. Default path below is unchanged.
    if gpu_expert_wave_enabled() {
        return moe_device_wave(
            weights,
            &[prefix.to_string()],
            &[1.0f32],
            x,
            x_len,
            tcb,
            ctx,
            waits,
        );
    }
    let mut outs = batched_mlp(
        weights,
        &[prefix.to_string()],
        x,
        x_len,
        pool,
        tcb,
        ctx,
        waits,
    )?;
    outs.pop()
        .ok_or_else(|| Error::Gravity("mlp_one empty".into()))
}

/// Gate/up/down co-issued across all prefixes via `matvec_batch` — three waits
/// total for the whole expert set (matches host `batched_mlp`). The residual
/// `x` and KV stay on device; per-expert gate/up/act vectors are ephemeral
/// because each down_proj takes a different input.
///
/// **Default resident path. Do not edit for expert-wave.** The flagged collapse
/// lives in [`moe_device_wave`]. Changing this function is a Parity V2.1 item 6
/// regression.
#[allow(clippy::too_many_arguments)]
fn batched_mlp<'a>(
    weights: &GpuWeightCache,
    prefixes: &[String],
    x: &Buffer,
    x_len: usize,
    _pool: &ActPool,
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    _ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<Vec<Vec<f32>>> {
    if prefixes.is_empty() {
        return Ok(Vec::new());
    }
    // Flush any pending attention/router encodes before the batch path, which
    // commits on its own.
    commit(tcb.take(), waits)?;
    let x_host = read_f32(x, x_len);
    let gate_names: Vec<String> = prefixes
        .iter()
        .map(|p| format!("{p}.gate_proj.weight"))
        .collect();
    let up_names: Vec<String> = prefixes
        .iter()
        .map(|p| format!("{p}.up_proj.weight"))
        .collect();
    let gate_calls: Vec<(&str, &[f32])> = gate_names
        .iter()
        .map(|n| (n.as_str(), x_host.as_slice()))
        .collect();
    let up_calls: Vec<(&str, &[f32])> = up_names
        .iter()
        .map(|n| (n.as_str(), x_host.as_slice()))
        .collect();
    let gate_outs = weights.matvec_batch(&gate_calls)?;
    waits.set(waits.get().saturating_add(1));
    let up_outs = weights.matvec_batch(&up_calls)?;
    waits.set(waits.get().saturating_add(1));
    let acts: Vec<Vec<f32>> = gate_outs
        .iter()
        .zip(&up_outs)
        .map(|(g, u)| {
            g.iter()
                .zip(u)
                .map(|(gv, uv)| (gv / (1.0 + (-gv).exp())) * uv)
                .collect()
        })
        .collect();
    let activation_elements = gate_outs.iter().map(Vec::len).sum::<usize>() as u64;
    crate::cost_ledger::record_source_modelled_operations(
        activation_elements.saturating_mul(4),
        0,
        0,
        activation_elements,
        0,
    );
    crate::cost_ledger::record_allocation(activation_elements.saturating_mul(4));
    let down_names: Vec<String> = prefixes
        .iter()
        .map(|p| format!("{p}.down_proj.weight"))
        .collect();
    let down_calls: Vec<(&str, &[f32])> = down_names
        .iter()
        .zip(&acts)
        .map(|(n, a)| (n.as_str(), a.as_slice()))
        .collect();
    let downs = weights.matvec_batch(&down_calls)?;
    waits.set(waits.get().saturating_add(1));
    Ok(downs)
}

// ── Future route-segment primitives (encode-only; default path untouched) ───

/// Typed ABI boundary for the dormant GLM resident kernels.
///
/// These wrappers only append work to a caller-owned [`TokenCommandBuffer`].
/// They never submit, wait, inspect flags, or participate in
/// [`forward_resident`]. That keeps the current default path byte-for-byte
/// structurally unchanged while making future graph work explicit and
/// independently testable.
///
/// The existing MLA append and sparse-attention shaders use the expanded
/// `[position][head][qk/value]` cache. They are transitional correctness
/// scaffolding, not the 32K cache solution: the compact design must consume
/// normalized 512-wide MLA latent plus the shared 64-wide RoPE tail, or
/// reconstruct expanded K/V only for selected positions.
#[allow(dead_code)]
mod route_segment_primitives {
    use super::*;

    const TG: u32 = 256;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct GlmRopeParams {
        pub n_heads: u32,
        pub rotary_dim: u32,
        pub in_stride: u32,
        pub out_stride: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct GlmMlaAppendParams {
        pub n_heads: u32,
        pub qk_nope: u32,
        pub qk_rope: u32,
        pub v_dim: u32,
        pub pos: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct GlmBuildQParams {
        pub n_heads: u32,
        pub qk_nope: u32,
        pub qk_rope: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(super) struct GlmDsaParams {
        pub n_keys: u32,
        pub n_heads: u32,
        pub head_dim: u32,
        pub pos: u32,
        pub dim_scale: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct GlmTopkParams {
        pub n: u32,
        pub k: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct GlmSortU32Params {
        pub n: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(super) struct GlmSparseAttnParams {
        pub n_heads: u32,
        pub qk_dim: u32,
        pub v_dim: u32,
        pub n_keys: u32,
        pub n_allow: u32,
        pub scale: f32,
    }

    const _: [(); 16] = [(); std::mem::size_of::<GlmRopeParams>()];
    const _: [(); 20] = [(); std::mem::size_of::<GlmMlaAppendParams>()];
    const _: [(); 12] = [(); std::mem::size_of::<GlmBuildQParams>()];
    const _: [(); 20] = [(); std::mem::size_of::<GlmDsaParams>()];
    const _: [(); 8] = [(); std::mem::size_of::<GlmTopkParams>()];
    const _: [(); 4] = [(); std::mem::size_of::<GlmSortU32Params>()];
    const _: [(); 24] = [(); std::mem::size_of::<GlmSparseAttnParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmRopeParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmMlaAppendParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmBuildQParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmDsaParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmTopkParams>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmSortU32Params>()];
    const _: [(); 4] = [(); std::mem::align_of::<GlmSparseAttnParams>()];

    fn u32_arg(value: usize, what: &str) -> Result<u32> {
        u32::try_from(value)
            .map_err(|_| Error::Gravity(format!("{what}: {value} does not fit the Metal u32 ABI")))
    }

    fn checked_add(a: usize, b: usize, what: &str) -> Result<usize> {
        a.checked_add(b)
            .ok_or_else(|| Error::Gravity(format!("{what}: size overflow ({a} + {b})")))
    }

    fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize> {
        a.checked_mul(b)
            .ok_or_else(|| Error::Gravity(format!("{what}: size overflow ({a} x {b})")))
    }

    fn require_range(
        buffer: &Buffer,
        element_offset: usize,
        elements: usize,
        element_bytes: usize,
        what: &str,
    ) -> Result<u64> {
        let offset = checked_mul(element_offset, element_bytes, what)?;
        let bytes = checked_mul(elements, element_bytes, what)?;
        let end = checked_add(offset, bytes, what)?;
        if end as u64 > buffer.length() {
            return Err(Error::Gravity(format!(
                "{what}: needs byte range [{offset}, {end}), buffer has {} bytes",
                buffer.length()
            )));
        }
        Ok(offset as u64)
    }

    fn require_f32(
        buffer: &Buffer,
        element_offset: usize,
        elements: usize,
        what: &str,
    ) -> Result<u64> {
        require_range(
            buffer,
            element_offset,
            elements,
            std::mem::size_of::<f32>(),
            what,
        )
    }

    fn grid_1d(elements: u32, what: &str) -> Result<(u32, u32, u32)> {
        let groups = elements.div_ceil(TG);
        let width = groups
            .checked_mul(TG)
            .ok_or_else(|| Error::Gravity(format!("{what}: rounded Metal grid width overflow")))?;
        Ok((width, 1, 1))
    }

    fn strided_elements(count: usize, stride: usize, width: usize, what: &str) -> Result<usize> {
        if count == 0 {
            return Ok(0);
        }
        let preceding = checked_mul(count - 1, stride, what)?;
        checked_add(preceding, width, what)
    }

    pub(super) fn encode_rmsnorm(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &Buffer,
        weight: &Buffer,
        out: &Buffer,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        require_f32(x, 0, n, "gravity_rmsnorm_f32 x")?;
        require_f32(weight, 0, n, "gravity_rmsnorm_f32 weight")?;
        require_f32(out, 0, n, "gravity_rmsnorm_f32 out")?;
        let n = u32_arg(n, "gravity_rmsnorm_f32 n")?;
        let xb = x.clone();
        let wb = weight.clone();
        let ob = out.clone();
        tcb.dispatch_threads("gravity_rmsnorm_f32", (TG, 1, 1), (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&xb), 0);
            enc.set_buffer(1, Some(&wb), 0);
            enc.set_buffer(2, Some(&ob), 0);
            enc.set_bytes(3, 4, &n as *const u32 as *const _);
            enc.set_bytes(4, 4, &eps as *const f32 as *const _);
            enc.set_threadgroup_memory_length(0, (TG as u64) * 4);
        })
    }

    pub(super) fn encode_layernorm_affine(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &Buffer,
        weight: &Buffer,
        bias: &Buffer,
        out: &Buffer,
        n: usize,
        eps: f32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        require_f32(x, 0, n, "gravity_layernorm_affine_f32 x")?;
        require_f32(weight, 0, n, "gravity_layernorm_affine_f32 weight")?;
        require_f32(bias, 0, n, "gravity_layernorm_affine_f32 bias")?;
        require_f32(out, 0, n, "gravity_layernorm_affine_f32 out")?;
        let n = u32_arg(n, "gravity_layernorm_affine_f32 n")?;
        let xb = x.clone();
        let wb = weight.clone();
        let bb = bias.clone();
        let ob = out.clone();
        tcb.dispatch_threads(
            "gravity_layernorm_affine_f32",
            (TG, 1, 1),
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&xb), 0);
                enc.set_buffer(1, Some(&wb), 0);
                enc.set_buffer(2, Some(&bb), 0);
                enc.set_buffer(3, Some(&ob), 0);
                enc.set_bytes(4, 4, &n as *const u32 as *const _);
                enc.set_bytes(5, 4, &eps as *const f32 as *const _);
                enc.set_threadgroup_memory_length(0, (TG as u64) * 4);
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_rope_interleaved(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &Buffer,
        input_element_offset: usize,
        out: &Buffer,
        output_element_offset: usize,
        cos: &Buffer,
        sin: &Buffer,
        n_heads: usize,
        rotary_dim: usize,
        in_stride: usize,
        out_stride: usize,
    ) -> Result<()> {
        if n_heads == 0 || rotary_dim == 0 {
            return Ok(());
        }
        if rotary_dim % 2 != 0 || in_stride < rotary_dim || out_stride < rotary_dim {
            return Err(Error::Gravity(format!(
                "gravity_rope_interleaved_f32 invalid geometry: heads={n_heads}, rotary_dim={rotary_dim}, in_stride={in_stride}, out_stride={out_stride}"
            )));
        }
        let input_len = strided_elements(
            n_heads,
            in_stride,
            rotary_dim,
            "gravity_rope_interleaved_f32 input",
        )?;
        let output_len = strided_elements(
            n_heads,
            out_stride,
            rotary_dim,
            "gravity_rope_interleaved_f32 output",
        )?;
        let input_byte_offset = require_f32(
            x,
            input_element_offset,
            input_len,
            "gravity_rope_interleaved_f32 input",
        )?;
        let output_byte_offset = require_f32(
            out,
            output_element_offset,
            output_len,
            "gravity_rope_interleaved_f32 output",
        )?;
        require_f32(cos, 0, rotary_dim / 2, "gravity_rope_interleaved_f32 cos")?;
        require_f32(sin, 0, rotary_dim / 2, "gravity_rope_interleaved_f32 sin")?;
        let params = GlmRopeParams {
            n_heads: u32_arg(n_heads, "gravity_rope_interleaved_f32 n_heads")?,
            rotary_dim: u32_arg(rotary_dim, "gravity_rope_interleaved_f32 rotary_dim")?,
            in_stride: u32_arg(in_stride, "gravity_rope_interleaved_f32 in_stride")?,
            out_stride: u32_arg(out_stride, "gravity_rope_interleaved_f32 out_stride")?,
        };
        let threads = params
            .n_heads
            .checked_mul(params.rotary_dim / 2)
            .ok_or_else(|| Error::Gravity("gravity_rope_interleaved_f32 grid overflow".into()))?;
        let grid = grid_1d(threads, "gravity_rope_interleaved_f32")?;
        let xb = x.clone();
        let ob = out.clone();
        let cb = cos.clone();
        let sb = sin.clone();
        tcb.dispatch_threads(
            "gravity_rope_interleaved_f32",
            grid,
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&xb), input_byte_offset);
                enc.set_buffer(1, Some(&ob), output_byte_offset);
                enc.set_buffer(2, Some(&cb), 0);
                enc.set_buffer(3, Some(&sb), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
            },
        )
    }

    pub(super) fn encode_copy_tail(
        tcb: &mut TokenCommandBuffer<'_>,
        src: &Buffer,
        dst: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        require_f32(src, src_offset, n, "gravity_copy_tail_f32 src")?;
        require_f32(dst, dst_offset, n, "gravity_copy_tail_f32 dst")?;
        let src_offset = u32_arg(src_offset, "gravity_copy_tail_f32 src_off")?;
        let dst_offset = u32_arg(dst_offset, "gravity_copy_tail_f32 dst_off")?;
        let n = u32_arg(n, "gravity_copy_tail_f32 n")?;
        let grid = grid_1d(n, "gravity_copy_tail_f32")?;
        let sb = src.clone();
        let db = dst.clone();
        tcb.dispatch_threads("gravity_copy_tail_f32", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&sb), 0);
            enc.set_buffer(1, Some(&db), 0);
            enc.set_bytes(2, 4, &src_offset as *const u32 as *const _);
            enc.set_bytes(3, 4, &dst_offset as *const u32 as *const _);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_mla_append_kv_expanded(
        tcb: &mut TokenCommandBuffer<'_>,
        kv: &Buffer,
        k_rot: &Buffer,
        keys: &Buffer,
        values: &Buffer,
        n_heads: usize,
        qk_nope: usize,
        qk_rope: usize,
        v_dim: usize,
        position: usize,
    ) -> Result<()> {
        let qk = checked_add(qk_nope, qk_rope, "gravity_glm_mla_append_kv qk")?;
        let per_kv = checked_add(qk_nope, v_dim, "gravity_glm_mla_append_kv per_kv")?;
        let key_elems = checked_mul(n_heads, qk, "gravity_glm_mla_append_kv key elements")?;
        let value_elems = checked_mul(n_heads, v_dim, "gravity_glm_mla_append_kv value elements")?;
        let total = checked_add(key_elems, value_elems, "gravity_glm_mla_append_kv grid")?;
        if total == 0 {
            return Ok(());
        }
        require_f32(
            kv,
            0,
            checked_mul(n_heads, per_kv, "gravity_glm_mla_append_kv kv")?,
            "gravity_glm_mla_append_kv kv",
        )?;
        require_f32(k_rot, 0, qk_rope, "gravity_glm_mla_append_kv k_rot")?;
        let positions = checked_add(position, 1, "gravity_glm_mla_append_kv position")?;
        require_f32(
            keys,
            0,
            checked_mul(positions, key_elems, "gravity_glm_mla_append_kv keys")?,
            "gravity_glm_mla_append_kv expanded keys",
        )?;
        require_f32(
            values,
            0,
            checked_mul(positions, value_elems, "gravity_glm_mla_append_kv values")?,
            "gravity_glm_mla_append_kv expanded values",
        )?;
        let params = GlmMlaAppendParams {
            n_heads: u32_arg(n_heads, "gravity_glm_mla_append_kv n_heads")?,
            qk_nope: u32_arg(qk_nope, "gravity_glm_mla_append_kv qk_nope")?,
            qk_rope: u32_arg(qk_rope, "gravity_glm_mla_append_kv qk_rope")?,
            v_dim: u32_arg(v_dim, "gravity_glm_mla_append_kv v_dim")?,
            pos: u32_arg(position, "gravity_glm_mla_append_kv pos")?,
        };
        let grid = grid_1d(
            u32_arg(total, "gravity_glm_mla_append_kv total")?,
            "gravity_glm_mla_append_kv",
        )?;
        let kvb = kv.clone();
        let krb = k_rot.clone();
        let kb = keys.clone();
        let vb = values.clone();
        tcb.dispatch_threads("gravity_glm_mla_append_kv", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&kvb), 0);
            enc.set_buffer(1, Some(&krb), 0);
            enc.set_buffer(2, Some(&kb), 0);
            enc.set_buffer(3, Some(&vb), 0);
            enc.set_bytes(
                4,
                std::mem::size_of_val(&params) as u64,
                &params as *const _ as *const _,
            );
        })
    }

    pub(super) fn encode_build_queries(
        tcb: &mut TokenCommandBuffer<'_>,
        q: &Buffer,
        q_rope_rot: &Buffer,
        queries: &Buffer,
        n_heads: usize,
        qk_nope: usize,
        qk_rope: usize,
    ) -> Result<()> {
        let qk = checked_add(qk_nope, qk_rope, "gravity_glm_build_queries qk")?;
        let total = checked_mul(n_heads, qk, "gravity_glm_build_queries total")?;
        if total == 0 {
            return Ok(());
        }
        require_f32(q, 0, total, "gravity_glm_build_queries q")?;
        require_f32(
            q_rope_rot,
            0,
            checked_mul(n_heads, qk_rope, "gravity_glm_build_queries q_rope_rot")?,
            "gravity_glm_build_queries q_rope_rot",
        )?;
        require_f32(queries, 0, total, "gravity_glm_build_queries queries")?;
        let params = GlmBuildQParams {
            n_heads: u32_arg(n_heads, "gravity_glm_build_queries n_heads")?,
            qk_nope: u32_arg(qk_nope, "gravity_glm_build_queries qk_nope")?,
            qk_rope: u32_arg(qk_rope, "gravity_glm_build_queries qk_rope")?,
        };
        let grid = grid_1d(
            u32_arg(total, "gravity_glm_build_queries total")?,
            "gravity_glm_build_queries",
        )?;
        let qb = q.clone();
        let rb = q_rope_rot.clone();
        let ob = queries.clone();
        tcb.dispatch_threads("gravity_glm_build_queries", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&qb), 0);
            enc.set_buffer(1, Some(&rb), 0);
            enc.set_buffer(2, Some(&ob), 0);
            enc.set_bytes(
                3,
                std::mem::size_of_val(&params) as u64,
                &params as *const _ as *const _,
            );
        })
    }

    pub(super) fn encode_append_index_key(
        tcb: &mut TokenCommandBuffer<'_>,
        k_full: &Buffer,
        index_keys: &Buffer,
        position: usize,
        head_dim: usize,
    ) -> Result<()> {
        if head_dim == 0 {
            return Ok(());
        }
        require_f32(k_full, 0, head_dim, "gravity_glm_append_index_key k_full")?;
        let positions = checked_add(position, 1, "gravity_glm_append_index_key position")?;
        require_f32(
            index_keys,
            0,
            checked_mul(
                positions,
                head_dim,
                "gravity_glm_append_index_key index_keys",
            )?,
            "gravity_glm_append_index_key index_keys",
        )?;
        let position = u32_arg(position, "gravity_glm_append_index_key pos")?;
        let head_dim = u32_arg(head_dim, "gravity_glm_append_index_key idim")?;
        let grid = grid_1d(head_dim, "gravity_glm_append_index_key")?;
        let kb = k_full.clone();
        let ib = index_keys.clone();
        tcb.dispatch_threads(
            "gravity_glm_append_index_key",
            grid,
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&kb), 0);
                enc.set_buffer(1, Some(&ib), 0);
                enc.set_bytes(2, 4, &position as *const u32 as *const _);
                enc.set_bytes(3, 4, &head_dim as *const u32 as *const _);
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_dsa_scores(
        tcb: &mut TokenCommandBuffer<'_>,
        q_full: &Buffer,
        index_keys: &Buffer,
        head_weights: &Buffer,
        scores: &Buffer,
        n_keys: usize,
        n_heads: usize,
        head_dim: usize,
        position: usize,
        dim_scale: f32,
    ) -> Result<()> {
        if n_keys == 0 {
            return Ok(());
        }
        require_f32(
            q_full,
            0,
            checked_mul(n_heads, head_dim, "gravity_glm_dsa_scores q_full")?,
            "gravity_glm_dsa_scores q_full",
        )?;
        require_f32(
            index_keys,
            0,
            checked_mul(n_keys, head_dim, "gravity_glm_dsa_scores index_keys")?,
            "gravity_glm_dsa_scores index_keys",
        )?;
        require_f32(
            head_weights,
            0,
            n_heads,
            "gravity_glm_dsa_scores head_weights",
        )?;
        require_f32(scores, 0, n_keys, "gravity_glm_dsa_scores scores")?;
        let params = GlmDsaParams {
            n_keys: u32_arg(n_keys, "gravity_glm_dsa_scores n_keys")?,
            n_heads: u32_arg(n_heads, "gravity_glm_dsa_scores n_heads")?,
            head_dim: u32_arg(head_dim, "gravity_glm_dsa_scores head_dim")?,
            pos: u32_arg(position, "gravity_glm_dsa_scores pos")?,
            dim_scale,
        };
        let grid = grid_1d(params.n_keys, "gravity_glm_dsa_scores")?;
        let qb = q_full.clone();
        let kb = index_keys.clone();
        let wb = head_weights.clone();
        let sb = scores.clone();
        tcb.dispatch_threads("gravity_glm_dsa_scores", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&qb), 0);
            enc.set_buffer(1, Some(&kb), 0);
            enc.set_buffer(2, Some(&wb), 0);
            enc.set_buffer(3, Some(&sb), 0);
            enc.set_bytes(
                4,
                std::mem::size_of_val(&params) as u64,
                &params as *const _ as *const _,
            );
        })
    }

    pub(super) fn encode_stable_topk(
        tcb: &mut TokenCommandBuffer<'_>,
        values: &Buffer,
        indices: &Buffer,
        selected_scratch: &Buffer,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if n == 0 || k == 0 {
            return Ok(());
        }
        let out_len = k.min(n);
        require_f32(values, 0, n, "gravity_glm_stable_topk_f32 values")?;
        require_range(
            indices,
            0,
            out_len,
            std::mem::size_of::<u32>(),
            "gravity_glm_stable_topk_f32 indices",
        )?;
        require_range(
            selected_scratch,
            0,
            n,
            std::mem::size_of::<u8>(),
            "gravity_glm_stable_topk_f32 selected",
        )?;
        let params = GlmTopkParams {
            n: u32_arg(n, "gravity_glm_stable_topk_f32 n")?,
            k: u32_arg(k, "gravity_glm_stable_topk_f32 k")?,
        };
        let vb = values.clone();
        let ib = indices.clone();
        let sb = selected_scratch.clone();
        tcb.dispatch_threads(
            "gravity_glm_stable_topk_f32",
            (1, 1, 1),
            (1, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&vb), 0);
                enc.set_buffer(1, Some(&ib), 0);
                enc.set_buffer(2, Some(&sb), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
            },
        )
    }

    /// Sort unique stable-top-k position IDs into ascending host accumulation
    /// order. Bounded to the flagship `index_topk <= 2048` contract.
    ///
    /// The kernel uses one 256-thread group and one power-of-two-padded u32
    /// array in dynamic threadgroup memory (maximum 8 KiB). Input and output
    /// may be the same Metal buffer.
    pub(super) fn encode_sort_positions_ascending(
        tcb: &mut TokenCommandBuffer<'_>,
        score_ordered_indices: &Buffer,
        ascending_indices: &Buffer,
        k: usize,
    ) -> Result<()> {
        const MAX_K: usize = 2048;
        if k == 0 {
            return Ok(());
        }
        if k > MAX_K {
            return Err(Error::Gravity(format!(
                "gravity_glm_sort_u32_ascending supports k <= {MAX_K}, got {k}"
            )));
        }
        require_range(
            score_ordered_indices,
            0,
            k,
            std::mem::size_of::<u32>(),
            "gravity_glm_sort_u32_ascending input",
        )?;
        require_range(
            ascending_indices,
            0,
            k,
            std::mem::size_of::<u32>(),
            "gravity_glm_sort_u32_ascending output",
        )?;
        let padded = k.checked_next_power_of_two().ok_or_else(|| {
            Error::Gravity("gravity_glm_sort_u32_ascending padded width overflow".into())
        })?;
        let shmem = checked_mul(
            padded,
            std::mem::size_of::<u32>(),
            "gravity_glm_sort_u32_ascending threadgroup memory",
        )?;
        let params = GlmSortU32Params {
            n: u32_arg(k, "gravity_glm_sort_u32_ascending n")?,
        };
        let input = score_ordered_indices.clone();
        let output = ascending_indices.clone();
        tcb.dispatch_threads(
            "gravity_glm_sort_u32_ascending",
            (TG, 1, 1),
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&input), 0);
                enc.set_buffer(1, Some(&output), 0);
                enc.set_bytes(
                    2,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem as u64);
            },
        )
    }

    /// Encode transitional expanded-cache sparse attention.
    ///
    /// `allow_idx` must contain unique positions in ascending position order
    /// to preserve the current host accumulation order. The stable-top-k
    /// output is score-ordered and must first pass through
    /// [`encode_sort_positions_ascending`].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_sparse_attention_expanded_ascending_allow(
        tcb: &mut TokenCommandBuffer<'_>,
        queries: &Buffer,
        keys: &Buffer,
        values: &Buffer,
        allow_idx: &Buffer,
        context: &Buffer,
        n_heads: usize,
        qk_dim: usize,
        v_dim: usize,
        n_keys: usize,
        n_allow: usize,
        scale: f32,
    ) -> Result<()> {
        if n_heads == 0 || qk_dim == 0 || v_dim == 0 {
            return Ok(());
        }
        require_f32(
            queries,
            0,
            checked_mul(n_heads, qk_dim, "gravity_glm_sparse_attn queries")?,
            "gravity_glm_sparse_attn queries",
        )?;
        require_f32(
            keys,
            0,
            checked_mul(
                checked_mul(n_keys, n_heads, "gravity_glm_sparse_attn keys")?,
                qk_dim,
                "gravity_glm_sparse_attn keys",
            )?,
            "gravity_glm_sparse_attn expanded keys",
        )?;
        require_f32(
            values,
            0,
            checked_mul(
                checked_mul(n_keys, n_heads, "gravity_glm_sparse_attn values")?,
                v_dim,
                "gravity_glm_sparse_attn values",
            )?,
            "gravity_glm_sparse_attn expanded values",
        )?;
        require_range(
            allow_idx,
            0,
            n_allow,
            std::mem::size_of::<u32>(),
            "gravity_glm_sparse_attn allow_idx",
        )?;
        require_f32(
            context,
            0,
            checked_mul(n_heads, v_dim, "gravity_glm_sparse_attn context")?,
            "gravity_glm_sparse_attn context",
        )?;
        let params = GlmSparseAttnParams {
            n_heads: u32_arg(n_heads, "gravity_glm_sparse_attn n_heads")?,
            qk_dim: u32_arg(qk_dim, "gravity_glm_sparse_attn qk_dim")?,
            v_dim: u32_arg(v_dim, "gravity_glm_sparse_attn v_dim")?,
            n_keys: u32_arg(n_keys, "gravity_glm_sparse_attn n_keys")?,
            n_allow: u32_arg(n_allow, "gravity_glm_sparse_attn n_allow")?,
            scale,
        };
        let grid_width = params
            .n_heads
            .checked_mul(TG)
            .ok_or_else(|| Error::Gravity("gravity_glm_sparse_attn grid overflow".into()))?;
        let shmem = checked_mul(
            n_allow.max(1),
            std::mem::size_of::<f32>(),
            "gravity_glm_sparse_attn threadgroup memory",
        )?;
        let qb = queries.clone();
        let kb = keys.clone();
        let vb = values.clone();
        let ab = allow_idx.clone();
        let cb = context.clone();
        tcb.dispatch_threads(
            "gravity_glm_sparse_attn",
            (grid_width, 1, 1),
            (TG, 1, 1),
            move |enc| {
                enc.set_buffer(0, Some(&qb), 0);
                enc.set_buffer(1, Some(&kb), 0);
                enc.set_buffer(2, Some(&vb), 0);
                enc.set_buffer(3, Some(&ab), 0);
                enc.set_buffer(4, Some(&cb), 0);
                enc.set_bytes(
                    5,
                    std::mem::size_of_val(&params) as u64,
                    &params as *const _ as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem as u64);
            },
        )
    }

    pub(super) fn encode_router_correction(
        tcb: &mut TokenCommandBuffer<'_>,
        logits: &Buffer,
        bias: &Buffer,
        scores: &Buffer,
        corrected: &Buffer,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        require_f32(logits, 0, n, "gravity_glm_router_correct logits")?;
        require_f32(bias, 0, n, "gravity_glm_router_correct bias")?;
        require_f32(scores, 0, n, "gravity_glm_router_correct scores")?;
        require_f32(corrected, 0, n, "gravity_glm_router_correct corrected")?;
        let n = u32_arg(n, "gravity_glm_router_correct n")?;
        let grid = grid_1d(n, "gravity_glm_router_correct")?;
        let lb = logits.clone();
        let bb = bias.clone();
        let sb = scores.clone();
        let cb = corrected.clone();
        tcb.dispatch_threads("gravity_glm_router_correct", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&lb), 0);
            enc.set_buffer(1, Some(&bb), 0);
            enc.set_buffer(2, Some(&sb), 0);
            enc.set_buffer(3, Some(&cb), 0);
            enc.set_bytes(4, 4, &n as *const u32 as *const _);
        })
    }

    /// Encode the residual add `x[i] += y[i]`.
    ///
    /// `x` and `y` may be the same Metal buffer: the shader assigns exactly
    /// one thread to each element and performs no cross-element access.
    pub(super) fn encode_residual_add_inplace(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &Buffer,
        y: &Buffer,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let n = u32_arg(n, "gravity_add_inplace_f32 n")?;
        let grid = grid_1d(n, "gravity_add_inplace_f32")?;
        require_f32(x, 0, n as usize, "gravity_add_inplace_f32 x")?;
        require_f32(y, 0, n as usize, "gravity_add_inplace_f32 y")?;
        let xb = x.clone();
        let yb = y.clone();
        tcb.dispatch_threads("gravity_add_inplace_f32", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&xb), 0);
            enc.set_buffer(1, Some(&yb), 0);
            enc.set_bytes(2, 4, &n as *const u32 as *const _);
        })
    }

    pub(super) fn encode_zero(
        tcb: &mut TokenCommandBuffer<'_>,
        buffer: &Buffer,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        require_f32(buffer, 0, n, "gravity_zero_f32 buffer")?;
        let n = u32_arg(n, "gravity_zero_f32 n")?;
        let grid = grid_1d(n, "gravity_zero_f32")?;
        let xb = buffer.clone();
        tcb.dispatch_threads("gravity_zero_f32", grid, (TG, 1, 1), move |enc| {
            enc.set_buffer(0, Some(&xb), 0);
            enc.set_bytes(1, 4, &n as *const u32 as *const _);
        })
    }
}

// ── Expert-wave (flagged; default path above is untouched) ─────────────────

fn encode_silu_mul_f32(
    tcb: &mut TokenCommandBuffer<'_>,
    gate: &Buffer,
    up: &Buffer,
    out: &Buffer,
    n: u32,
) -> Result<()> {
    crate::cost_ledger::record_source_modelled_operations(
        (n as u64).saturating_mul(4),
        0,
        0,
        n as u64,
        0,
    );
    const TG: u32 = 256;
    let n_u = n;
    let g = gate.clone();
    let u = up.clone();
    let o = out.clone();
    tcb.dispatch_threads(
        "gravity_silu_mul_f32",
        (n.div_ceil(TG) * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&g), 0);
            enc.set_buffer(1, Some(&u), 0);
            enc.set_buffer(2, Some(&o), 0);
            enc.set_bytes(3, 4, &n_u as *const u32 as *const _);
        },
    )
}

fn encode_axpy_f32(
    tcb: &mut TokenCommandBuffer<'_>,
    y: &Buffer,
    x: &Buffer,
    scale: f32,
    n: u32,
) -> Result<()> {
    crate::cost_ledger::record_source_modelled_operations((n as u64).saturating_mul(2), 0, 0, 0, 0);
    const TG: u32 = 256;
    let s = scale;
    let n_u = n;
    let yb = y.clone();
    let xb = x.clone();
    tcb.dispatch_threads(
        "gravity_axpy_f32",
        (n.div_ceil(TG) * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&yb), 0);
            enc.set_buffer(1, Some(&xb), 0);
            enc.set_bytes(2, 4, &s as *const f32 as *const _);
            enc.set_bytes(3, 4, &n_u as *const u32 as *const _);
        },
    )
}

fn encode_pq_matvec_device(
    tcb: &mut TokenCommandBuffer<'_>,
    codebooks: &Buffer,
    codes: &Buffer,
    params: crate::gravity_glm::gpu::PqParams,
    x: &Buffer,
    y: &Buffer,
) -> Result<()> {
    const TG: u32 = 256;
    let n_tg = params.rows.div_ceil(8);
    let p = params;
    let cb = codebooks.clone();
    let co = codes.clone();
    let xb = x.clone();
    let yb = y.clone();
    tcb.dispatch_threads(
        "gravity_pq_matvec",
        (n_tg * TG, 1, 1),
        (TG, 1, 1),
        move |enc| {
            enc.set_buffer(0, Some(&cb), 0);
            enc.set_buffer(1, Some(&co), 0);
            enc.set_buffer(2, Some(&xb), 0);
            enc.set_buffer(3, Some(&yb), 0);
            enc.set_bytes(
                4,
                std::mem::size_of_val(&p) as u64,
                &p as *const _ as *const _,
            );
        },
    )
}

/// Encode one weight matvec (device x → device y) into an open command buffer.
/// Host-native weights are applied immediately into `y` (no encode).
fn encode_weight_matvec(
    tcb: &mut TokenCommandBuffer<'_>,
    weights: &GpuWeightCache,
    name: &str,
    x: &Buffer,
    x_len: usize,
    y: &Buffer,
) -> Result<()> {
    crate::cost_ledger::record_matvec_call();
    let mut cache = weights.cache.lock().expect("gpu weight cache");
    weights.ensure_many_locked(&mut cache, &[name])?;
    match cache.get(name).expect("ensured") {
        GpuTensor::NativeCpu(w) => {
            crate::cost_ledger::record_active_bytes_for(name, (w.len() * 4) as u64);
            record_dense_matvec_ops((w.len() / x_len) as u64, x_len as u64);
            let x_host = read_f32(x, x_len);
            let y_host = matvec_dense(w, &x_host, name)?;
            write_f32(y, &y_host);
            Ok(())
        }
        GpuTensor::NativeGpuBf16 { buf, rows, cols } => {
            if x_len != *cols as usize {
                return Err(Error::Gravity(format!(
                    "expert-wave matvec {name}: x_len {x_len} != cols {cols}"
                )));
            }
            crate::cost_ledger::record_active_bytes_for(name, buf.length());
            record_dense_matvec_ops(*rows as u64, *cols as u64);
            encode_gemv_native_bf16_seq(tcb, buf, *rows, *cols, x, y)
        }
        GpuTensor::Pq {
            codebooks,
            codes,
            params,
        } => {
            if x_len != params.cols as usize {
                return Err(Error::Gravity(format!(
                    "expert-wave matvec {name}: x_len {x_len} != cols {}",
                    params.cols
                )));
            }
            crate::cost_ledger::record_active_bytes_for(name, codebooks.length() + codes.length());
            record_pq_matvec_ops(*params);
            encode_pq_matvec_device(tcb, codebooks, codes, *params, x, y)
        }
    }
}

/// Isolated device path: **gate + up → SiLU → down → weighted combine** in one
/// command buffer (one `commit_and_wait`). Flagged via
/// [`crate::gravity_glm::GPU_EXPERT_WAVE_ENV`]; never called from the default
/// resident path.
///
/// `scales[i]` multiplies prefix `i`'s down projection into the sum (MoE router
/// weights for routed experts, `1.0` for shared / dense). Accumulation order
/// matches the host: prefixes are already sorted ascending-expert then shared.
///
/// Requires every gate/up/down weight to be device-resident (`Pq` or
/// `NativeGpuBf16`). Host-native tensors fall back to a single host pass with
/// one wait tick (tiny fixtures without `HAWKING_GLM_GPU_LM_HEAD`); the pure
/// device path is what flagship PQ experts hit.
#[allow(clippy::too_many_arguments)]
fn moe_device_wave<'a>(
    weights: &GpuWeightCache,
    prefixes: &[String],
    scales: &[f32],
    x: &Buffer,
    x_len: usize,
    tcb: &mut Option<TokenCommandBuffer<'a>>,
    ctx: &'a MetalContext,
    waits: &Cell<u64>,
) -> Result<Vec<f32>> {
    if prefixes.is_empty() {
        return Ok(vec![0f32; x_len]);
    }
    if scales.len() != prefixes.len() {
        return Err(Error::Gravity(format!(
            "expert-wave: scales.len() {} != prefixes.len() {}",
            scales.len(),
            prefixes.len()
        )));
    }
    // Flush pending attention encodes; this path owns the next commit.
    commit(tcb.take(), waits)?;

    // Pin every projection for this layer before encoding so LRU cannot drop
    // a tensor mid-wave (same invariant as matvec_batch).
    let mut all_names: Vec<String> = Vec::with_capacity(prefixes.len() * 3);
    for p in prefixes {
        all_names.push(format!("{p}.gate_proj.weight"));
        all_names.push(format!("{p}.up_proj.weight"));
        all_names.push(format!("{p}.down_proj.weight"));
    }
    {
        let name_refs: Vec<&str> = all_names.iter().map(String::as_str).collect();
        let mut cache = weights.cache.lock().expect("gpu weight cache");
        weights.ensure_many_locked(&mut cache, &name_refs)?;
    }

    // Device-resident weights only on the pure CB path. Host-native needs the
    // host fallback (cannot encode silu→down dependence without a wait).
    let all_device = {
        let cache = weights.cache.lock().expect("gpu weight cache");
        all_names.iter().all(|n| {
            matches!(
                cache.get(n),
                Some(GpuTensor::Pq { .. }) | Some(GpuTensor::NativeGpuBf16 { .. })
            )
        })
    };
    if !all_device {
        return moe_wave_host_fallback(weights, prefixes, scales, x, x_len, waits);
    }

    // Discover intermediate width from the first gate projection.
    let inter = {
        let cache = weights.cache.lock().expect("gpu weight cache");
        let gname = format!("{}.gate_proj.weight", prefixes[0]);
        match cache.get(&gname).expect("ensured gate") {
            GpuTensor::Pq { params, .. } => params.rows as usize,
            GpuTensor::NativeGpuBf16 { rows, .. } => *rows as usize,
            GpuTensor::NativeCpu(_) => {
                return Err(Error::Gravity(
                    "expert-wave: gate is NativeCpu after device check".into(),
                ));
            }
        }
    };

    let n_exp = prefixes.len();
    let mut gate_bufs = Vec::with_capacity(n_exp);
    let mut up_bufs = Vec::with_capacity(n_exp);
    let mut act_bufs = Vec::with_capacity(n_exp);
    let mut down_bufs = Vec::with_capacity(n_exp);
    for _ in 0..n_exp {
        gate_bufs.push(ctx.new_buffer_checked(inter * 4)?);
        up_bufs.push(ctx.new_buffer_checked(inter * 4)?);
        act_bufs.push(ctx.new_buffer_checked(inter * 4)?);
        down_bufs.push(ctx.new_buffer_checked(x_len * 4)?);
    }
    let combined = ctx.new_buffer_checked(x_len * 4)?;
    // Host-zero the accumulator so device axpy starts from 0 (Metal shared).
    write_f32(&combined, &vec![0f32; x_len]);

    let mut wave = TokenCommandBuffer::new(ctx);
    for (i, p) in prefixes.iter().enumerate() {
        encode_weight_matvec(
            &mut wave,
            weights,
            &format!("{p}.gate_proj.weight"),
            x,
            x_len,
            &gate_bufs[i],
        )?;
        encode_weight_matvec(
            &mut wave,
            weights,
            &format!("{p}.up_proj.weight"),
            x,
            x_len,
            &up_bufs[i],
        )?;
    }
    for i in 0..n_exp {
        encode_silu_mul_f32(
            &mut wave,
            &gate_bufs[i],
            &up_bufs[i],
            &act_bufs[i],
            inter as u32,
        )?;
    }
    for (i, p) in prefixes.iter().enumerate() {
        encode_weight_matvec(
            &mut wave,
            weights,
            &format!("{p}.down_proj.weight"),
            &act_bufs[i],
            inter,
            &down_bufs[i],
        )?;
    }
    // Weighted combine in prefix order (associativity matches host).
    for i in 0..n_exp {
        encode_axpy_f32(&mut wave, &combined, &down_bufs[i], scales[i], x_len as u32)?;
    }

    // Encode/submit/sync + dispatch count fold at TCB commit when ledger on.
    wave.commit_and_wait()?;
    waits.set(waits.get().saturating_add(1));

    let out = read_f32(&combined, x_len);
    crate::cost_ledger::record_transfer((x_len * 4) as u64, false, "expert_wave_y_download");
    Ok(out)
}

/// Host-side gate/up/SiLU/down/combine used when expert weights are not
/// device-resident. Still bills **one** wait for the wave accounting contract
/// (flag on ⇒ collapsed MLP drain count); no intermediate readback loop.
fn moe_wave_host_fallback(
    weights: &GpuWeightCache,
    prefixes: &[String],
    scales: &[f32],
    x: &Buffer,
    x_len: usize,
    waits: &Cell<u64>,
) -> Result<Vec<f32>> {
    let x_host = read_f32(x, x_len);
    let mut combined = vec![0f32; x_len];
    for (p, &scale) in prefixes.iter().zip(scales.iter()) {
        let gate = weights.matvec(&format!("{p}.gate_proj.weight"), &x_host)?;
        let up = weights.matvec(&format!("{p}.up_proj.weight"), &x_host)?;
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let down = weights.matvec(&format!("{p}.down_proj.weight"), &act)?;
        for (c, d) in combined.iter_mut().zip(&down) {
            *c += *d * scale;
        }
    }
    // One wait tick: collapsed accounting for the flag-on path.
    waits.set(waits.get().saturating_add(1));
    Ok(combined)
}

/// Holds the long-lived resident state for a [`crate::gravity_glm::gpu::GravityGlmGpu`].
pub struct ResidentRuntime {
    pub session: Mutex<ResidentSession>,
    pub pool: ActPool,
}

impl ResidentRuntime {
    pub fn new(ctx: &MetalContext, arch: &GlmArch) -> Result<Self> {
        Ok(Self {
            session: Mutex::new(ResidentSession::new(
                ctx,
                arch,
                RESIDENT_RUNTIME_INITIAL_KV_CAPACITY_TOKENS,
            )?),
            pool: ActPool::new(ctx, arch)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::route_segment_primitives::*;
    use super::*;
    use crate::numeric_parity::{score_pair, Bounds};

    fn tiny_arch() -> GlmArch {
        GlmArch {
            n_layers: 1,
            hidden: 4,
            n_heads: 1,
            q_lora_rank: 2,
            kv_lora_rank: 2,
            qk_nope_head_dim: 1,
            qk_rope_head_dim: 1,
            v_head_dim: 1,
            index_n_heads: 1,
            index_head_dim: 1,
            index_topk: 2,
            n_routed_experts: 2,
            n_group: 1,
            topk_group: 1,
            num_experts_per_tok: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            vocab_size: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            indexer_types: vec!["full".into()],
            mlp_layer_types: vec!["dense".into()],
        }
    }

    fn f32_buffer(ctx: &MetalContext, values: &[f32]) -> Buffer {
        let buffer = ctx
            .new_buffer_checked(values.len() * std::mem::size_of::<f32>())
            .expect("f32 test buffer");
        write_f32(&buffer, values);
        buffer
    }

    fn filled_f32_buffer(ctx: &MetalContext, len: usize, value: f32) -> Buffer {
        f32_buffer(ctx, &vec![value; len])
    }

    fn u32_buffer(ctx: &MetalContext, values: &[u32]) -> Buffer {
        let buffer = ctx
            .new_buffer_checked(values.len() * std::mem::size_of::<u32>())
            .expect("u32 test buffer");
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                buffer.contents() as *mut u32,
                values.len(),
            );
        }
        buffer
    }

    fn empty_u8_buffer(ctx: &MetalContext, len: usize) -> Buffer {
        ctx.new_buffer_checked(len).expect("u8 test buffer")
    }

    fn assert_v21_pair(label: &str, host: &[f32], device: &[f32], reference: &[f64]) {
        let score = score_pair(host, device, reference, &Bounds::continuous_only());
        assert!(
            score.pass,
            "{label}: Numeric Parity V2.1 failure against FP64 authority; host={:?}, device={:?}",
            score.host.failures, score.device.failures
        );
    }

    fn topk_desc_f64(values: &[f64], k: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..values.len()).collect();
        indices.sort_by(|&a, &b| {
            values[b]
                .partial_cmp(&values[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        indices.truncate(k);
        indices
    }

    #[test]
    fn route_segment_parameter_abis_are_frozen_and_ranges_fail_closed() {
        assert_eq!(std::mem::size_of::<GlmRopeParams>(), 16);
        assert_eq!(std::mem::offset_of!(GlmRopeParams, n_heads), 0);
        assert_eq!(std::mem::offset_of!(GlmRopeParams, rotary_dim), 4);
        assert_eq!(std::mem::offset_of!(GlmRopeParams, in_stride), 8);
        assert_eq!(std::mem::offset_of!(GlmRopeParams, out_stride), 12);

        assert_eq!(std::mem::size_of::<GlmMlaAppendParams>(), 20);
        assert_eq!(std::mem::offset_of!(GlmMlaAppendParams, n_heads), 0);
        assert_eq!(std::mem::offset_of!(GlmMlaAppendParams, pos), 16);

        assert_eq!(std::mem::size_of::<GlmBuildQParams>(), 12);
        assert_eq!(std::mem::offset_of!(GlmBuildQParams, qk_rope), 8);

        assert_eq!(std::mem::size_of::<GlmDsaParams>(), 20);
        assert_eq!(std::mem::offset_of!(GlmDsaParams, pos), 12);
        assert_eq!(std::mem::offset_of!(GlmDsaParams, dim_scale), 16);

        assert_eq!(std::mem::size_of::<GlmTopkParams>(), 8);
        assert_eq!(std::mem::offset_of!(GlmTopkParams, k), 4);

        assert_eq!(std::mem::size_of::<GlmSortU32Params>(), 4);
        assert_eq!(std::mem::align_of::<GlmSortU32Params>(), 4);
        assert_eq!(std::mem::offset_of!(GlmSortU32Params, n), 0);

        assert_eq!(std::mem::size_of::<GlmSparseAttnParams>(), 24);
        assert_eq!(std::mem::offset_of!(GlmSparseAttnParams, n_allow), 16);
        assert_eq!(std::mem::offset_of!(GlmSparseAttnParams, scale), 20);

        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let one = f32_buffer(&ctx, &[1.0]);
        let mut tcb = TokenCommandBuffer::new(&ctx);
        let error = encode_rmsnorm(&mut tcb, &one, &one, &one, 2, 1e-6)
            .expect_err("undersized buffers must be rejected before encoding");
        assert!(error.to_string().contains("byte range"));
        assert_eq!(tcb.dispatch_count(), 0);
    }

    #[test]
    fn route_segment_reorder_is_exact_at_edges_and_after_tied_topk() {
        let shader = include_str!("../shaders/gravity_pq.metal");
        assert!(shader.contains("kernel void gravity_glm_sort_u32_ascending("));
        let registry = include_str!("metal/mod.rs");
        assert!(registry
            .contains("\"gravity_glm_sort_u32_ascending\" => \"gravity_glm_sort_u32_ascending\""));

        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let edge_sizes = [1usize, 2, 3, 31, 32, 33, 255, 256, 257, 1023, 2047, 2048];
        let mut fixtures = Vec::new();
        let mut tcb = TokenCommandBuffer::new(&ctx);

        let dummy = u32_buffer(&ctx, &[0]);
        encode_sort_positions_ascending(&mut tcb, &dummy, &dummy, 0)
            .expect("k=0 is an encode no-op");
        assert_eq!(tcb.dispatch_count(), 0);

        for &k in &edge_sizes {
            let mut input: Vec<u32> = (0..k as u32).collect();
            for index in 0..k {
                let peer = (index.wrapping_mul(37).wrapping_add(11)) % k;
                input.swap(index, peer);
            }
            if k == 3 {
                input = vec![32767, 0, 8192];
            }
            let input_buffer = u32_buffer(&ctx, &input);
            let output_buffer = if k == 33 {
                input_buffer.clone()
            } else {
                u32_buffer(&ctx, &vec![u32::MAX; k])
            };
            encode_sort_positions_ascending(&mut tcb, &input_buffer, &output_buffer, k)
                .expect("encode bounded ascending reorder");
            let mut expected = input.clone();
            expected.sort_unstable();
            fixtures.push((output_buffer, expected));
        }
        assert_eq!(tcb.dispatch_count(), edge_sizes.len());
        tcb.commit_and_wait().expect("edge reorder command buffer");
        for (output, expected) in fixtures {
            assert_eq!(
                read_u32(&output, expected.len()),
                expected,
                "GPU reorder must be exact"
            );
        }

        let oversized_input = u32_buffer(&ctx, &vec![0; 2049]);
        let oversized_output = u32_buffer(&ctx, &vec![0; 2049]);
        let mut oversized_tcb = TokenCommandBuffer::new(&ctx);
        let error = encode_sort_positions_ascending(
            &mut oversized_tcb,
            &oversized_input,
            &oversized_output,
            2049,
        )
        .expect_err("k above the flagship bound must fail before encode");
        assert!(error.to_string().contains("k <= 2048"));
        assert_eq!(oversized_tcb.dispatch_count(), 0);

        let tied_values = vec![3.0, 7.0, 7.0, -1.0, 7.0, 2.0, 9.0, 9.0, 0.0, 9.0];
        let k = 7usize;
        let values = f32_buffer(&ctx, &tied_values);
        let score_order = u32_buffer(&ctx, &vec![u32::MAX; k]);
        let selected = empty_u8_buffer(&ctx, tied_values.len());
        let ascending = u32_buffer(&ctx, &vec![u32::MAX; k]);
        let mut chain = TokenCommandBuffer::new(&ctx);
        encode_stable_topk(
            &mut chain,
            &values,
            &score_order,
            &selected,
            tied_values.len(),
            k,
        )
        .expect("encode tied stable top-k");
        encode_sort_positions_ascending(&mut chain, &score_order, &ascending, k)
            .expect("encode top-k reorder");
        assert_eq!(chain.dispatch_count(), 2);
        chain.commit_and_wait().expect("top-k reorder chain");

        let expected_score_order: Vec<u32> = topk_desc(&tied_values, k)
            .into_iter()
            .map(|index| index as u32)
            .collect();
        assert_eq!(read_u32(&score_order, k), expected_score_order);
        let mut expected_ascending = expected_score_order;
        expected_ascending.sort_unstable();
        assert_eq!(read_u32(&ascending, k), expected_ascending);
    }

    #[test]
    fn route_segment_residual_add_is_exact_alias_safe_and_fail_closed() {
        let shader = include_str!("../shaders/gravity_pq.metal");
        assert!(shader.contains("kernel void gravity_add_inplace_f32("));
        let registry = include_str!("metal/mod.rs");
        assert!(registry.contains("\"gravity_add_inplace_f32\" => \"gravity_add_inplace_f32\""));

        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let x: Vec<f32> = (0..257)
            .map(|index| ((index % 31) as i32 - 15) as f32 * 0.5)
            .collect();
        let y: Vec<f32> = (0..257)
            .map(|index| (((index * 7) % 29) as i32 - 14) as f32 * 0.25)
            .collect();
        let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| x + y).collect();
        let xb = f32_buffer(&ctx, &x);
        let yb = f32_buffer(&ctx, &y);

        let alias_values = vec![1.5, -2.0, 0.25, 4.0];
        let alias = f32_buffer(&ctx, &alias_values);
        let mut tcb = TokenCommandBuffer::new(&ctx);
        encode_residual_add_inplace(&mut tcb, &xb, &yb, x.len())
            .expect("encode residual add over a rounded grid");
        encode_residual_add_inplace(&mut tcb, &alias, &alias, alias_values.len())
            .expect("encode explicitly supported full-buffer alias");
        assert_eq!(tcb.dispatch_count(), 2);
        tcb.commit_and_wait().expect("residual add command buffer");
        assert_eq!(read_f32(&xb, x.len()), expected);
        assert_eq!(
            read_f32(&alias, alias_values.len()),
            vec![3.0, -4.0, 0.5, 8.0]
        );

        let one = f32_buffer(&ctx, &[1.0]);
        let mut rejected = TokenCommandBuffer::new(&ctx);
        encode_residual_add_inplace(&mut rejected, &one, &one, 0)
            .expect("zero elements are an encode no-op");
        assert_eq!(rejected.dispatch_count(), 0);

        let error = encode_residual_add_inplace(&mut rejected, &one, &one, 2)
            .expect_err("undersized buffers must fail before dispatch");
        assert!(error.to_string().contains("byte range"));
        assert_eq!(rejected.dispatch_count(), 0);

        let error = encode_residual_add_inplace(&mut rejected, &one, &one, u32::MAX as usize)
            .expect_err("rounded grid overflow must fail before dispatch");
        assert!(error
            .to_string()
            .contains("rounded Metal grid width overflow"));
        assert_eq!(rejected.dispatch_count(), 0);

        let oversized_n = (u32::MAX as usize)
            .checked_add(1)
            .expect("usize exceeds the Metal u32 ABI on supported hosts");
        let error = encode_residual_add_inplace(&mut rejected, &one, &one, oversized_n)
            .expect_err("oversized geometry must fail before dispatch");
        assert!(error.to_string().contains("does not fit the Metal u32 ABI"));
        assert_eq!(rejected.dispatch_count(), 0);
    }

    #[test]
    fn route_segment_norm_rope_copy_and_zero_match_host_and_f64() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };

        let x = vec![0.5, -1.25, 2.0, 0.75, -0.125, 3.5, -2.25];
        let weight = vec![0.75, -0.5, 1.25, 0.875, -1.5, 0.25, 2.0];
        let bias = vec![0.1, -0.2, 0.3, -0.4, 0.05, 0.125, -0.25];
        let xb = f32_buffer(&ctx, &x);
        let wb = f32_buffer(&ctx, &weight);
        let bb = f32_buffer(&ctx, &bias);
        let rms_out = filled_f32_buffer(&ctx, x.len(), f32::NAN);
        let affine_out = filled_f32_buffer(&ctx, x.len(), f32::NAN);

        let rope_input = vec![
            91.0, 92.0, 1.0, 2.0, 3.0, 4.0, // head 0, rotate tail
            81.0, 82.0, -1.5, 0.5, 2.5, -3.0, // head 1
        ];
        let cos = vec![0.875, -0.25];
        let sin = vec![0.125, 0.75];
        let rope_in = f32_buffer(&ctx, &rope_input);
        let cosb = f32_buffer(&ctx, &cos);
        let sinb = f32_buffer(&ctx, &sin);
        let rope_out = filled_f32_buffer(&ctx, 8, f32::NAN);

        let copy_src = f32_buffer(&ctx, &[10.0, 11.0, 12.0, 13.0, 14.0]);
        let copy_dst = filled_f32_buffer(&ctx, 7, -9.0);
        let zero_out = filled_f32_buffer(&ctx, 9, 7.0);

        let mut tcb = TokenCommandBuffer::new(&ctx);
        encode_rmsnorm(&mut tcb, &xb, &wb, &rms_out, x.len(), 1e-6).expect("encode rms");
        encode_layernorm_affine(&mut tcb, &xb, &wb, &bb, &affine_out, x.len(), 1e-6)
            .expect("encode affine layernorm");
        encode_rope_interleaved(
            &mut tcb, &rope_in, 2, &rope_out, 0, &cosb, &sinb, 2, 4, 6, 4,
        )
        .expect("encode GLM RoPE");
        encode_copy_tail(&mut tcb, &copy_src, &copy_dst, 1, 3, 3).expect("encode copy tail");
        encode_zero(&mut tcb, &zero_out, 9).expect("encode zero");
        assert_eq!(tcb.dispatch_count(), 5);
        tcb.commit_and_wait().expect("primitive command buffer");

        let mean_sq_f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv_f32 = 1.0 / (mean_sq_f32 + 1e-6).sqrt();
        let rms_host: Vec<f32> = x
            .iter()
            .zip(&weight)
            .map(|(v, w)| v * inv_f32 * w)
            .collect();
        let mean_sq_f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
        let inv_f64 = 1.0 / (mean_sq_f64 + 1e-6f64).sqrt();
        let rms_f64: Vec<f64> = x
            .iter()
            .zip(&weight)
            .map(|(&v, &w)| (v as f64) * inv_f64 * (w as f64))
            .collect();
        assert_v21_pair("rmsnorm", &rms_host, &read_f32(&rms_out, x.len()), &rms_f64);

        let mean_f32 = x.iter().sum::<f32>() / x.len() as f32;
        let var_f32 = x
            .iter()
            .map(|v| (v - mean_f32) * (v - mean_f32))
            .sum::<f32>()
            / x.len() as f32;
        let affine_inv_f32 = 1.0 / (var_f32 + 1e-6).sqrt();
        let affine_host: Vec<f32> = x
            .iter()
            .zip(&weight)
            .zip(&bias)
            .map(|((&v, &w), &b)| (v - mean_f32) * affine_inv_f32 * w + b)
            .collect();
        let mean_f64 = x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64;
        let var_f64 = x
            .iter()
            .map(|&v| {
                let d = v as f64 - mean_f64;
                d * d
            })
            .sum::<f64>()
            / x.len() as f64;
        let affine_inv_f64 = 1.0 / (var_f64 + 1e-6f64).sqrt();
        let affine_f64: Vec<f64> = x
            .iter()
            .zip(&weight)
            .zip(&bias)
            .map(|((&v, &w), &b)| (v as f64 - mean_f64) * affine_inv_f64 * w as f64 + b as f64)
            .collect();
        assert_v21_pair(
            "affine layernorm",
            &affine_host,
            &read_f32(&affine_out, x.len()),
            &affine_f64,
        );

        let mut rope_host = Vec::new();
        let mut rope_f64 = Vec::new();
        for head in 0..2 {
            let base = head * 6 + 2;
            rope_host.extend(rope_interleaved(&rope_input[base..base + 4], &cos, &sin));
            let src = &rope_input[base..base + 4];
            for i in 0..2 {
                rope_f64.push(
                    src[2 * i] as f64 * cos[i] as f64 - src[2 * i + 1] as f64 * sin[i] as f64,
                );
            }
            for i in 0..2 {
                rope_f64.push(
                    src[2 * i + 1] as f64 * cos[i] as f64 + src[2 * i] as f64 * sin[i] as f64,
                );
            }
        }
        assert_v21_pair(
            "interleaved RoPE",
            &rope_host,
            &read_f32(&rope_out, 8),
            &rope_f64,
        );

        assert_eq!(
            read_f32(&copy_dst, 7),
            vec![-9.0, -9.0, -9.0, 11.0, 12.0, 13.0, -9.0]
        );
        assert_eq!(read_f32(&zero_out, 9), vec![0.0; 9]);
    }

    #[test]
    fn route_segment_mla_build_and_index_append_match_host_exactly() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let (n_heads, qk_nope, qk_rope, v_dim, position) = (2usize, 2usize, 2usize, 2usize, 1usize);
        let qk = qk_nope + qk_rope;
        let kv = vec![1.0, 2.0, 11.0, 12.0, 3.0, 4.0, 13.0, 14.0];
        let k_rot = vec![21.0, 22.0];
        let kvb = f32_buffer(&ctx, &kv);
        let krb = f32_buffer(&ctx, &k_rot);
        let keys = filled_f32_buffer(&ctx, 3 * n_heads * qk, -99.0);
        let values = filled_f32_buffer(&ctx, 3 * n_heads * v_dim, -77.0);

        let q = vec![1.0, 2.0, 90.0, 91.0, 3.0, 4.0, 92.0, 93.0];
        let q_rot = vec![31.0, 32.0, 33.0, 34.0];
        let qb = f32_buffer(&ctx, &q);
        let qrb = f32_buffer(&ctx, &q_rot);
        let queries = filled_f32_buffer(&ctx, n_heads * qk, f32::NAN);

        let k_full = vec![41.0, 42.0, 43.0, 44.0];
        let kfb = f32_buffer(&ctx, &k_full);
        let index_keys = filled_f32_buffer(&ctx, 3 * k_full.len(), -55.0);

        let mut tcb = TokenCommandBuffer::new(&ctx);
        encode_mla_append_kv_expanded(
            &mut tcb, &kvb, &krb, &keys, &values, n_heads, qk_nope, qk_rope, v_dim, position,
        )
        .expect("encode expanded MLA append");
        encode_build_queries(&mut tcb, &qb, &qrb, &queries, n_heads, qk_nope, qk_rope)
            .expect("encode build queries");
        encode_append_index_key(&mut tcb, &kfb, &index_keys, 2, k_full.len())
            .expect("encode index-key append");
        assert_eq!(tcb.dispatch_count(), 3);
        tcb.commit_and_wait().expect("MLA primitive command buffer");

        let mut expected_keys = vec![-99.0; 3 * n_heads * qk];
        let mut expected_values = vec![-77.0; 3 * n_heads * v_dim];
        for head in 0..n_heads {
            let kv_base = head * (qk_nope + v_dim);
            let key_base = (position * n_heads + head) * qk;
            expected_keys[key_base..key_base + qk_nope]
                .copy_from_slice(&kv[kv_base..kv_base + qk_nope]);
            expected_keys[key_base + qk_nope..key_base + qk].copy_from_slice(&k_rot);
            let value_base = (position * n_heads + head) * v_dim;
            expected_values[value_base..value_base + v_dim]
                .copy_from_slice(&kv[kv_base + qk_nope..kv_base + qk_nope + v_dim]);
        }
        assert_eq!(read_f32(&keys, expected_keys.len()), expected_keys);
        assert_eq!(read_f32(&values, expected_values.len()), expected_values);

        let expected_queries = vec![1.0, 2.0, 31.0, 32.0, 3.0, 4.0, 33.0, 34.0];
        assert_eq!(read_f32(&queries, expected_queries.len()), expected_queries);
        assert_eq!(
            read_f32(&index_keys, 3 * k_full.len()),
            vec![-55.0, -55.0, -55.0, -55.0, -55.0, -55.0, -55.0, -55.0, 41.0, 42.0, 43.0, 44.0,]
        );
    }

    #[test]
    fn route_segment_dsa_topk_sparse_and_router_pass_v21_and_exact_ids() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let (n_keys, n_heads, head_dim) = (5usize, 2usize, 3usize);
        let q_full = vec![1.0, -0.5, 2.0, -1.5, 0.25, 0.75];
        let index_keys = vec![
            0.5, 1.0, -0.25, 2.0, -0.5, 1.0, 2.0, -0.5, 1.0, 0.125, 0.25, 0.5, -0.75, 1.5, -1.0,
        ];
        let head_weights = vec![0.75, -0.25];
        let dim_scale = (head_dim as f32).powf(-0.5);
        let dsa_position = n_keys - 2;
        let qfb = f32_buffer(&ctx, &q_full);
        let ikb = f32_buffer(&ctx, &index_keys);
        let hwb = f32_buffer(&ctx, &head_weights);
        let dsa_scores = filled_f32_buffer(&ctx, n_keys, f32::NAN);
        let topk_indices = u32_buffer(&ctx, &[u32::MAX; 3]);
        let selected = empty_u8_buffer(&ctx, n_keys);

        let qk_dim = 3usize;
        let v_dim = 2usize;
        let queries = vec![0.5, -1.0, 1.5, -0.75, 0.25, 2.0];
        let sparse_keys: Vec<f32> = (0..n_keys * n_heads * qk_dim)
            .map(|i| ((i as f32 * 0.173).sin() * 1.25) + 0.05)
            .collect();
        let sparse_values: Vec<f32> = (0..n_keys * n_heads * v_dim)
            .map(|i| ((i as f32 * 0.219).cos() * 0.75) - 0.1)
            .collect();
        let allow = vec![0u32, 2, 4];
        let queryb = f32_buffer(&ctx, &queries);
        let sparse_keyb = f32_buffer(&ctx, &sparse_keys);
        let sparse_valueb = f32_buffer(&ctx, &sparse_values);
        let allowb = u32_buffer(&ctx, &allow);
        let context = filled_f32_buffer(&ctx, n_heads * v_dim, f32::NAN);
        let sparse_scale = (qk_dim as f32).powf(-0.5);

        let logits = vec![-4.0, -0.25, 0.0, 1.25, 5.0, 0.75];
        let bias = vec![0.125, -0.05, 0.2, -0.125, 0.01, 0.3];
        let logitb = f32_buffer(&ctx, &logits);
        let biasb = f32_buffer(&ctx, &bias);
        let router_scores = filled_f32_buffer(&ctx, logits.len(), f32::NAN);
        let corrected = filled_f32_buffer(&ctx, logits.len(), f32::NAN);

        let mut tcb = TokenCommandBuffer::new(&ctx);
        encode_dsa_scores(
            &mut tcb,
            &qfb,
            &ikb,
            &hwb,
            &dsa_scores,
            n_keys,
            n_heads,
            head_dim,
            dsa_position,
            dim_scale,
        )
        .expect("encode DSA scores");
        encode_stable_topk(&mut tcb, &dsa_scores, &topk_indices, &selected, n_keys, 3)
            .expect("encode exact stable top-k");
        encode_sparse_attention_expanded_ascending_allow(
            &mut tcb,
            &queryb,
            &sparse_keyb,
            &sparse_valueb,
            &allowb,
            &context,
            n_heads,
            qk_dim,
            v_dim,
            n_keys,
            allow.len(),
            sparse_scale,
        )
        .expect("encode expanded sparse attention");
        encode_router_correction(
            &mut tcb,
            &logitb,
            &biasb,
            &router_scores,
            &corrected,
            logits.len(),
        )
        .expect("encode router correction");
        assert_eq!(tcb.dispatch_count(), 4);
        tcb.commit_and_wait()
            .expect("decision primitive command buffer");

        let mut dsa_host = vec![0.0f32; n_keys];
        let mut dsa_f64 = vec![0.0f64; n_keys];
        for key_index in 0..n_keys {
            if key_index > dsa_position {
                dsa_host[key_index] = f32::NEG_INFINITY;
                dsa_f64[key_index] = f64::NEG_INFINITY;
                continue;
            }
            let key = &index_keys[key_index * head_dim..(key_index + 1) * head_dim];
            let mut host_acc = 0.0f32;
            let mut authority_acc = 0.0f64;
            for head in 0..n_heads {
                let query = &q_full[head * head_dim..(head + 1) * head_dim];
                let host_dot = query.iter().zip(key).map(|(x, y)| x * y).sum::<f32>();
                host_acc += head_weights[head] * (host_dot * dim_scale).max(0.0);
                let authority_dot = query
                    .iter()
                    .zip(key)
                    .map(|(&x, &y)| x as f64 * y as f64)
                    .sum::<f64>();
                authority_acc +=
                    head_weights[head] as f64 * (authority_dot * dim_scale as f64).max(0.0);
            }
            dsa_host[key_index] = host_acc;
            dsa_f64[key_index] = authority_acc;
        }
        let dsa_device = read_f32(&dsa_scores, n_keys);
        assert_v21_pair(
            "DSA scores",
            &dsa_host[..=dsa_position],
            &dsa_device[..=dsa_position],
            &dsa_f64[..=dsa_position],
        );
        assert_eq!(dsa_device[dsa_position + 1], f32::NEG_INFINITY);
        let topk_device: Vec<usize> = read_u32(&topk_indices, 3)
            .into_iter()
            .map(|v| v as usize)
            .collect();
        assert_eq!(topk_device, topk_desc(&dsa_host, 3));
        assert_eq!(topk_device, topk_desc_f64(&dsa_f64, 3));
        assert!(
            topk_device.iter().position(|&index| index == 1)
                < topk_device.iter().position(|&index| index == 2),
            "the lower index must win the exact DSA score tie"
        );

        let mut sparse_host = vec![0.0f32; n_heads * v_dim];
        let mut sparse_f64 = vec![0.0f64; n_heads * v_dim];
        for head in 0..n_heads {
            let query = &queries[head * qk_dim..(head + 1) * qk_dim];
            let mut host_logits = Vec::new();
            let mut authority_logits = Vec::new();
            for &position in &allow {
                let position = position as usize;
                let key_base = (position * n_heads + head) * qk_dim;
                let key = &sparse_keys[key_base..key_base + qk_dim];
                host_logits
                    .push(query.iter().zip(key).map(|(x, y)| x * y).sum::<f32>() * sparse_scale);
                authority_logits.push(
                    query
                        .iter()
                        .zip(key)
                        .map(|(&x, &y)| x as f64 * y as f64)
                        .sum::<f64>()
                        * sparse_scale as f64,
                );
            }
            let host_best = host_logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut host_probs: Vec<f32> =
                host_logits.iter().map(|v| (v - host_best).exp()).collect();
            let host_total = host_probs.iter().sum::<f32>();
            for value in &mut host_probs {
                *value /= host_total;
            }
            let authority_best = authority_logits
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut authority_probs: Vec<f64> = authority_logits
                .iter()
                .map(|v| (v - authority_best).exp())
                .collect();
            let authority_total = authority_probs.iter().sum::<f64>();
            for value in &mut authority_probs {
                *value /= authority_total;
            }
            for (slot, &position) in allow.iter().enumerate() {
                let value_base = (position as usize * n_heads + head) * v_dim;
                for dim in 0..v_dim {
                    sparse_host[head * v_dim + dim] +=
                        host_probs[slot] * sparse_values[value_base + dim];
                    sparse_f64[head * v_dim + dim] +=
                        authority_probs[slot] * sparse_values[value_base + dim] as f64;
                }
            }
        }
        assert_v21_pair(
            "expanded sparse attention",
            &sparse_host,
            &read_f32(&context, sparse_host.len()),
            &sparse_f64,
        );

        let router_host: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
        let router_f64: Vec<f64> = logits
            .iter()
            .map(|&l| 1.0 / (1.0 + (-(l as f64)).exp()))
            .collect();
        assert_v21_pair(
            "router sigmoid",
            &router_host,
            &read_f32(&router_scores, logits.len()),
            &router_f64,
        );
        let corrected_host: Vec<f32> = router_host.iter().zip(&bias).map(|(s, b)| s + b).collect();
        let corrected_f64: Vec<f64> = router_f64
            .iter()
            .zip(&bias)
            .map(|(s, &b)| s + b as f64)
            .collect();
        assert_v21_pair(
            "router correction",
            &corrected_host,
            &read_f32(&corrected, logits.len()),
            &corrected_f64,
        );
    }

    #[test]
    fn sequence_scratch_covers_8k_boundary_and_32k_last_position() {
        let cases = [
            (8191usize, 8192usize),
            (8192usize, 16384usize),
            (32767usize, 32768usize),
        ];
        let mut host = HostSequenceScratch::new(64);
        let mut capacity = 64usize;

        for (position, expected_capacity) in cases {
            let need = position.checked_add(1).expect("fixture length");
            capacity = grown_sequence_capacity(capacity, need).expect("capacity growth");
            assert_eq!(capacity, expected_capacity);
            host.grow_preserving(capacity);

            let active =
                active_sequence_len(position, capacity, "test scratch").expect("position fits");
            assert_eq!(active, need);
            assert!(
                checked_sequence_bytes(active, std::mem::size_of::<f32>(), "test active")
                    .expect("active bytes")
                    <= checked_sequence_bytes(
                        capacity,
                        std::mem::size_of::<f32>(),
                        "test capacity"
                    )
                    .expect("capacity bytes")
            );

            // These are the exact four sequence-sized writes used by the
            // resident indexer/selection path.
            host.index_scores[active - 1] = position as f32;
            host.selection_indices[active - 1] = position;
            host.attention_allowed[active - 1] = 1;
            host.attention_scores[active - 1] = -(position as f32);
        }

        assert!(
            active_sequence_len(8192, 8192, "old fixed ActPool score buffer").is_err(),
            "the former fixed buffer must be recognized as too small at position 8192"
        );
    }

    #[test]
    fn host_scratch_growth_preserves_state_and_adequate_reserve_is_allocation_free() {
        let mut host = HostSequenceScratch::new(8192);
        host.index_scores[0] = 1.25;
        host.index_scores[8191] = -7.5;
        host.selection_indices[8191] = 4096;
        host.attention_allowed[8191] = 1;
        host.attention_scores[8191] = 3.75;

        host.grow_preserving(16384);
        assert_eq!(host.index_scores[0], 1.25);
        assert_eq!(host.index_scores[8191], -7.5);
        assert_eq!(host.selection_indices[8191], 4096);
        assert_eq!(host.attention_allowed[8191], 1);
        assert_eq!(host.attention_scores[8191], 3.75);

        host.index_scores[8192] = 9.5;
        host.grow_preserving(32768);
        assert_eq!(host.index_scores[8192], 9.5);

        let pointers = (
            host.index_scores.as_ptr(),
            host.selection_indices.as_ptr(),
            host.attention_allowed.as_ptr(),
            host.attention_scores.as_ptr(),
        );
        let capacities = (
            host.index_scores.capacity(),
            host.selection_indices.capacity(),
            host.attention_allowed.capacity(),
            host.attention_scores.capacity(),
        );
        host.grow_preserving(32768);
        assert_eq!(
            pointers,
            (
                host.index_scores.as_ptr(),
                host.selection_indices.as_ptr(),
                host.attention_allowed.as_ptr(),
                host.attention_scores.as_ptr(),
            ),
            "an adequate reserve must not replace any sequence workspace"
        );
        assert_eq!(
            capacities,
            (
                host.index_scores.capacity(),
                host.selection_indices.capacity(),
                host.attention_allowed.capacity(),
                host.attention_scores.capacity(),
            ),
            "an adequate reserve must not allocate more host capacity"
        );
    }

    #[test]
    fn reused_sequence_topk_matches_numeric_parity_oracle() {
        let mut values = vec![f32::NEG_INFINITY; 32768];
        values[0] = 2.0;
        values[8191] = 8.0;
        values[8192] = 8.0;
        values[16384] = -1.0;
        values[32767] = 7.0;
        let expected = topk_desc(&values, 4);
        let mut selection = vec![usize::MAX; values.len()];
        let pointer = selection.as_ptr();
        let capacity = selection.capacity();

        let actual = topk_desc_with_scratch(&values, 4, &mut selection).expect("scratch selection");
        assert_eq!(actual, expected);
        assert_eq!(actual[..2], [8191, 8192], "lower index wins a score tie");

        let again =
            topk_desc_with_scratch(&values, 4, &mut selection).expect("scratch selection reuse");
        assert_eq!(again, expected);
        assert_eq!(selection.as_ptr(), pointer);
        assert_eq!(selection.capacity(), capacity);
    }

    #[test]
    fn device_index_score_growth_copies_prior_state() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let mut scratch = SequenceScratch::new(&ctx, 8192).expect("initial scratch");
        scratch.host.index_scores[0] = 0.25;
        scratch.host.index_scores[8191] = -3.5;
        scratch
            .store_index_scores(8192)
            .expect("store initial score state");

        scratch.reserve(&ctx, 8193).expect("grow past 8K");
        assert_eq!(scratch.capacity, 16384);
        assert_eq!(scratch.device_score_len, 8192);
        let copied = read_f32(&scratch.index_scores_device, 8192);
        assert_eq!(copied[0], 0.25);
        assert_eq!(copied[8191], -3.5);

        let device_contents = scratch.index_scores_device.contents();
        scratch
            .reserve(&ctx, 16384)
            .expect("adequate reserve is a no-op");
        assert_eq!(scratch.index_scores_device.contents(), device_contents);
    }

    #[test]
    fn resident_growth_preserves_kv_index_keys_and_scores_through_32k_reserve() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let arch = tiny_arch();
        let qk = arch.qk_dim();
        let mut session = ResidentSession::new(&ctx, &arch, 8192).expect("initial session");

        {
            let cache = &session.layers[0];
            unsafe {
                *(cache.keys.contents() as *mut f32) = 1.0;
                *(cache.keys.contents() as *mut f32).add(8191 * qk + (qk - 1)) = 2.0;
                *(cache.values.contents() as *mut f32).add(8191) = 3.0;
                *(cache.index_keys.contents() as *mut f32).add(8191) = 4.0;
            }
        }
        session.sequence_scratch.host.index_scores[0] = 5.0;
        session.sequence_scratch.host.index_scores[8191] = 6.0;
        session
            .sequence_scratch
            .store_index_scores(8192)
            .expect("store 8K scores");
        session.seq_len = 8192;

        session
            .reserve(&ctx, &arch, 8193)
            .expect("grow beyond the old 8K limit");
        assert_eq!(session.layers[0].capacity, 16384);
        assert_eq!(session.sequence_scratch.capacity, 16384);
        {
            let cache = &session.layers[0];
            unsafe {
                assert_eq!(*(cache.keys.contents() as *const f32), 1.0);
                assert_eq!(
                    *(cache.keys.contents() as *const f32).add(8191 * qk + (qk - 1)),
                    2.0
                );
                assert_eq!(*(cache.values.contents() as *const f32).add(8191), 3.0);
                assert_eq!(*(cache.index_keys.contents() as *const f32).add(8191), 4.0);
            }
        }
        let scores = read_f32(&session.sequence_scratch.index_scores_device, 8192);
        assert_eq!(scores[0], 5.0);
        assert_eq!(scores[8191], 6.0);

        {
            let cache = &session.layers[0];
            unsafe {
                *(cache.keys.contents() as *mut f32).add(8192 * qk) = 7.0;
                *(cache.values.contents() as *mut f32).add(8192) = 8.0;
                *(cache.index_keys.contents() as *mut f32).add(8192) = 9.0;
            }
        }
        session.sequence_scratch.host.index_scores[8192] = 10.0;
        session
            .sequence_scratch
            .store_index_scores(8193)
            .expect("store post-8K score");
        session.seq_len = 8193;

        session
            .reserve(&ctx, &arch, 32768)
            .expect("reserve through position 32767");
        assert_eq!(session.layers[0].capacity, 32768);
        assert_eq!(session.sequence_scratch.capacity, 32768);
        {
            let cache = &session.layers[0];
            unsafe {
                assert_eq!(*(cache.keys.contents() as *const f32), 1.0);
                assert_eq!(
                    *(cache.keys.contents() as *const f32).add(8191 * qk + (qk - 1)),
                    2.0
                );
                assert_eq!(*(cache.index_keys.contents() as *const f32).add(8191), 4.0);
                assert_eq!(*(cache.keys.contents() as *const f32).add(8192 * qk), 7.0);
                assert_eq!(*(cache.values.contents() as *const f32).add(8192), 8.0);
                assert_eq!(*(cache.index_keys.contents() as *const f32).add(8192), 9.0);
            }
        }
        let scores = read_f32(&session.sequence_scratch.index_scores_device, 8193);
        assert_eq!(scores[0], 5.0);
        assert_eq!(scores[8191], 6.0);
        assert_eq!(scores[8192], 10.0);
    }
}
