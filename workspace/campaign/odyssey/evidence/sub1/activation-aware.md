## Why real-activation is the only path

Weight-space codecs code frozen \(W\). Activation-aware codes \(W\) **restricted to \(\mathrm{span}(X)\)**. That is a source change. NS-raw-weight-pq-vq-at-one-bit kills the former, not the latter.

| Trap | What happened | Why it fools you |
|---|---|---|
| Gaussian / synthetic \(X\) | v1 `functional_score` **output-side** probes with `rng.standard_normal` (`glm52_activation_aware_pack.py`). v2 **forbids** that path (`assert_no_gaussian_promotion_path`; `attention_o_proj` stays native rather than take a Gaussian). G016: every prior sub-bit negative traced to synthetic/gaussian \(X\). | Proxy \(X\) is ~isotropic. Real \(X\) is low-dim, mean-dominated, family-growing (G046/G047). The proxy spends bits on energy the model never uses. |
| Qwen3.8 gaussian-proxy sub-bit | Packet prior: all collapsed 0.5–0.8 BPW, output-div ~0.69. **Originating JSON not re-opened this turn — INFERRED-from-packet.** | Same class as v1 output-side Gaussian. |
| `beats_null` as admission | v1: lowest rank that beats the constant-mean null. Median surplus **+0.0266**, min **+0.0000**. 93.49% “success” was vacuous. G_math collapsed prompt-independently to argmax 50379 (`HEL`). | Null is a floor test (“better than the mean”), not a quality bar. Gate null is 0.949 — centered SVD **loses to the mean**. |
| Weight cosine / \(\|W-\hat W\|\) | Packer labels this `reconstruction_relative_error_INADMISSIBLE`. Centered r16 gate: W-rel-err 0.996 while function-cos is 0.87. | \(W\) is near-full-rank (G033: weight 99% energy needs 92.5% ranks; function-space 56%). Matching \(W\) wastes bits on unseen directions. |
| Raw PQ/VQ, AQLM, QTIP, QuIP# at ≤1 bit | F0 collapse 0.80–0.89 per-expert; F1 collapse **complete 1.0075 and 0.493**, 6/6. NS-post-hoc-coding: mid/late ~Gaussian, 0.059 decades from Shannon. G024: SLB **1.92 bpe** at q3 weight-MSE on Qwen3.8; sub-1 **excluded for memoryless weight coding**. | Those methods do not change the source. |
| Functional student / hypernet | GLM-5.2: local skill 0.53 @ 0.010 local BPW; residual stream **expansive at every magnitude** (2.41× at 0.77% L2). Cascade L74 skill 0.098 in 4 layers. | GENERATE-not-STORE fails under depth here. Not a NOW path. |
| Phase-B functional low-rank | Fit-set win, **held-out loss** vs q3 (0.39/0.31 vs 0.22). | Small captures overfit (G1 guard). |

**v2 laws to keep:** uncentered; route-conditioned, fail-closed on empty route; real SwiGLU \(X\) for down; Gaussian forbidden for promotion; `beats_null` diagnostic only; absolute floors; budget never lowers floor; native fallback billed at **source payload width**.

**Prior art → Hawking receipts**

- **GPTQ** Hessian \(H=X^\top X\) ≡ G069 (function-fit scale \(\sum d_j|r_j|/\sum d_j\)).
- **AWQ** channel scales ≡ allocate by \(d_j=\mathrm{diag}(H)\).
- **SqueezeLLM / OWQ / SpQR** outliers ≡ G074 FUNCTIONAL exceptions \(d_j(w-q(w))^2\), which beat MAGNITUDE and RESIDUAL at equal \(k\).
- **QuIP# / QuaRot / SpinQuant**: G032 block-Hadamard at q2, \(\Delta\cos\approx +0.003\) — negligible.
- **AQLM / QTIP / PQ**: weight-space; F1 + NS-raw-weight-pq-vq.
- **BitNet 1.58**: QAT changes the source (NS: untested, not blocked). PTQ ternary **dominated** VQ (NS-ternary-factorization).
- **Hypernetworks**: GLM-5.2 functional student (above).

---

## Precedent (do not inflate)

| Object | Rate | Quality | Class | Source |
|---|---:|---|---|---|
| GLM-5.2 `up_proj` e0, A1 AA rank-16 f16 | 0.16667 | 0.75515 cos vs null 0.651; beats_null | MEASURED organ screen, **not Doctor** | `GLM52_REAL_ACTIVATION_SWEEP.json` |
| Same tensor, raw-weight rank-16 | 0.1667 | 0.18876 cos; **fails** null | MEASURED | same |
| AA k16 / k64, 12 experts | 0.1667 / 0.6667 | 12/12 beat null; raw r64 = 0/12 | MEASURED | `GLM52_ACTIVATION_AWARE_REPLICATION.json` |
| GLM-5.2 high-traffic **gate** L5 E11, **uncentered** r16 f16 | 0.16671 | held-out cos **0.9924**, rel_out 0.099, beats null 0.949 | MEASURED | `GLM52_BASIS_PILOT_RECEIPT.json` |
| Same gate, **centered** r16 | 0.16671 | cos 0.870, **fails** null 0.949, rel_out 0.484 | MEASURED | same |
| Uncentered rank ladder, same gate | 0.167 / 0.667 / 1.33 / 2.67 | cos 0.992 / 0.997 / 0.998 / 0.999; rel_out 0.099 / 0.049 / 0.037 / 0.030 | MEASURED | same |
| GLM-5.2 v1 whole pack | 0.0538 on 93.49% of weights; **complete 1.0924** | G_math FAIL; 55389/55398 tensors <0.90 cos | MEASURED | `GLM52_BYTE_ATTRIBUTION.json`, `GLM52_GENERATION_B_CAPABILITY_VERDICT.json` |
| v1 pass-through | 6.51% of weights, **95.4% of bytes**, mostly gate/up at BF16 | — | MEASURED | BYTE_ATTRIBUTION |
| v2 all-routed r64 f16 ledger | complete **0.815** | non-authorizing lower bound | DERIVED | `GLM52_V2_PROGRAM_FEASIBILITY.md` |
| v2 all-routed r128 f16 | complete **1.303** | over 49/50; authorizing **for bytes only** | DERIVED | same |
| Transfer-share 459 bases | complete **0.599** | UNVALIDATED; must not authorize | HYPOTHESIS | same |
| Q80 mixed-1.5 screen | complete 1.4305 @ 8-bit NE | organ bar 0.8604; **not packed, not generated, no kernel** | MEASURED screen | `QWEN80_MIXED_REPRESENTATION_UNDER_1_5.json` |
| Q80 mixed-sub655 packed | complete 0.6462; expert 0.535 | mean component cos **0.424**; no kernel; no generation | MEASURED bytes, **not quality** | `mixed-sub655-v1/PACK_REPORT.json` |
| F1 raw PQ/VQ | complete 1.0075 and 0.493 | collapse 6/6 real forward | MEASURED | `f1_qwen3_235b.json` / NS-raw-weight-pq-vq |
| O005 conventional | 4.025 stored / 4.230 active | battery 10/12, \(\Delta 0\) | MEASURED | O005 packet |
| O005 routing | entropy 6.16/7, **0 cold**, top-16 mass 18% | — | MEASURED | same |

**NOW on Cycle-1:** organ-level AA screen is reachable. Doctor-valid whole-model sub-0.2 is **not claimed**. No AA pack exists on O001/O003/O004/O005/O006.

---

## Allocate the tiny bit budget by activation sensitivity

Law: **capability is the constraint; BPW is minimized subject to it** (GEN_B). Inverse (min BPW s.t. `beats_null`) is REFUTED.

### What the bits actually buy

| Lever | Measured effect | Source |
|---|---|---|
| Keep the mean (uncentered / explicit-\(\mu\)) vs centered | gate L5 E11 r16: cos 0.992 vs 0.870; median lift **+0.111** | BASIS_PILOT |
| Extra rank on uncentered (same tensor) | r16→r64 buys 0.099→0.049 rel_out; after that, crumbs | BASIS_PILOT |
| Function-fit vs weight-fit planes | gate L63 k2: 0.193 vs 0.226 rel_fro at 2.5 bpe | G069 |
| FUNCTIONAL exception vs MAGNITUDE @ 0.25 bpe | gate 14.3% vs 0.22% err drop; down 27.7% vs 0.72% | G074 |
| Profile mismatch vs pooled | mismatch mean 2.91% max 8.85%; pooled mean 0.44% max 0.92% | G081 |
| Organ inversion | gate/up Doctor-sensitive, down tolerant. **Necessary, not sufficient** (F1 A1_1p0 still collapsed). | F0/F1, NS-uniform-subbit-allocation |
| Capture family | L31: 79% of code energy outside prose span; unseen-dir gain 0.71–1.03× seen | G047 |
| O005 single-expert zero | \(\Delta\)hits = 0 (hot 49 and random 77) | Packet. Battery **too coarse** — not redundancy. |

### Spend order

1. **Mean direction** (1 uncentered column or explicit \(\mu\)). Centered SVD is forbidden for promotion.
2. **Absolute function floor** on held-out real \(X\) (v2: panel min 0.85 / median 0.96 high-traffic; per-tensor 0.91 low-traffic; router 0.99). Floor fail → native at **source width**. Never lower the floor to hit a BPW target.
3. **Organ inversion:** extra rank / exception bits to gate/up before down. Down only on **real post-SwiGLU \(X\)** (v2: output-side down is a forbidden production control).
4. Leftover bits as G074 FUNCTIONAL exceptions or G069 function-fit scale, **not** magnitude outliers.
5. **Route-condition** expert bases on routed rows (\(\ge 32\); else fail closed).
6. **Pooled capture** (prose+code+math+instruction+tool). Do not fit on prose and declare general.
7. Protect router / norms / embed / lm_head (O005 router 0.03 GB; zeroing any organ kills the battery).
8. Never entropy-code Lloyd PQ indices (NS: 0–0.7%).
9. Never share expert **weights** (NS-inter-expert cosine \(\sim 10^{-4}\) / Q80 0.004). Sharing the **activation basis** is a different object (unvalidated; M3).

### Geometry: 0.167 is GLM-5.2-shaped

Per-tensor billed basis, f16:

\[\mathrm{BPW}(r)=8\cdot(64+2r(\mathrm{in}+\mathrm{out}))/(\mathrm{in}\cdot\mathrm{out})\]

| Body | Organ | in×out | r16 f16 BPW | \(r\) for ~0.167 f16 | Class |
|---|---|---|---:|---:|---|
| GLM-5.2 expert | [2048, 6144] | 12.58M | **0.1667** | 16 | MEASURED |
| O005 expert | [768, 2048] | 1.57M | **0.4586** | \(r\approx 6\) → 0.172 | DERIVED from census |
| O003 expert | ~[1408, 2048] | 2.88M | 0.306 | \(r\approx 9\) | DERIVED |
| O001 dense MLP | [12288, 3072] | 37.7M | 0.104 | \(r\approx 26\) | DERIVED |
| O004 dense MLP | [32768, 5120] | 168M | 0.058 | \(r\approx 46\) | DERIVED |

Copying “rank-16 = 0.167” onto O005 is a category error. On O005, 0.167 f16 is ~rank 6; quality of r6 is **UNKNOWN** (pilot ranks start at 16). The rate path that **keeps** rank is factor-quant (M2).

### Stored vs active (O005, DERIVED)

Experts 28.991B (95% stored). Always-on (attn+embed+lm_head+router+norm) 1.541B. Selected experts 8/128 = 1.812B. Active 3.353B.

Always-on at 4-bit = 6.164B bits / 3.353B active params = **1.84 active BPW before any expert byte**. Complete **active** sub-1 on O005 is **impossible** while always-on stays at 4-bit.

| Expert stored BPW | Nonexpert BPW | Stored complete | Active complete |
|---:|---:|---:|---:|
| 0.167 | 4 | 0.359 | 1.93 |
| 0.459 (O005 r16 f16) | 4 | 0.635 | 2.09 |
| 0.667 (r64 f16 GLM-shaped) | 4 | 0.834 | 2.20 |
| 0.167 | 2 | 0.259 | 1.01 |
| 0.167 | 1.5 | 0.234 | 0.78 |

Stored sub-1 is the expert-side win. Active sub-1 needs attn/embed/head compressed. Report both. (NS-001: Q80 mixed-sub655 stored 0.646 → **active 2.518** because decode reads 10/512.)

---

## How far AA realistically reaches

| Rung | Stored complete | Active complete (MoE) | Quality bar | NOW? |
|---|---|---|---|---|
| Organ screen, uncentered, input-side, high-traffic, held-out \(X\) | GLM-shaped **0.17** (r16 f16); O005-shaped **0.14–0.46** | n/a | cos 0.99 on **captured** \(X\) | YES, cheap |
| Same + factor-quant r64 b3–b4 | GLM-shaped **0.08–0.17** (HYPOTHESIS rate; HGRAVS01 exists as r160_b3 on Q80 down) | n/a | UNKNOWN vs uncentered f16 | codec exists; not Doctor-gated |
| + shared layer basis (459 unique on GLM-5.2) | 0.599 informational | n/a | UNVALIDATED | do not authorize |
| Whole-model AA f16, absolute floors, native leftovers | 0.82–1.30 (v2 ledgers) | always-on bound | v1 G_math FAIL; v2 untraversed | GLM-5.2 pack failed; Cycle-1 unrun |
| Doctor-valid whole-model sub-0.2 | **UNKNOWN — not claimed** | O005: only if always-on \(\lesssim\) 1.5 bit | residual amplification; min-cosine tensors; capture coverage | NO |
| Dense (O004/O001) sub-1 stored=active | G016 Q80-recipe-on-Qwen38: **1.89** @ 4-bit non-MLP; **1.24** if attn compressed | = stored | G024 weight SLB ~1.92 bpe at q3-dist; AA beats that **only on \(\mathrm{span}(X)\)** | Qwen3.8 retired; O004 unrun |

**Binding limits (not the codec):** allocator law; min-cosine tensors (pilot math_live **min** still fails at r256); capture span \(\neq\) operator span (G047); residual-stream expansion (GLM-5.2 measured; Odyssey **UNKNOWN**); always-on organs for active BPW; no measured AA decode kernel (`decode_kernel_exists: false` on Q80 mixed; `load_engine` Unimplemented for qwen3moe); depth cascade of 0.10 rel_out × 48 layers **UNKNOWN**.

---

## Proposals

### M1 — AA-UNCENTERED-KEEP-MEAN

| Field | |
|---|---|
| **mechanism** | Uncentered (or explicit-\(\mu\) + residual) SVD of **real** route-conditioned \(X\); project \(W\) onto the leading subspace. Same 0.167, higher quality than v1 centered 0.755. |
| **complete_byte_accounting** | 64 B header + f16 \(L[out,r]\) + f16 \(B[in,r]\) (input-side). Optional: bill \(B\) once under `runtime_tables` (then M3). Native leftovers at **source payload width**. `complete_bpw = 8·Σ(header+coeff+basis+native+packaging)/n_weights`. |
| **stored_bpw** | GLM-shaped r16 = **0.1667** (MEASURED). O005-shaped r16 = **0.4586** (DERIVED). |
| **active_bpw** | Selected-expert fraction × expert stored + always-on. O005 4-bit always-on dominates (§ above). |
| **expected_reachable_bpw** | Quality-at-0.167: uncentered r16 already 0.992 cos on one gate (MEASURED). Lower at **this** mechanism: r8 f16 ≈ 0.083 GLM-shaped / 0.229 O005-shaped — quality **UNKNOWN**. |
| **quality_risk** | Residual amplification of 0.10 rel_out; G047 family mismatch; min-cosine tensors; cosine ≠ Doctor. Limiter: low-traffic gate/up; down if scored on Gaussian (forbidden). |
| **cheapest_falsifier** | O005 L0, 1 hot + 1 random expert, gate/up/down. Uncentered vs centered vs gaussian-proxy \(X\), \(\ge 1000\) routed tokens disjoint from holdout. **Kill** if uncentered does not beat centered by \(\ge 0.05\) held-out cos **and** proxy ranking disagrees with real-\(X\) ranking. |
| **execution_path** | Native skinny GEMV \(y=(x B)L^\top\). \(2(in+out)r\) FLOPs. **Forbid** expand-to-dense. Device cost/token **UNKNOWN**. |
| **applicability** | MoE expert input-side (O005/O006/O003/O010). Dense MLP (O001/O004) — every row has \(X\). Down only with real SwiGLU \(X\). Attn `o_proj` without real intermediate: native. |
| **confidence** | HIGH uncentered > centered on mean-dominated organs (MEASURED). LOW that r16 uncentered is Doctor-valid (G_math FAIL on a looser pack). |
| **transfer** | Uncentered law: expect TRANSFER. Numeric floors/rank: RETUNE per geometry. Not two-Qwen universal. |

### M2 — AA-FACTOR-QUANT-KEEP-RANK

| Field | |
|---|---|
| **mechanism** | Keep uncentered rank in the v2 quality band (r64–r128 routed, **not** r16). Store \(L,B\) quantized (HGRAVS01 / `MAGIC_ACT_SVD`). Honest path **below 0.167 at higher rank**. |
| **complete_byte_accounting** | Packed \(L\) + packed \(B\) (or shared \(B\)) + per-group scales/biases + header + native + packaging. Scales are f16 per group-\(g\) on the **factors**. Must sit in the ledger. |
| **stored_bpw** | GLM-shaped payload, no scales (DERIVED): r64 b4 = 0.167; r64 b3 = 0.125; r64 b2 = 0.083; r128 b2 = 0.167. Q80 down HGRAVS01 r160_b3 expert_bpw **1.27** (MEASURED, different geometry + scales). |
| **active_bpw** | Same split as § stored-vs-active. Packer already has `r64_b3, r128_b3, r192_b4, r256_b3`. |
| **expected_reachable_bpw** | Organ GLM-shaped **0.08–0.17** at r64–r128 b2–b4 (HYPOTHESIS rate; quality UNKNOWN). Whole-model complete UNKNOWN. Do not quote 0.08 as complete. |
| **quality_risk** | Quantizing factors can destroy the mean column (the M1 win). Q80: 11057/24576 down fits gram-rank-deficient. Use function-fit scales (G069), not absmax. |
| **cheapest_falsifier** | M1 r64 f16 as teacher on one O005 expert triple. Re-encode \(L,B\) at b4/b3/b2. **Kill** if held-out rel_out exceeds r16 uncentered f16 on **this** patient (re-measure; do not import 0.099). Second kill: fast-Doctor \(\Delta\)hits \(\le -2\) vs q3-g32-experts. |
| **execution_path** | Dequant then skinny GEMV, or fused qGEMV on \(r\) columns. **Not** expand-to-q4 (`qwen38_sub15_pack` confound). `MAGIC_ACT_SVD` exists in Python; Metal/NX does **not**. Kernel-bound runtime: fewer bytes help only if decode is efficient. |
| **applicability** | Same as M1. Dense down_proj is the HGRAVS01 home. |
| **confidence** | MEDIUM on rate arithmetic (DERIVED). LOW on quality at b2–b3. HGRAVS01 r160_b3 is a Q80-down screen (cos 0.886–0.898, bar 0.860), not a generation gate. |
| **transfer** | Codec is geometry-agnostic (ndarray in). Rank/bits RETUNE. Never-routed weight-space SVD fallback N/A on dense; N/A on O005 if 0-cold holds (MEASURED O005/O003/O006, short prompts). |

### M3 — AA-SHARED-LAYER-BASIS

| Field | |
|---|---|
| **mechanism** | One uncentered \(B\) per (layer, organ-class, rank), billed once in `runtime_tables`. Per-tensor stores only \(L\). Does **not** share expert weights (NS-inter-expert stays dead). |
| **complete_byte_accounting** | `runtime_tables += 64 + 2·width·r` per unique basis. Per tensor: `64 + 2·rows·r`. GLM-5.2 transfer-share: 459 unique bases, **56.42 GB, complete 0.599** (NON-AUTHORIZING). Target-local v2: 39,219 unique bases. |
| **stored_bpw** | Toward coefficient-only: GLM-shaped r16 \(L\)-only ≈ **0.0417** + amortized \(B\). |
| **active_bpw** | Still includes always-on. Shared \(B\) is read **per-token per-layer**, not per-expert. |
| **expected_reachable_bpw** | Stored complete **0.4–0.8** if overlap is high (HYPOTHESIS). 0.599 is arithmetic, not a measurement. |
| **quality_risk** | Route-conditioned \(B_e\) may not nest in a layer \(B\). G047 family energy. All-row \(B\) is the v2 **forbidden** fallback for routed experts. |
| **cheapest_falsifier** | O005 one layer, 16 experts. Route-conditioned \(B_e\) vs layer \(B\). **Kill** share if mean top-16 subspace overlap < 0.7 **or** held-out cos drop > 0.03. Cost: 16 SVDs. |
| **execution_path** | Same skinny GEMV; \(B\) is a layer-resident table. Stronger NX story than per-tensor \(B\). |
| **applicability** | MoE with shared hidden width. Dense: one \(B\) per layer already. Cross-layer: GLM52 `READOUT_IS_LAYER_SPECIFIC` — **do not** tie bases across depth. |
| **confidence** | MEDIUM that some share exists (same \(X\)). LOW that it survives route-conditioning + absolute floors. |
| **transfer** | UNVALIDATED by v2 law. Re-measure overlap. NS-cross-expert-tying does **not** auto-kill this (that kill is weight-space templates). |

### M4 — AA-FUNC-EXCEPTION-RESIDUAL

| Field | |
|---|---|
| **mechanism** | Uncentered AA base (M1/M2) + G074 FUNCTIONAL exceptions on the residual: pick \(k\) weights maximizing \(d_j(w-q(w))^2\), \(d_j\) from a disjoint half of \(X\). |
| **complete_byte_accounting** | Base ledger + exception index + values. Index: Rice/bitmap/group-local (Q80 residual encodings). Values q4–q8. **Count index+value+base.** Q80 rice_q1_rms @ 2% on up: expert_bpw 1.2918, 8.24 bits/outlier (MEASURED). G074: 0.25 bpe ≈ 0.64% of a gate tensor (50412 exceptions). |
| **stored_bpw** | Base + 0.25–1.0 bpe **only on tensors that fail the absolute floor**, not uniformly. Target: replace v1 BF16 pass-through (6.51% of weights, 95.4% of bytes). |
| **active_bpw** | Exceptions travel with the selected tensor. |
| **expected_reachable_bpw** | BYTE_ATTRIBUTION arithmetic: “even 4 BPW on failing gate/up → ~0.32 complete” (HYPOTHESIS). AA base + 0.25 bpe on the failing **subset**: complete **UNKNOWN**. G074 is a quality lever, not automatically a complete-BPW lever. |
| **quality_risk** | G074 is one tensor, one layer, Qwen3.8. Exceptions inherit capture bias (G079 prose-heavy). Sparse repairs may not contract residual-stream expansion. |
| **cheapest_falsifier** | One O005 gate that fails uncentered r16 floor. MAGNITUDE vs RESIDUAL vs FUNCTIONAL at matched exception **bytes**. **Kill** FUNCTIONAL if held-out improvement vs MAGNITUDE < 2×. |
| **execution_path** | Skinny GEMV + scatter-add of \(k\) exceptions. If exceptions force a dense gather, the exec story dies — account FLOPs. |
| **applicability** | Organs that miss AA floors (v1: 2681 gate + 669 up pass-throughs on GLM-5.2). Dense and MoE. |
| **confidence** | HIGH FUNCTIONAL > MAGNITUDE at equal \(k\) on Qwen3.8 MLP (MEASURED). MEDIUM it rescues GLM-style gate pass-throughs. LOW on Doctor. |
| **transfer** | \(d_j\) is patient+workload specific (G081). Re-capture. Do not ship prose-\(d_j\) into a code/math Doctor. |

### M5 — AA-ABS-FLOOR-DOCTOR-ALLOC

| Field | |
|---|---|
| **mechanism** | Allocator: absolute held-out function floor **and** a one-organ Doctor probe, then min rate. `beats_null` diagnostic only. `BudgetFailure` never reduces floor (v2 `select_program_or_native`). This **raises** complete_bpw vs v1. That is the correct direction. |
| **complete_byte_accounting** | v2 slots: f16 basis + f16 coeff + headers + `native_source_payload` + packaging. Native at source width. v2 r64: native **29.82 GB of 76.75 GB** (38.9%) — leftovers dominate if floors have teeth. |
| **stored_bpw** | GLM-5.2 shaped **0.82–1.30** (DERIVED ledgers). Not sub-0.2. O005: UNKNOWN (no ledger). |
| **active_bpw** | Leftover always-on at source/4-bit still bind. |
| **expected_reachable_bpw** | **Above** v1’s 1.09 if floors bite; **below** O005 q3 4.03 if most expert mass clears r64-class AA. O005 stored **0.8–2.5** HYPOTHESIS. Sub-0.2 complete: **not this mechanism alone**. |
| **quality_risk** | Leftover native mass. Floors were transferred from a Llama-1B calibration — re-measure. 12-prompt battery is coarse. |
| **cheapest_falsifier** | Pack **one** O005 layer, all 128 experts, v2 floors, no `beats_null` admission. Fast-Doctor vs bf16/q3. **Kill** if \(\Delta\)hits \(\le -2\) **or** \(\ge 20\%\) of expert bytes fall through to native. |
| **execution_path** | Mixed: AA skinny GEMV + native affine/q3 on leftovers. Router protected. |
| **applicability** | All Cycle-1. **Mandatory** before any sub-0.2 claim. |
| **confidence** | HIGH this allocator is the correct law (GEN_B MEASURED). MEDIUM O005 expert mass clears 0.85 min-cos at r64. LOW complete_bpw lands under 1. |
| **transfer** | Law transfers; floors RETUNE. |

### M6 — AA-ORGAN×dⱼ-RANKMAP

| Field | |
|---|---|
| **mechanism** | Per organ × per channel, set rank and exception budget from (Doctor-organ sensitivity) \(\times\) (activation energy \(d_j\)). Gate/up: more ranks + FUNCTIONAL exceptions. Down: fewer ranks, real SwiGLU \(X\). Tiny-\(d_j\) channels: rank 1 (mean) or drop-to-scale. Not uniform sub-bit (DEAD). Not routing-frequency on O005 (near-uniform, 0 cold). |
| **complete_byte_accounting** | Sum of M1/M2/M4 ledgers over a heterogeneous rank map. Report per-organ stored and active. |
| **stored_bpw** | UNKNOWN until a **rate** map exists. O005 organ sensitivity is zero-ablation (any organ → 0/12) plus near-identity round8 on a 4-bit specimen — not a bit-sensitivity map. |
| **active_bpw** | UNKNOWN. |
| **expected_reachable_bpw** | Policy allows component floors to 0.01. Honest complete: **stored 0.5–1.5** on MoE experts + 4-bit always-on if maps are peaked. Not sub-0.2 complete without M2+M3. |
| **quality_risk** | Battery cannot see single-expert damage. G079: top-1% locality prose-vs-code 0.74 vs control 0.91. Layer 0 is a different source (LIVE). |
| **cheapest_falsifier** | O005, layers {0, mid, last}, 4 experts. Held-out rel_out vs rank for gate/up/down. **Kill** organ-inversion-on-AA if gate and down curves lie within 10% (R-organ-inversion `reopen_if`). |
| **execution_path** | Per-tensor rank prefix of a shared uncentered basis (compatible with M8). One kernel, variable \(r\). |
| **applicability** | MoE + dense. Hybrid O001: MLP yes; SSM/state organs are a different object. |
| **confidence** | HIGH gate \(\neq\) down (F0/F1). LOW O005 rank maps are peaked enough to buy a full BPW. |
| **transfer** | Organ inversion confirmed on 2 parents. Rank maps PATIENT-SPECIFIC. |

### M7 — AA-SKINNY-GEMV-NATIVE

| Field | |
|---|---|
| **mechanism** | Make AA the **executable** object: fused \(y=(x B)L^\top\) (optionally q-factors). A tiny stored object that expands to dense \(W\) is a fake win (G17 / qwen38_sub15 expand-to-Q4). |
| **complete_byte_accounting** | Resident = encoded factors + scales + \(B\) tables + kernel constants. Working set \(\approx\) resident, not dense \(W\) (`dense_w_materialized=0` is the G17 pass). Per-token DRAM **measured**, not selected/full × stored. |
| **stored_bpw** | Same as the representation it executes. |
| **active_bpw** | Measured bytes/token. NS-001 forbids treating storage average as active. |
| **expected_reachable_bpw** | Not a rate mechanism. It **authorizes** M1–M4 as NX rather than NR. Without it, sub-0.2 stored is not a Hawking win. |
| **quality_risk** | Kernel-bound runtime (O005 mlx 35 tps SPECIMEN; gather is the MoE lever). Skinny r=16 may lose occupancy vs q3 grouped GEMV. G072: k=1/2/3 plane GEMV 0.606 / 0.668 / 0.987 ps/elem; saturated by k=2. |
| **cheapest_falsifier** | Microbench one O005 expert gate, r16 and r64, f16 skinny vs materialize-then-GEMV vs mlx q3. **Kill** NX claim if skinny DRAM \(\neq\) factor bytes **or** skinny token_ns > q3 at equal quality. |
| **execution_path** | Metal/NX fused kernel TBD (`HIDE_KERNEL_TURN` still false on GLM v2). Specimen: two mlx matmuls, labeled SPECIMEN. MoE: gather selected \((L,B)\), not the inventory. |
| **applicability** | All AA packs. Mandatory companion to M1–M4. |
| **confidence** | HIGH expand-to-dense is a false win. UNKNOWN whether skinny beats q3 on M3 Ultra. |
| **transfer** | Kernel RETUNE per width (O005 768×2048 vs GLM 2048×6144 vs O004 32768×5120). |

### M8 — AA-MATRYOSHKA-UNCENTERED

| Field | |
|---|---|
| **mechanism** | Nested uncentered prefixes: T0 = mean + small rank (organ-level sub-0.2 **draft**), T1 = r16, T2 = r64, T3 = M4 exceptions. Decode a prefix of \(B\) and \(L\). Count stored tiers and active tiers separately. |
| **complete_byte_accounting** | Stored = nested union (not additive if prefixes share). Active = bytes of the highest tier **hit this token**. Headers per tier. Do not claim T0 rate while T3 is always loaded. |
| **stored_bpw** | \(\approx\) T2/T3 complete, **0.7–1.3** HYPOTHESIS. |
| **active_bpw** | T0 organ GLM-shaped could be r8 f16 **0.083** or r16 b4 **0.042** (DERIVED, organ only). Whole-model active still always-on-bound. |
| **expected_reachable_bpw** | T0 organ sub-0.2: YES as a draft body. Doctor-valid T0: UNKNOWN. Phase-B hybrid q3+correction **generalizes but adds bytes** — same risk. |
| **quality_risk** | T0 is the G_math failure mode unless floors apply **per tier**. Do not ship T0 as the model. |
| **cheapest_falsifier** | O005 one expert, decode r8 vs r16 vs r64 prefixes of the **same** uncentered basis. **Kill** nesting if r16 error is not \(\le\) r8 (must be monotone). Then fast-Doctor T0-only vs T0+T1; drop T1 storage if T1 never restores a hit. |
| **execution_path** | One kernel, extra columns. Clean NX story. |
| **applicability** | All. Fits `matryoshka_tiers` but **activation-aware prefixes**, not affine q-tiers. |
| **confidence** | HIGH uncentered prefixes are monotone in fit (SVD). MEDIUM they stay monotone held-out. LOW T0 is Doctor-valid. |
| **transfer** | Prefix monotonicity should transfer; T0 rate RETUNE with geometry. |

---

## NOW on current patients (cheapest first)

1. **Proxy-kill (M1 falsifier)** on O005 L0 — must precede any rate claim. If gaussian-proxy ranking matches real-\(X\), this lane is wrong on this patient.
2. Uncentered vs centered rank ladder, gate/up/down, 1 layer, \(\ge 1000\) routed tokens, disjoint holdout.
3. Factor-quant r64 b4 vs r16 f16 at matched bytes (M2).
4. Basis overlap, 16 experts (M3).
5. One-layer v2-floor pack + fast-Doctor (M5).
6. Skinny vs materialize microbench (M7) — SPECIMEN, not `BASE_TRUE_TPS`.

Do not start from whole-model 0.167. Do not use `beats_null` admission. Do not Gaussian-probe down.

---

## UNKNOWN (not guessed)

- Originating JSON for Qwen3.8 “output-div ~0.69”.
- AA rank–quality curves on O001/O003/O004/O005/O006.
- Odyssey residual-stream amplification spectrum.
- Doctor at any AA rate on Cycle-1.
- Device token_ns of skinny AA GEMV.
- Quality of \(r<16\) uncentered and of b2–b3 factors on the mean column.
- Whether shared route-conditioned bases survive 0.85 min-cos.
- Always-on (attn/embed/head) AA without real intermediates.
- Sub-0.2 **complete** whole-model on any current patient.

---

## Guardrails

| Guardrail | How honored |
|---|---|
| Complete accounting | Every mechanism names header + \(L\) + \(B\) + scales + exceptions + native-at-source-width + packaging. Nominal \(\neq\) complete. v1 0.0538-on-93% is complete **1.0924**. |
| Activation-aware only | No raw-weight PQ/VQ reopen. No Gaussian promotion path. Down only on real SwiGLU \(X\). |
| Doctor is authority | Cosine/null are screens. v1 G_math FAIL is the cautionary artifact. Fast-Doctor is the kill in M2/M5/M8. A gibberish sub-0.2 pack is still useful if it localizes the broken organ. |
| Native execution | Skinny GEMV, no dense expand. M7 is mandatory. Wrapper TPS is not a win. |
| Stored vs active | Tables separate them. O005 active sub-1 is impossible at 4-bit always-on. 16× gather (8/128) is an exec lever, not a compression stat. |

---

## Completion report

**Lane:** ACTIVATION-AWARE-EXTREME  
**Proposals:** M1–M8 in the packet schema  
**Headline:** 0.167 is real as an organ screen; uncentered already lifts the same rate from 0.755 (centered up_proj) / 0.870 (centered gate) to **0.992** on a high-traffic gate. Whole-model v1 at complete 1.0924 **failed Doctor** because `beats_null` admitted noise. Below 0.167 at higher quality is **factor-quant of a higher rank** (M2) and/or **shared activation basis** (M3), never cheaper centered rank.  
**NOW:** organ screen yes; Doctor-valid sub-0.2 no; O005 active sub-1 no at 4-bit always-on.  
**Files:** packet is this message. A worktree write of `evidence/sub1/ACTIVATION_AWARE_EXTREME.md` was started and cancelled after the writer stalled; harvest from this text.
