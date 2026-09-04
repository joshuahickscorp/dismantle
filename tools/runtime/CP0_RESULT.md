# R3 CP0/CP1/CP2 — chunked prefill for the gated delta rule

## The recurrence, read off the kernel

`shaders/qwen_next.metal::qwen_next_gated_delta_decode_single`, per head:

```
S <- d*S ;  kv = S^T k ;  delta = (v - kv)*b ;  S <- S + k (x) delta ;  out = S^T q
```

which is affine in S:

```
S_t = A_t S_{t-1} + B_t
A_t = d_t (I - b_t k_t k_t^T)     K x K, scaled rank-1 update of I
B_t = b_t k_t v_t^T               K x V, rank one
```

Affine maps compose, so T positions collapse to one `(A, B)` and the K x V
state is touched once per chunk instead of T times.

## CP1/CP2 — proven numerically, not argued

`tools/runtime/chunked_delta_reference.py`, numpy, under a second:

| check | result |
|---|---|
| token-step vs affine composition, T = 1,2,3,8,32,64,128 | max rel 3.05e-15 |
| `prod(A_t)` vs its WY form `I - K^T W` | max rel 1.11e-14 |
| two unequal chunks composed at the boundary, splits 1/7/16/63 | max rel 3.13e-15 |

Machine precision. The composition closes exactly.

## Operation classification

| operation | class |
|---|---|
| RMSNorm, q/k/v/ba projections, output projection, MLP gate/up/down | BATCHABLE (GEMM over T) |
| causal conv, kernel 4 | BATCHABLE with a 3-position halo |
| GQA layers (16 of 64) | BATCHABLE, ordinary blocked attention |
| `prod(A_t)` via WY | STATE_COMPOSITION, short recurrence over T rows of length K |
| the single state apply `A S + B` | STATE_COMPOSITION, once per chunk |
| nothing found | STRICTLY_SEQUENTIAL |

## Where the win comes from — and where it does NOT

**Not FLOPs.** Chunking is FLOP-neutral to slightly worse: measured ratio 0.44x
at T=16, 1.33x at T=128, because the WY build is O(T^2 K). Anyone selling this
as an arithmetic saving is wrong.

**Bandwidth.** The model is DENSE -- `qwen38_geometry.rs` refuses MoE keys
outright -- so every position re-reads the full weight set. Per layer that is
`H*H*2 + H*I*3` with H=5120, I=17408: about 20.5 GB per position across 64
layers at one byte per parameter.

The measured prefill was 17,048 positions in 595 s, implying roughly 349 TB of
weight traffic. At the machine's measured 778.8 GB/s that traffic alone needs
**448 s against 595 s observed -- 75% of peak.** That is a bandwidth-bound loop,
and a GEMM over T positions reads those weights once instead of T times.

## Projected

Prefill is 75% of accepted-unit model wall (595 s of 799 s), at 28.7 prompt tok/s.

| prefill speedup | prefill wall | model wall | prompt tok/s |
|---|---|---|---|
| 1x (today) | 595 s | 799 s | 28.7 |
| 4x | 149 s | 353 s | 114.6 |
| 8x | 74 s | 278 s | 229.2 |

## Next rungs

CP3 one organ one layer physical, CP4 measured prompt throughput against the
sequential baseline, CP5 multi-layer, CP6 full resident prompt path, CP7 complete
WorkUnit wall. Acceptance stays capability-equivalence AND lower complete prompt
wall -- not "chunked code runs".
