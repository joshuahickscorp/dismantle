# CP6a — real catalog weights, all 64 layers: 1.892x, and one instrument caught lying

Raw: `receipts/runtime/CP6A_REAL_SESSION_ORGAN.json`
Harness: `crates/hawking-core/examples/ascension_qwen38_cp6a_real_session_chunked_organ.rs`
Runtime: `measure_isolated_organ_chunked`, added beside `measure_isolated_organ`.

Every prior CP measured buffers this crate filled. `CP5C_RESULT.md` listed "real artifact weight
bytes at these shapes" as untested. This is that test: weights come from the session's own
catalog via `affine()`, all 64 layers are encoded, and each arm sits in one command buffer — the
same instrument that produced the production organ number. Zero loaded residents, 50.8 GB free,
noop control 208 ns.

## The headline the harness printed was wrong, and the roof says so

The harness reported **0.226x** against its fusion-matched pair baseline. That baseline is
invalid. `gate_up` across 64 layers is 3.565 GB of codes, scales and biases, and the machine's
measured roof is 778.8 GB/s, so **no arm that reads those weights can finish under 4.578 ms**:

| arm | GPU ns | positions | weights read | implied GB/s |
|---|---|---|---|---|
| fused production | 7,248,624 | 1 | 3.565 GB | 491.8 |
| chunked r4k4 | 15,323,125 | 4 | 3.565 GB | 232.7 |
| **pair baseline** | **866,874** | 1 | 3.565 GB | **4112.7 — 5.3x the roof** |

A number 5.3x above a measured physical ceiling is not a fast baseline, it is an arm that is not
doing the work. `encode_fused_gate_up(layer, with_swiglu=false)` on this catalog does not read
the weight set, and every ratio computed against it is meaningless. **The 0.226x is discarded.**

That the roof caught it is the point. Without an independent physical bound, 0.226x would have
read as a clean refutation of the whole campaign.

## What the measurement actually says

Against the SwiGLU-fused production baseline, which is the path that really runs:

    fused production   7,248,624 ns  ->  1 position,  3.565 GB read
    chunked r4k4      15,323,125 ns  ->  4 positions, 3.565 GB read

    per position:      7,248,624  ->  3,830,781 ns        1.892x
    per position:          3.565 GB  ->  0.891 GB         4x less traffic

**1.892x per position on real catalog weights across all 64 layers**, with the weight traffic cut
exactly 4x — which is the mechanism the whole campaign predicted: read the weights once, emit K
positions.

## Why it is 1.892x here and 2.49x in CP4

CP4 measured 2.49x on one layer's worth of gate_up at the same shape. Here the same kernel over
64 real layers gives 1.892x, and the bandwidth column says why: the chunked arm runs at **232.7
GB/s** against a 778.8 GB/s roof. It is no longer bandwidth-bound. Having cut traffic 4x it has
made itself compute- and register-bound, so the remaining win is smaller than the traffic
saving. CP4 saw the same shape — its r4k4 arm managed ~136 GB/s against a 219 GB/s baseline.

The production path at 491.8 GB/s is at 63% of roof; the chunked path at 232.7 GB/s is at 30%.
**The next lever on this organ is arithmetic efficiency, not more batching.**

## Caveats, stated rather than buried

* **This is a timing measurement only.** The isolated-organ instrument runs whatever the shared
  workspace holds, so it cannot check per-position outputs. Correctness for these kernels was
  established by CP4 — bit-identical at the real gate_up shape against the production affine
  matvec, with a negative control observed to reject — and CP6a does **not** re-establish it.
* **1.892x is an upper bound, not a matched comparison.** The production baseline fuses SwiGLU
  and the chunked kernel does not, so the baseline does slightly more work. The honest
  fusion-matched number needs a pair baseline that actually reads the weights, which is now a
  known defect rather than an assumption.
* Still a single organ. Not a prompt wall.

## What this changes on the frontier

The claim "batching wins on the real weights" is now measured rather than projected, at 1.892x
for the largest organ. The projection in `CP5_DECOMPOSITION.md` used 2.49x for gate_up; on real
layers that term should read 1.892x, which lowers the projected step-level gain and is the more
honest input for CP6 stage 1.

Two new items for the frontier, both found here:
1. `encode_fused_gate_up(..., with_swiglu=false)` does not read the weight set on this catalog.
   That is a live defect in a production-reachable code path, not just in a benchmark arm.
2. The chunked organ is compute-bound at 30% of roof. More positions will not help it; better
   arithmetic per weight will.
