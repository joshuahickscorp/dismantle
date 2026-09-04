# CP6e -- the first interleaved kernel, and a correction to CP6_SCOPE

2026-09-04 · frontier G019 · no resident loaded

## The correction

`CP6_SCOPE.md` classified the norms and elementwise kernels on the chunked path
as **"trivially K-parallel -- launching K x the threads over a K-wide buffer is
the whole change."**

That is **WRONG**, and it was found by trying to build the chunked MLP rather
than by re-reading the document.

The multi-position matmul kernels require **interleaved** activations,
`input[col * K + k]`. Their inner loop reads K contiguous floats at
`(col + i) * K` -- one coalesced 16-byte load at K=4 that feeds all R rows.
A blocked layout `[k * cols + col]` would turn that into K strided 4-byte loads
and hand back the traffic saving batching just bought. **Interleaving is a
performance requirement, not an incidental choice.**

Therefore a per-position kernel cannot be reused by binding it at a byte offset:
offsets address a BLOCKED layout. Every elementwise and norm kernel on the
chunked path needs its own strided variant.

The remaining stage-1 cost is larger than CP6_SCOPE claimed. It is not 7 launch
changes; it is 7 kernels.

## What was built

`qwen38_residual_rmsnorm_tg_interleaved` in
`crates/hawking-core/shaders/qwen80_device_activations.metal`. One threadgroup
per position, `threadgroup_position_in_grid` IS k, reduction over that position's
stride alone. The norm weight is per-hidden-dim and shared across positions, so
it stays un-interleaved -- only activations are strided.

## Result

Harness `ascension_qwen38_cp6e_interleaved_rmsnorm`. Bare `MetalContext`, no
resident, seconds to run -- cheap enough to keep as a standing gate.

```
  K=1  EQUIVALENCE PASS  max_abs 0.000e0
  K=2  EQUIVALENCE PASS  max_abs 0.000e0
  K=2  ISOLATION   PASS  bled 0 of 5120 (target moved: true)
  K=4  EQUIVALENCE PASS  max_abs 0.000e0
  K=4  ISOLATION   PASS  bled 0 of 15360 (target moved: true)
  K=8  EQUIVALENCE PASS  max_abs 0.000e0
  K=8  ISOLATION   PASS  bled 0 of 35840 (target moved: true)
```

**Bit-identical**, not merely close, against the production per-position kernel
at every K. Same arithmetic, same order, same threadgroup width, so anything
other than 0.000e0 would mean the reduction changed shape.

## Mutation check -- and why both checks exist

Reverting the load-bearing stride in the reduction read
(`input[index * chunk + k]` to `input[index * chunk]`):

```
  K=1  EQUIVALENCE PASS   <- correct: at K=1 the paths ARE identical
  K=2  EQUIVALENCE FAIL  max_abs 4.190e-2
  K=4  EQUIVALENCE FAIL  max_abs 4.190e-2
  K=8  EQUIVALENCE FAIL  max_abs 4.190e-2
  K=2  ISOLATION   PASS
```

Two things this establishes:

1. The check bites, and it bites with the **correct discrimination** -- K=1 must
   still pass, because at one position the interleaved and blocked layouts are
   byte-identical. A mutation check that failed K=1 too would mean the harness
   was testing the wrong thing.

2. **ISOLATION stayed PASS under a mutation that EQUIVALENCE caught.** The two
   checks are independent, not redundant. Dropping `+ k` from the reduction
   corrupts the scale while the output write still carries `+ k`, so positions
   remain isolated. A kernel that instead reduced over the whole interleaved
   buffer would be self-consistent and would fail ISOLATION alone. Each check
   covers a defect class the other misses.

Marker `MUTATION_CP6E` grepped to 0 occurrences after restore; rebuilt and
re-run green.

## Standing

One of the 7. `mlp_down`'s residual add, the swiglu when unfused, and the
mixer-side norms remain. The gate_up and down matmuls already exist (CP4, CP4b).
