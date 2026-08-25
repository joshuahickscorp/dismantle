"""Representation-native execution. FRONT D / P7 (steer S015 §19, §70, §71).

The steer's rule: NO DENSE REMATERIALIZATION BY DEFAULT. A compact Gravity
representation must feed computation directly. The failure mode it names is

    compact bytes -> reconstruct dense tensor -> conventional GEMM

which throws away most of the point of having a compact representation, because the
dense tensor is exactly the traffic the representation existed to avoid.

So this implements a matvec that reads ws_rtn_q4_g64 -- grouped absmax, 4 bits per
weight, group 64, one f16 scale per group, the representation the KernelPlanner
actually selected for model #2's routed experts -- straight out of its packed bytes,
against a control that dequantizes to f32 first and then does the same matvec.
"""
from __future__ import annotations

import numpy as np

GROUP = 64
BITS = 4
BOUND = (1 << (BITS - 1)) - 1          # 7; offset-binary around it


def pack_q4_g64(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grouped absmax to 4-bit offset binary, two weights per byte.

    Returns (packed uint8 [rows, cols/2], scales f16 [rows, cols/GROUP]).
    """
    rows, cols = w.shape
    assert cols % GROUP == 0, "columns must be a multiple of the group"
    g = w.reshape(rows, cols // GROUP, GROUP).astype(np.float32)
    amax = np.abs(g).max(axis=2)
    scale = np.where(amax > 0, amax / BOUND, 1.0).astype(np.float32)
    q = np.rint(g / scale[:, :, None]).clip(-BOUND, BOUND).astype(np.int32)
    q = (q + BOUND).astype(np.uint8)              # offset binary: 0..14
    q = q.reshape(rows, cols)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale.astype(np.float16)


def dequantize(packed: np.ndarray, scale: np.ndarray, cols: int) -> np.ndarray:
    rows = packed.shape[0]
    lo = (packed & 0x0F).astype(np.int32)
    hi = (packed >> 4).astype(np.int32)
    q = np.empty((rows, cols), dtype=np.int32)
    q[:, 0::2] = lo
    q[:, 1::2] = hi
    w = (q - BOUND).astype(np.float32) * np.repeat(
        scale.astype(np.float32), GROUP, axis=1)[:, :cols]
    return w


# One thread per output row. It walks the row's packed bytes, unpacking two weights
# per byte and accumulating -- the dense tensor is never written anywhere.
NATIVE_MATVEC = """
uint row = thread_position_in_grid.x;
if (row >= %(ROWS)du) return;
float acc = 0.0f;
uint pbase = row * %(PACKED_COLS)du;
uint sbase = row * %(GROUPS)du;
for (uint g = 0; g < %(GROUPS)du; ++g) {
    float s = (float)scales[sbase + g];
    uint c0 = g * %(GROUP)du;
    for (uint k = 0; k < %(GROUP)du; k += 2) {
        uchar byte = packed[pbase + (c0 + k) / 2u];
        float w0 = (float)((int)(byte & 0x0F) - %(BOUND)d) * s;
        float w1 = (float)((int)(byte >> 4)   - %(BOUND)d) * s;
        acc += w0 * x[c0 + k] + w1 * x[c0 + k + 1u];
    }
}
out[row] = acc;
"""

# The control: the same arithmetic, but reading a dense f32 tensor that some earlier
# kernel had to materialise.
DENSE_MATVEC = """
uint row = thread_position_in_grid.x;
if (row >= %(ROWS)du) return;
float acc = 0.0f;
uint base = row * %(COLS)du;
for (uint c = 0; c < %(COLS)du; ++c) acc += w[base + c] * x[c];
out[row] = acc;
"""

DEQUANT = """
uint idx = thread_position_in_grid.x;
if (idx >= %(ROWS)du * %(COLS)du) return;
uint row = idx / %(COLS)du;
uint col = idx %% %(COLS)du;
uchar byte = packed[row * %(PACKED_COLS)du + col / 2u];
int q = (col %% 2u == 0u) ? (int)(byte & 0x0F) : (int)(byte >> 4);
float s = (float)scales[row * %(GROUPS)du + col / %(GROUP)du];
out[idx] = (float)(q - %(BOUND)d) * s;
"""


def sources(rows: int, cols: int) -> dict[str, str]:
    d = {"ROWS": rows, "COLS": cols, "PACKED_COLS": cols // 2,
         "GROUPS": cols // GROUP, "GROUP": GROUP, "BOUND": BOUND}
    return {"native": NATIVE_MATVEC % d, "dense": DENSE_MATVEC % d,
            "dequant": DEQUANT % d}


def kernel_identity(rows: int, cols: int) -> str:
    """The identity of the emitted native kernel: a hash of the MSL itself.

    Two tensors share a kernel exactly when this matches. Not a similarity score,
    not a model-family heuristic -- the bytes that will be compiled.
    """
    import hashlib
    return hashlib.sha256(sources(rows, cols)["native"].encode()).hexdigest()[:16]


def stored_gate_shape(on_disk_shape: list[int]) -> tuple[tuple[int, int], bool]:
    """(gate shape as STORED, whether preparation is needed) for a MoE expert tensor.

    Two storage conventions are in the specimens on disk and they are NOT
    interchangeable:

      [out, in]            one 2-D tensor per expert per projection (Qwen3-30B-A3B,
                           Kimi-VL). Already in the orientation the kernel wants.

      [experts, in, 2*out] one 3-D tensor stacking every expert, with gate and up
                           FUSED along the last axis (Qwen3-VL-30B-A3B). The gate
                           half is stored [in, out] -- TRANSPOSED relative to what
                           the kernel wants -- so it needs de-interleaving and
                           transposing before the same kernel applies.

    This distinction is the reason kernel reuse is a claim about STORAGE LAYOUT and
    not only about GEMV shape: the fused tensor's stored orientation emits a
    DIFFERENT kernel from the one the same model's shape suggests.
    """
    if len(on_disk_shape) == 3:
        _, in_dim, two_out = on_disk_shape
        return (in_dim, two_out // 2), True
    out_dim, in_dim = on_disk_shape
    return (out_dim, in_dim), False


def accept_pack(w_true, packed, scale, cols, gpu_out, x, *,
                kernel_tol_rel: float = 1e-3,
                min_cosine: float = 0.99,
                magnitude_band: tuple[float, float] = (0.9, 1.1)) -> dict:
    """Is a packed tensor ACCEPTED? Two INDEPENDENT gates, both required.

    THIS EXISTS BECAUSE THE OBVIOUS PREDICATE CERTIFIES NOTHING. Comparing the GPU
    kernel against a numpy decode of THE SAME PACKED BYTES tests only that the
    kernel implements the representation. The original tensor never enters it, so
    an ALL-ZEROS PACK PASSES -- measured, not supposed: zeros, swapped nibbles and
    scales x1000 were all accepted by that predicate, at relative errors against the
    true tensor of 1.50, 1.54 and 1003.60.

    Gate 1, KERNEL FIDELITY: the kernel agrees with a decode of its own bytes.
    Necessary, and it is the ONLY thing the old predicate checked.

    Gate 2, REPRESENTATION FIDELITY: the decode resembles the TRUE tensor. Its
    tolerances are anchored on quantities THE PACK CANNOT INFLUENCE, which is what
    makes scaling or zeroing the pack fail rather than cancel.

    COSINE IS CHECKED WITH A MAGNITUDE BAND, NEVER ALONE. This campaign sealed that
    law on 2026-08-17 after an adequacy gate scored 0.01*W at 1.000000 on every axis
    for a whole campaign: cosine is SCALE-INVARIANT, so a pack whose scales are
    1000x too large points the right way and is catastrophically wrong. The band is
    the half of the check that notices.
    """
    import numpy as np
    w_true = np.asarray(w_true, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    decoded = np.asarray(dequantize(packed, scale, cols), dtype=np.float64)
    ref = decoded @ x
    true = w_true @ x
    got = np.asarray(gpu_out, dtype=np.float64)

    kernel_err = float(np.max(np.abs(got - ref)))
    kernel_tol = float(np.max(np.abs(ref))) * kernel_tol_rel
    kernel_ok = kernel_err <= kernel_tol

    nt, na = float(np.linalg.norm(true)), float(np.linalg.norm(ref))
    cos = float(np.dot(true, ref) / (nt * na)) if nt > 0 and na > 0 else 0.0
    ratio = (na / nt) if nt > 0 else float("inf")
    lo, hi = magnitude_band
    rep_ok = cos >= min_cosine and lo <= ratio <= hi

    return {
        "accepted": bool(kernel_ok and rep_ok),
        "kernel_fidelity": {"ok": bool(kernel_ok), "max_abs_err": kernel_err,
                            "tolerance": kernel_tol,
                            "means": "the kernel implements the representation"},
        "representation_fidelity": {
            "ok": bool(rep_ok), "cosine": cos, "magnitude_ratio": ratio,
            "min_cosine": min_cosine, "magnitude_band": list(magnitude_band),
            "means": "the representation resembles the tensor",
            "why_both": "cosine alone is scale-invariant; the band is what catches a "
                        "pack that points the right way at the wrong magnitude"},
        "relative_error_vs_true": float(np.max(np.abs(ref - true)) /
                                        max(1e-30, float(np.max(np.abs(true))))),
    }
