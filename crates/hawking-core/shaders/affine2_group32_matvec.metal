// HGRAVF01 affine-2 group-32 GEMV (codec: unsigned 2-bit + fp16/bf16 scale + bias).
//
// On-disk MLX QuantizedLinear (bits=2, group_size=32, mode=affine):
//   weight  uint32 [N, K/16]   16 LSB-first 2-bit codes per word
//   scales  f16/bf16 [N, K/32] one scale per group of 32
//   biases  f16/bf16 [N, K/32] one affine intercept per group of 32
//
// Reconstruction (exact, unsigned q, scale may be negative):
//   q = (word >> (2*i)) & 3
//   w = float(q) * scale + bias
//
// Codes stay packed. The GEMV dequants in registers and never writes a dense W.
// Host widens on-disk f16/bf16 scale+bias to IEEE f32 before bind so the
// kernel matches the CPU oracle bit-for-bit (q is exact; scale/bias are the
// widened storage values).
//
// Do not confuse per-group `biases` with an nn.Linear output bias.

#include <metal_stdlib>
using namespace metal;

constant uint AFFINE2_GROUP = 32u;
constant uint AFFINE2_CODES_PER_WORD = 16u;

static inline float affine2_w(uint q, float scale, float bias) {
    return float(q) * scale + bias;
}

static inline uint affine2_code(uint word, uint i) {
    return (word >> (2u * i)) & 3u;
}

// One group of 32: two uint32 words, one (scale, bias).
static inline float affine2_group_dot(
    uint word0,
    uint word1,
    float scale,
    float bias,
    device const float* x,
    uint col)
{
    float sum = 0.0f;
    for (uint i = 0u; i < AFFINE2_CODES_PER_WORD; ++i) {
        sum += affine2_w(affine2_code(word0, i), scale, bias) * x[col + i];
    }
    for (uint i = 0u; i < AFFINE2_CODES_PER_WORD; ++i) {
        sum += affine2_w(affine2_code(word1, i), scale, bias) * x[col + 16u + i];
    }
    return sum;
}

// Eight consecutive weights (16 packed bits) inside a group of 32.
// `col` is 8-aligned and sits inside one group (`col % 32` in {0,8,16,24}).
static inline float affine2_unpack8(
    uint packed16,
    float scale,
    float bias,
    device const float* x,
    uint col)
{
    float sum = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        const uint q = (packed16 >> (2u * i)) & 3u;
        sum += affine2_w(q, scale, bias) * x[col + i];
    }
    return sum;
}

// Serial family. One thread per output row. Grid (rows,1,1), TG up to 256.
// Same left-to-right association as the CPU oracle: for each group, codes
// 0..15 from word0 then 16..31 from word1.
kernel void affine2_group32_matvec(
    device const uint*  codes  [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    constant uint& group_size  [[buffer(7)]],
    uint row                    [[thread_position_in_grid]])
{
    if (row >= rows) return;
    if (group_size != AFFINE2_GROUP || (cols & 31u) != 0u) {
        output[row] = 0.0f;
        return;
    }
    const uint groups_per_row = cols >> 5u;
    const uint packed_row = row * (cols >> 4u);
    float sum = 0.0f;
    for (uint g = 0u; g < groups_per_row; ++g) {
        const uint rgb = row * groups_per_row + g;
        const float scale = scales[rgb];
        const float bias = biases[rgb];
        const uint word0 = codes[packed_row + (g << 1u)];
        const uint word1 = codes[packed_row + (g << 1u) + 1u];
        sum += affine2_group_dot(word0, word1, scale, bias, input, g * AFFINE2_GROUP);
    }
    output[row] = sum;
}

// geo_tpr64 occupancy (same thread map as HGRAVU01 geo, different reconstruction).
//   TG 128, 4 simdgroups, 2 rows/TG, 64 threads/row
//   col = lane_in_row * 8, stride 512
//   grid threadgroups = ceil(rows/2), tg = 128
// Every 8-wide tile sits inside one group of 32. In-register dequant only.
kernel void affine2_group32_matvec_geo_tpr64_tg128(
    device const uchar* codes  [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* biases [[buffer(2)]],
    device const float* input  [[buffer(3)]],
    device float*       output [[buffer(4)]],
    constant uint& rows        [[buffer(5)]],
    constant uint& cols        [[buffer(6)]],
    constant uint& group_size  [[buffer(7)]],
    uint group_id               [[threadgroup_position_in_grid]],
    uint simd_lane              [[thread_index_in_simdgroup]],
    uint simd_id                [[simdgroup_index_in_threadgroup]])
{
    threadgroup float red[4];
    constexpr uint kSplit = 2u;
    const uint team = simd_id / kSplit;
    const uint split = simd_id % kSplit;
    const uint lane_in_row = split * 32u + simd_lane;
    const uint row = group_id * 2u + team;
    float acc = 0.0f;
    if (row < rows && group_size == AFFINE2_GROUP && (cols & 31u) == 0u) {
        const uint groups_per_row = cols >> 5u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col >> 5u;
            const uint local = col & 31u;
            const uint rgb = row * groups_per_row + group;
            const float scale = scales[rgb];
            const float bias = biases[rgb];
            const uint byte0 = rgb * 8u + (local >> 2u);
            const uint packed16 = uint(*((device const ushort*)(codes + byte0)));
            acc += affine2_unpack8(packed16, scale, bias, input, col);
        }
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        red[simd_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split == 0u && simd_lane == 0u && row < rows) {
        output[row] = red[team * kSplit] + red[team * kSplit + 1u];
    }
}

// Reconstruct w = q*scale + bias into a dense f32 buffer. Used only by the
// standalone parity harness to prove the in-register decode; the GEMV path
// above never materializes this buffer.
kernel void affine2_group32_dequant(
    device const uint*  codes  [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* biases [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    constant uint& cols        [[buffer(5)]],
    constant uint& group_size  [[buffer(6)]],
    uint gid                    [[thread_position_in_grid]])
{
    const uint n = rows * cols;
    if (gid >= n) return;
    if (group_size != AFFINE2_GROUP || (cols & 31u) != 0u) {
        output[gid] = 0.0f;
        return;
    }
    const uint row = gid / cols;
    const uint col = gid - row * cols;
    const uint groups_per_row = cols >> 5u;
    const uint group = col >> 5u;
    const uint local = col & 31u;
    const uint rgb = row * groups_per_row + group;
    const uint packed_index = row * (cols >> 4u) + (group << 1u) + (local >> 4u);
    const uint word = codes[packed_index];
    const uint q = affine2_code(word, local & 15u);
    output[gid] = affine2_w(q, scales[rgb], biases[rgb]);
}
