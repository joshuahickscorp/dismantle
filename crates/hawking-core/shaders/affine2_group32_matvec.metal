// HGRAVF01 affine-2 GEMV (codec: unsigned 2-bit + fp16 scale + bias).
//
// Reconstruction (exact, unsigned q, scale may be negative):
//   q = (word >> (2*i)) & 3
//   w = float(q) * scale + bias
//
// Same kernel family as group-32. group_size is a bind-time parameter
// and must be 32 or 64. Codes stay packed. The GEMV dequants in
// registers and never writes a dense W.
//
// Host widens on-disk f16 scale+bias to IEEE f32 before bind so the
// kernel matches the CPU oracle bit-for-bit (q is exact; scale/bias are
// the widened storage values).
//
// Do not confuse per-group `biases` with an nn.Linear output bias.

#include <metal_stdlib>
using namespace metal;

constant uint AFFINE2_CODES_PER_WORD = 16u;

static inline float affine2_w(uint q, float scale, float bias) {
    return float(q) * scale + bias;
}

static inline uint affine2_code(uint word, uint i) {
    return (word >> (2u * i)) & 3u;
}

static inline bool affine2_group_ok(uint group_size, uint cols) {
    return (group_size == 32u || group_size == 64u) && (cols % group_size) == 0u;
}

// Eight consecutive weights (16 packed bits) inside a group.
// `col` is 8-aligned and sits inside one group.
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
// Walks 16-code words left to right inside each group.
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
    if (!affine2_group_ok(group_size, cols)) {
        output[row] = 0.0f;
        return;
    }
    const uint groups_per_row = cols / group_size;
    const uint words_per_group = group_size / AFFINE2_CODES_PER_WORD;
    const uint packed_row = row * (cols / AFFINE2_CODES_PER_WORD);
    float sum = 0.0f;
    for (uint g = 0u; g < groups_per_row; ++g) {
        const uint rgb = row * groups_per_row + g;
        const float scale = scales[rgb];
        const float bias = biases[rgb];
        const uint word_base = packed_row + g * words_per_group;
        for (uint w = 0u; w < words_per_group; ++w) {
            const uint word = codes[word_base + w];
            const uint col = g * group_size + w * AFFINE2_CODES_PER_WORD;
            for (uint i = 0u; i < AFFINE2_CODES_PER_WORD; ++i) {
                sum += affine2_w(affine2_code(word, i), scale, bias) * input[col + i];
            }
        }
    }
    output[row] = sum;
}

// Compile-time group 32/64 so col/GS is a shift. A runtime group_size on
// this path is a non-constant divide (see qwen_uniform_q4 group-128 note).
static inline float affine2_geo_acc_g32(
    device const uchar* codes,
    device const float* scales,
    device const float* biases,
    device const float* input,
    uint row,
    uint cols,
    uint lane_in_row)
{
    const uint groups_per_row = cols >> 5u;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
        const uint group = col >> 5u;
        const uint local = col & 31u;
        const uint rgb = row * groups_per_row + group;
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 8u + (local >> 2u))));
        acc += affine2_unpack8(packed16, scales[rgb], biases[rgb], input, col);
    }
    return acc;
}

static inline float affine2_geo_acc_g64(
    device const uchar* codes,
    device const float* scales,
    device const float* biases,
    device const float* input,
    uint row,
    uint cols,
    uint lane_in_row)
{
    const uint groups_per_row = cols >> 6u;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * groups_per_row + group;
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        acc += affine2_unpack8(packed16, scales[rgb], biases[rgb], input, col);
    }
    return acc;
}

// geo_tpr64 occupancy (same thread map as HGRAVU01 geo, different reconstruction).
//   TG 128, 4 simdgroups, 2 rows/TG, 64 threads/row
//   col = lane_in_row * 8, stride 512
//   grid threadgroups = ceil(rows/2), tg = 128
// Every 8-wide tile sits inside one group of 32 or 64. In-register dequant only.
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
    if (row < rows && affine2_group_ok(group_size, cols)) {
        if (group_size == 32u) {
            acc = affine2_geo_acc_g32(codes, scales, biases, input, row, cols, lane_in_row);
        } else {
            acc = affine2_geo_acc_g64(codes, scales, biases, input, row, cols, lane_in_row);
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

// Diagnostic: pre-specialization G0 body with runtime col/group_size.
kernel void affine2_group32_matvec_geo_tpr64_tg128_runtime_div(
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
    if (row < rows && affine2_group_ok(group_size, cols)) {
        const uint groups_per_row = cols / group_size;
        const uint bytes_per_group = group_size >> 2u;
        for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
            const uint group = col / group_size;
            const uint local = col % group_size;
            const uint rgb = row * groups_per_row + group;
            const float scale = scales[rgb];
            const float bias = biases[rgb];
            const uint byte0 = rgb * bytes_per_group + (local >> 2u);
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
    if (!affine2_group_ok(group_size, cols)) {
        output[gid] = 0.0f;
        return;
    }
    const uint row = gid / cols;
    const uint col = gid - row * cols;
    const uint groups_per_row = cols / group_size;
    const uint group = col / group_size;
    const uint local = col % group_size;
    const uint rgb = row * groups_per_row + group;
    const uint words_per_group = group_size / AFFINE2_CODES_PER_WORD;
    const uint packed_index =
        row * (cols / AFFINE2_CODES_PER_WORD) + group * words_per_group + (local / AFFINE2_CODES_PER_WORD);
    const uint word = codes[packed_index];
    const uint q = affine2_code(word, local & 15u);
    output[gid] = affine2_w(q, scales[rgb], biases[rgb]);
}

// ── Q2F: 4-level LS-fitted 2-bit, group 64, delta only ──
// w = (float(q) - 1.5) * delta. Host widens f16 delta to f32 before bind
// (same as affine2 standalone). Compile-time group 64; no bias buffer.

static inline float q2f_w(uint q, float delta) {
    return (float(q) - 1.5f) * delta;
}

static inline float q2f_unpack8_f32(
    uint packed16,
    float delta,
    device const float* x,
    uint col)
{
    float sum = 0.0f;
    for (uint i = 0u; i < 8u; ++i) {
        const uint q = (packed16 >> (2u * i)) & 3u;
        sum += q2f_w(q, delta) * x[col + i];
    }
    return sum;
}

static inline float q2f_geo_acc_g64_f32(
    device const uchar* codes,
    device const float* deltas,
    device const float* input,
    uint row,
    uint cols,
    uint lane_in_row)
{
    const uint groups_per_row = cols >> 6u;
    float acc = 0.0f;
    for (uint col = lane_in_row * 8u; col + 8u <= cols; col += 512u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * groups_per_row + group;
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        acc += q2f_unpack8_f32(packed16, deltas[rgb], input, col);
    }
    return acc;
}

kernel void q2f_group64_matvec(
    device const uchar* codes  [[buffer(0)]],
    device const float* deltas [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    constant uint& cols        [[buffer(5)]],
    uint row                    [[thread_position_in_grid]])
{
    if (row >= rows || (cols % 64u) != 0u) {
        return;
    }
    const uint groups_per_row = cols >> 6u;
    float sum = 0.0f;
    for (uint col = 0u; col + 8u <= cols; col += 8u) {
        const uint group = col >> 6u;
        const uint local = col & 63u;
        const uint rgb = row * groups_per_row + group;
        const uint packed16 = uint(*((device const ushort*)(codes + rgb * 16u + (local >> 2u))));
        sum += q2f_unpack8_f32(packed16, deltas[rgb], input, col);
    }
    output[row] = sum;
}

kernel void q2f_group64_matvec_geo_tpr64_tg128(
    device const uchar* codes  [[buffer(0)]],
    device const float* deltas [[buffer(1)]],
    device const float* input  [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint& rows        [[buffer(4)]],
    constant uint& cols        [[buffer(5)]],
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
    if (row < rows && (cols % 64u) == 0u) {
        acc = q2f_geo_acc_g64_f32(codes, deltas, input, row, cols, lane_in_row);
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

// Reconstruction into dense f32. Parity harness only; GEMV never writes this.
kernel void q2f_group64_dequant(
    device const uchar* codes  [[buffer(0)]],
    device const float* deltas [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant uint& rows        [[buffer(3)]],
    constant uint& cols        [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]])
{
    const uint n = rows * cols;
    if (gid >= n || (cols % 64u) != 0u) {
        return;
    }
    const uint row = gid / cols;
    const uint col = gid - row * cols;
    const uint groups_per_row = cols >> 6u;
    const uint group = col >> 6u;
    const uint local = col & 63u;
    const uint rgb = row * groups_per_row + group;
    const uint byte = uint(codes[rgb * 16u + (local >> 2u)]);
    const uint q = (byte >> (2u * (local & 3u))) & 3u;
    output[gid] = q2f_w(q, deltas[rgb]);
}
