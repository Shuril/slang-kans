/**
 * GPU dequantization kernels for Slang-KANs.
 * Supports INT8, INT4, Ternary (1.58-bit), and INT2 codebook dequantization.
 */

module quantize;

[shader("compute")]
[numthreads(64, 1, 1)]
public void dequantize_int8(
    uint3 threadId : SV_DispatchThreadID,
    uniform StructuredBuffer<uint> packed_weight, // uint8 stored in uint buffer
    uniform StructuredBuffer<float> scales,
    uniform StructuredBuffer<float> biases,
    uniform RWStructuredBuffer<float> output,
    uniform int num_groups,
    uniform int group_size
) {
    uint idx = threadId.x;
    uint total = (uint)(num_groups * group_size);
    if (idx >= total) return;

    uint g = idx / (uint)group_size;
    float q = (float)packed_weight[idx];
    output[idx] = q * scales[g] + biases[g];
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void dequantize_int4(
    uint3 threadId : SV_DispatchThreadID,
    uniform StructuredBuffer<uint> packed_weight, // uint8 packed (2 nibbles)
    uniform StructuredBuffer<float> scales,
    uniform StructuredBuffer<float> biases,
    uniform RWStructuredBuffer<float> output,
    uniform int num_groups,
    uniform int group_size
) {
    uint idx = threadId.x;
    uint total = (uint)(num_groups * group_size);
    if (idx >= total) return;

    uint g = idx / (uint)group_size;
    uint byte_idx = idx / 2;
    uint byte_val = packed_weight[byte_idx];
    float q = (idx % 2 == 0) ? (float)(byte_val & 0x0F) : (float)((byte_val >> 4) & 0x0F);

    output[idx] = q * scales[g] + biases[g];
}

[shader("compute")]
[numthreads(16, 16, 1)]
public void dequantize_ternary_fast(
    uint3 threadId : SV_DispatchThreadID,
    uniform StructuredBuffer<uint> packed_weight, // uint32: 16 x 2-bit
    uniform StructuredBuffer<float> scales,        // [D_out, groups_per_row]
    uniform RWStructuredBuffer<float> output,      // [D_out, K_dim]
    uniform int D_out,
    uniform int K_dim,
    uniform int group_size
) {
    uint k = threadId.x;
    uint j = threadId.y;

    if (j >= (uint)D_out || k >= (uint)K_dim) return;

    uint words_per_row = ((uint)K_dim + 15) / 16;
    uint word_idx = k / 16;
    uint shift = (k % 16) * 2;

    uint word = packed_weight[j * words_per_row + word_idx];
    uint code = (word >> shift) & 0x03;

    float w_val = 0.0f;
    if (code == 0) w_val = -1.0f;
    else if (code == 2) w_val = 1.0f;

    uint g = k / (uint)group_size;
    uint groups_per_row = ((uint)K_dim + (uint)group_size - 1) / (uint)group_size;
    float sc = scales[j * groups_per_row + g];

    output[j * (uint)K_dim + k] = w_val * sc;
}

[shader("compute")]
[numthreads(16, 16, 1)]
public void dequantize_int2_fast(
    uint3 threadId : SV_DispatchThreadID,
    uniform StructuredBuffer<uint> packed_weight,
    uniform StructuredBuffer<float> scales,
    uniform StructuredBuffer<float> codebook,     // 4 floats
    uniform RWStructuredBuffer<float> output,
    uniform int D_out,
    uniform int K_dim,
    uniform int group_size
) {
    uint k = threadId.x;
    uint j = threadId.y;

    if (j >= (uint)D_out || k >= (uint)K_dim) return;

    uint words_per_row = ((uint)K_dim + 15) / 16;
    uint word_idx = k / 16;
    uint shift = (k % 16) * 2;

    uint word = packed_weight[j * words_per_row + word_idx];
    uint code = (word >> shift) & 0x03;

    float centroid = codebook[code];

    uint g = k / (uint)group_size;
    uint groups_per_row = ((uint)K_dim + (uint)group_size - 1) / (uint)group_size;
    float sc = scales[j * groups_per_row + g];

    output[j * (uint)K_dim + k] = centroid * sc;
}
