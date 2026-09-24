/**
 * 16x16 shared-memory tiled GEMM kernel in Slang.
 * Computes C = A * B_mat^T + bias, where A is [M, K], B_mat is [N, K], C is [M, N].
 */

module gemm;

static const int TILE_SIZE = 16;
groupshared float sA[16][17];
groupshared float sB[16][17];

[shader("compute")]
[numthreads(16, 16, 1)]
public void gemm_tiled(
    uint3 groupThreadId : SV_GroupThreadID,
    uint3 groupId : SV_GroupID,
    uniform RWStructuredBuffer<float> C,
    uniform StructuredBuffer<float> A,
    uniform StructuredBuffer<float> B_mat,
    uniform StructuredBuffer<float> bias,
    uniform int M,
    uniform int N,
    uniform int K,
    uniform int has_bias
) {
    uint tx = groupThreadId.x;
    uint ty = groupThreadId.y;
    uint row = groupId.y * TILE_SIZE + ty;
    uint col = groupId.x * TILE_SIZE + tx;

    float acc = (has_bias != 0 && col < (uint)N) ? bias[col] : 0.0f;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        uint tiled_k_a = t * TILE_SIZE + tx;
        uint tiled_k_b = t * TILE_SIZE + ty;

        sA[ty][tx] = (row < (uint)M && tiled_k_a < (uint)K) ? A[row * K + tiled_k_a] : 0.0f;
        sB[ty][tx] = (col < (uint)N && tiled_k_b < (uint)K) ? B_mat[col * K + tiled_k_b] : 0.0f;

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += sA[ty][k] * sB[k][tx];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < (uint)M && col < (uint)N) {
        C[row * N + col] = acc;
    }
}
