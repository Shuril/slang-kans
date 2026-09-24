/**
 * Tensor core KAN layer using linalg::CoopMat for 16x16 matrix multiply-accumulate.
 * Supports NVIDIA Tensor Cores, AMD Matrix Cores, and Apple Silicon simdgroup_matrix.
 */

import linalg;

// 16x16 Cooperative Matrix Tile fragments
using MatrixA   = linalg::CoopMat<half,  16, 16, linalg::MatrixUse::A>;
using MatrixB   = linalg::CoopMat<half,  16, 16, linalg::MatrixUse::B>;
using MatrixAcc = linalg::CoopMat<float, 16, 16, linalg::MatrixUse::Accumulator>;

[shader("compute")]
[numthreads(32, 1, 1)] // 1 Warp (32 threads) cooperative execution
public void tensor_core_kan_gemm(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> output_matrix, // [M, N]
    uniform StructuredBuffer<half> basis_matrix,     // [M, K]
    uniform StructuredBuffer<half> weight_matrix,    // [K, N]
    uniform int M_dim,
    uniform int N_dim,
    uniform int K_dim
)
{
    // Warp-level Cooperative Matrix Fragment accumulation
    MatrixA a_frag;
    MatrixB b_frag;
    MatrixAcc c_frag = MatrixAcc(0.0f);

    // Accumulate tiles along inner dimension K in chunks of 16
    for (int k = 0; k < K_dim; k += 16)
    {
        // Load basis and weight matrix tiles into tensor core registers
        a_frag.load(basis_matrix, int2(k, 0));
        b_frag.load(weight_matrix, int2(0, k));

        // Single-cycle hardware MMA instruction
        c_frag = linalg::coopMatMulAdd(a_frag, b_frag, c_frag);
    }

    // Store final accumulated result tile
    c_frag.store(output_matrix, int2(0, 0));
}
