/**
 * ==============================================================================
 *  RELU.SLANG - Piecewise Linear Tent Basis KAN Kernel with Autodiff
 * ==============================================================================
 *  Computes y_j = bias_j + sum_{i=0}^{d_in-1} sum_{c=0}^{G-1} max(0, 1 - |x_i - c| * inv_h) * w_{i, j, c}
 * ==============================================================================
 */

module relu;

[Differentiable]
public float relu_tent_basis(float x, float center, float inv_h)
{
    float diff = abs(x - center) * inv_h;
    return max(0.0f, 1.0f - diff);
}

[shader("compute")]
[numthreads(16, 4, 1)]
public void relu_kan_forward(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> output,     // [B, D_out]
    uniform StructuredBuffer<float> input,       // [B, D_in]
    uniform StructuredBuffer<float> weights,     // [D_in, D_out, num_grids]
    uniform StructuredBuffer<float> bias,        // [D_out]
    uniform int batch_size,
    uniform int in_features,
    uniform int out_features,
    uniform int num_grids,
    uniform float grid_min,
    uniform float grid_max
)
{
    uint b = threadId.x;
    uint j_base = threadId.y * 4;

    if (b >= (uint)batch_size)
        return;

    float acc0 = (j_base + 0 < (uint)out_features) ? bias[j_base + 0] : 0.0f;
    float acc1 = (j_base + 1 < (uint)out_features) ? bias[j_base + 1] : 0.0f;
    float acc2 = (j_base + 2 < (uint)out_features) ? bias[j_base + 2] : 0.0f;
    float acc3 = (j_base + 3 < (uint)out_features) ? bias[j_base + 3] : 0.0f;

    float h = (grid_max - grid_min) / max((float)(num_grids - 1), 1.0f);
    float inv_h = 1.0f / max(h, 1e-6f);

    for (int i = 0; i < in_features; i++)
    {
        float x_val = input[b * in_features + i];

        float basis_val[16];
        for (int c = 0; c < 16; c++) basis_val[c] = 0.0f;

        for (int c = 0; c < num_grids; c++)
        {
            float center = grid_min + (float)c * h;
            basis_val[c] = relu_tent_basis(x_val, center, inv_h);
        }

        if (j_base + 0 < (uint)out_features) {
            int w0 = (i * out_features + (int)j_base + 0) * num_grids;
            for (int c = 0; c < num_grids; c++) acc0 += basis_val[c] * weights[w0 + c];
        }
        if (j_base + 1 < (uint)out_features) {
            int w1 = (i * out_features + (int)j_base + 1) * num_grids;
            for (int c = 0; c < num_grids; c++) acc1 += basis_val[c] * weights[w1 + c];
        }
        if (j_base + 2 < (uint)out_features) {
            int w2 = (i * out_features + (int)j_base + 2) * num_grids;
            for (int c = 0; c < num_grids; c++) acc2 += basis_val[c] * weights[w2 + c];
        }
        if (j_base + 3 < (uint)out_features) {
            int w3 = (i * out_features + (int)j_base + 3) * num_grids;
            for (int c = 0; c < num_grids; c++) acc3 += basis_val[c] * weights[w3 + c];
        }
    }

    if (j_base + 0 < (uint)out_features) output[b * out_features + j_base + 0] = acc0;
    if (j_base + 1 < (uint)out_features) output[b * out_features + j_base + 1] = acc1;
    if (j_base + 2 < (uint)out_features) output[b * out_features + j_base + 2] = acc2;
    if (j_base + 3 < (uint)out_features) output[b * out_features + j_base + 3] = acc3;
}
