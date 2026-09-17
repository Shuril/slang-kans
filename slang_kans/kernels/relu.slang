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
[numthreads(16, 16, 1)]
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
    uint j = threadId.y;

    if (b >= (uint)batch_size || j >= (uint)out_features)
        return;

    float acc = bias[j];
    float h = (grid_max - grid_min) / max((float)(num_grids - 1), 1.0f);
    float inv_h = 1.0f / max(h, 1e-6f);

    for (int i = 0; i < in_features; i++)
    {
        float x_val = input[b * in_features + i];
        int w_base = (i * out_features + (int)j) * num_grids;

        for (int c = 0; c < num_grids; c++)
        {
            float center = grid_min + (float)c * h;
            float basis_val = relu_tent_basis(x_val, center, inv_h);
            acc += basis_val * weights[w_base + c];
        }
    }

    output[b * out_features + j] = acc;
}
