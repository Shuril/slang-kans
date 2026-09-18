/**
 * ==============================================================================
 *  BSPLINE.SLANG - Cubic B-Spline Cox-de Boor KAN Kernel with Autodiff
 * ==============================================================================
 */

module bspline;

/**
 * Computes cubic B-spline basis values for normalized input u in [0, 1].
 */
[Differentiable]
public void compute_cubic_bspline_4(float u, out float b[4])
{
    // Standard uniform cubic B-spline polynomial basis functions on [0, 1]
    float u2 = u * u;
    float u3 = u2 * u;

    b[0] = (1.0f - u) * (1.0f - u) * (1.0f - u) / 6.0f;
    b[1] = (3.0f * u3 - 6.0f * u2 + 4.0f) / 6.0f;
    b[2] = (-3.0f * u3 + 3.0f * u2 + 3.0f * u + 1.0f) / 6.0f;
    b[3] = u3 / 6.0f;
}

[shader("compute")]
[numthreads(16, 4, 1)]
public void bspline_kan_forward(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> output,
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> weights,
    uniform StructuredBuffer<float> bias,
    uniform int batch_size,
    uniform int in_features,
    uniform int out_features,
    uniform int grid_size,
    uniform float grid_min,
    uniform float grid_max
)
{
    uint b_idx = threadId.x;
    uint j_base = threadId.y * 4;

    if (b_idx >= (uint)batch_size)
        return;

    float acc0 = (j_base + 0 < (uint)out_features) ? bias[j_base + 0] : 0.0f;
    float acc1 = (j_base + 1 < (uint)out_features) ? bias[j_base + 1] : 0.0f;
    float acc2 = (j_base + 2 < (uint)out_features) ? bias[j_base + 2] : 0.0f;
    float acc3 = (j_base + 3 < (uint)out_features) ? bias[j_base + 3] : 0.0f;

    float span = max(grid_max - grid_min, 1e-6f);
    int weight_stride = grid_size + 3;

    for (int i = 0; i < in_features; i++)
    {
        float x_raw = input[b_idx * in_features + i];
        // Normalize to [0, grid_size]
        float t = (clamp(x_raw, grid_min, grid_max) - grid_min) / span * (float)grid_size;
        int interval = clamp((int)floor(t), 0, grid_size - 1);
        float u = t - (float)interval;

        float bases[4];
        compute_cubic_bspline_4(u, bases);

        if (j_base + 0 < (uint)out_features) {
            int wb = (i * out_features + (int)j_base + 0) * weight_stride + interval;
            [unroll] for (int k = 0; k < 4; k++) acc0 += bases[k] * weights[wb + k];
        }
        if (j_base + 1 < (uint)out_features) {
            int wb = (i * out_features + (int)j_base + 1) * weight_stride + interval;
            [unroll] for (int k = 0; k < 4; k++) acc1 += bases[k] * weights[wb + k];
        }
        if (j_base + 2 < (uint)out_features) {
            int wb = (i * out_features + (int)j_base + 2) * weight_stride + interval;
            [unroll] for (int k = 0; k < 4; k++) acc2 += bases[k] * weights[wb + k];
        }
        if (j_base + 3 < (uint)out_features) {
            int wb = (i * out_features + (int)j_base + 3) * weight_stride + interval;
            [unroll] for (int k = 0; k < 4; k++) acc3 += bases[k] * weights[wb + k];
        }
    }

    if (j_base + 0 < (uint)out_features) output[b_idx * out_features + j_base + 0] = acc0;
    if (j_base + 1 < (uint)out_features) output[b_idx * out_features + j_base + 1] = acc1;
    if (j_base + 2 < (uint)out_features) output[b_idx * out_features + j_base + 2] = acc2;
    if (j_base + 3 < (uint)out_features) output[b_idx * out_features + j_base + 3] = acc3;
}
