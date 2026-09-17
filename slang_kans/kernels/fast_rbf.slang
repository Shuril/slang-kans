/**
 * ==============================================================================
 *  FAST_RBF.SLANG - FastKAN Gaussian Radial Basis Function Kernel
 * ==============================================================================
 */

module fast_rbf;

[Differentiable]
public float gaussian_rbf(float x, float center, float inv_scale)
{
    float d = (x - center) * inv_scale;
    return exp(-d * d);
}

[shader("compute")]
[numthreads(16, 16, 1)]
public void fast_kan_forward(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> output,
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> weights,
    uniform StructuredBuffer<float> bias,
    uniform int batch_size,
    uniform int in_features,
    uniform int out_features,
    uniform int num_centers,
    uniform float grid_min,
    uniform float grid_max
)
{
    uint b_idx = threadId.x;
    uint j = threadId.y;

    if (b_idx >= (uint)batch_size || j >= (uint)out_features)
        return;

    float acc = bias[j];
    float step = (grid_max - grid_min) / max((float)(num_centers - 1), 1.0f);
    float inv_sigma = 1.0f / max(step * 1.5f, 1e-6f);

    for (int i = 0; i < in_features; i++)
    {
        float x = input[b_idx * in_features + i];
        int w_base = (i * out_features + (int)j) * num_centers;

        for (int c = 0; c < num_centers; c++)
        {
            float center = grid_min + (float)c * step;
            float rbf_val = gaussian_rbf(x, center, inv_sigma);
            acc += rbf_val * weights[w_base + c];
        }
    }

    output[b_idx * out_features + j] = acc;
}
