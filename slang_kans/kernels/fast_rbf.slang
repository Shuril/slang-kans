/**
 * FastKAN Gaussian radial basis function kernel.
 */

module fast_rbf;

[Differentiable]
public float gaussian_rbf(float x, float center, float inv_scale)
{
    float d = (x - center) * inv_scale;
    return exp(-d * d);
}

[shader("compute")]
[numthreads(16, 4, 1)]
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
    uint j_base = threadId.y * 4;

    if (b_idx >= (uint)batch_size)
        return;

    float acc0 = (j_base + 0 < (uint)out_features) ? bias[j_base + 0] : 0.0f;
    float acc1 = (j_base + 1 < (uint)out_features) ? bias[j_base + 1] : 0.0f;
    float acc2 = (j_base + 2 < (uint)out_features) ? bias[j_base + 2] : 0.0f;
    float acc3 = (j_base + 3 < (uint)out_features) ? bias[j_base + 3] : 0.0f;

    float step = (grid_max - grid_min) / max((float)(num_centers - 1), 1.0f);
    float inv_sigma = 1.0f / max(step * 1.5f, 1e-6f);

    for (int i = 0; i < in_features; i++)
    {
        float x = input[b_idx * in_features + i];

        int w0 = (i * out_features + (int)j_base + 0) * num_centers;
        int w1 = (i * out_features + (int)j_base + 1) * num_centers;
        int w2 = (i * out_features + (int)j_base + 2) * num_centers;
        int w3 = (i * out_features + (int)j_base + 3) * num_centers;

        bool has0 = (j_base + 0 < (uint)out_features);
        bool has1 = (j_base + 1 < (uint)out_features);
        bool has2 = (j_base + 2 < (uint)out_features);
        bool has3 = (j_base + 3 < (uint)out_features);

        for (int c = 0; c < num_centers; c++)
        {
            float center = grid_min + (float)c * step;
            float d = (x - center) * inv_sigma;
            float rbf_val = exp(-d * d);

            if (has0) acc0 += rbf_val * weights[w0 + c];
            if (has1) acc1 += rbf_val * weights[w1 + c];
            if (has2) acc2 += rbf_val * weights[w2 + c];
            if (has3) acc3 += rbf_val * weights[w3 + c];
        }
    }

    if (j_base + 0 < (uint)out_features) output[b_idx * out_features + j_base + 0] = acc0;
    if (j_base + 1 < (uint)out_features) output[b_idx * out_features + j_base + 1] = acc1;
    if (j_base + 2 < (uint)out_features) output[b_idx * out_features + j_base + 2] = acc2;
    if (j_base + 3 < (uint)out_features) output[b_idx * out_features + j_base + 3] = acc3;
}

[shader("compute")]
[numthreads(16, 16, 1)]
public void eval_rbf_basis(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> phi,
    uniform StructuredBuffer<float> input,
    uniform int batch_size,
    uniform int in_features,
    uniform int num_centers,
    uniform float grid_min,
    uniform float grid_max
) {
    uint b = threadId.x;
    uint din = threadId.y;
    if (b >= (uint)batch_size || din >= (uint)in_features) return;
    float x = input[b * in_features + din];
    float step = (grid_max - grid_min) / max((float)(num_centers - 1), 1.0f);
    float inv_sigma = 1.0f / max(step * 1.5f, 1e-6f);
    uint out_offset = b * (in_features * num_centers) + din * num_centers;
    for (int c = 0; c < num_centers; c++) {
        float center = grid_min + (float)c * step;
        float d = (x - center) * inv_sigma;
        phi[out_offset + c] = exp(-d * d);
    }
}

