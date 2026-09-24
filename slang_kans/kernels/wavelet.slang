/**
 * Continuous wavelet KAN kernel with autodiff.
 * Supports Mexican Hat, Morlet, and Derivative of Gaussian (DOG) wavelets.
 */

module wavelet;

[Differentiable]
public float eval_wavelet(float x, float translation, float scale, int wavelet_type)
{
    float z = (x - translation) / (abs(scale) + 1e-4f);
    float z2 = z * z;
    float gauss = exp(-0.5f * z2);

    if (wavelet_type == 1) // Morlet
    {
        return cos(5.0f * z) * gauss;
    }
    else if (wavelet_type == 2) // DOG
    {
        return -z * gauss;
    }
    else // Mexican Hat / Ricker (type 0)
    {
        return (1.0f - z2) * gauss;
    }
}

[shader("compute")]
[numthreads(16, 4, 1)]
public void wavelet_kan_forward(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> output,          // [B, D_out]
    uniform StructuredBuffer<float> input,            // [B, D_in]
    uniform StructuredBuffer<float> weights,          // [D_in, D_out, num_wavelets]
    uniform StructuredBuffer<float> translations,     // [D_in, num_wavelets]
    uniform StructuredBuffer<float> scales,           // [D_in, num_wavelets]
    uniform StructuredBuffer<float> bias,             // [D_out]
    uniform int batch_size,
    uniform int in_features,
    uniform int out_features,
    uniform int num_wavelets,
    uniform int wavelet_type
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

    for (int i = 0; i < in_features; i++)
    {
        float x_val = input[b * in_features + i];
        int param_base = i * num_wavelets;

        int w0 = (i * out_features + (int)j_base + 0) * num_wavelets;
        int w1 = (i * out_features + (int)j_base + 1) * num_wavelets;
        int w2 = (i * out_features + (int)j_base + 2) * num_wavelets;
        int w3 = (i * out_features + (int)j_base + 3) * num_wavelets;

        bool has0 = (j_base + 0 < (uint)out_features);
        bool has1 = (j_base + 1 < (uint)out_features);
        bool has2 = (j_base + 2 < (uint)out_features);
        bool has3 = (j_base + 3 < (uint)out_features);

        for (int w = 0; w < num_wavelets; w++)
        {
            float trans = translations[param_base + w];
            float sc = scales[param_base + w];
            float phi_val = eval_wavelet(x_val, trans, sc, wavelet_type);

            if (has0) acc0 += phi_val * weights[w0 + w];
            if (has1) acc1 += phi_val * weights[w1 + w];
            if (has2) acc2 += phi_val * weights[w2 + w];
            if (has3) acc3 += phi_val * weights[w3 + w];
        }
    }

    if (j_base + 0 < (uint)out_features) output[b * out_features + j_base + 0] = acc0;
    if (j_base + 1 < (uint)out_features) output[b * out_features + j_base + 1] = acc1;
    if (j_base + 2 < (uint)out_features) output[b * out_features + j_base + 2] = acc2;
    if (j_base + 3 < (uint)out_features) output[b * out_features + j_base + 3] = acc3;
}

[shader("compute")]
[numthreads(16, 16, 1)]
public void eval_wavelet_basis(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> phi,
    uniform StructuredBuffer<float> input,
    uniform StructuredBuffer<float> translations,
    uniform StructuredBuffer<float> scales,
    uniform int batch_size,
    uniform int in_features,
    uniform int num_wavelets,
    uniform int wavelet_type
) {
    uint b = threadId.x;
    uint din = threadId.y;
    if (b >= (uint)batch_size || din >= (uint)in_features) return;
    float x = input[b * in_features + din];
    uint param_offset = din * num_wavelets;
    uint out_offset = b * (in_features * num_wavelets) + din * num_wavelets;

    for (int w = 0; w < num_wavelets; w++) {
        float tr = translations[param_offset + w];
        float sc = scales[param_offset + w];
        phi[out_offset + w] = eval_wavelet(x, tr, sc, wavelet_type);
    }
}
