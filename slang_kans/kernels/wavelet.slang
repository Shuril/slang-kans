/**
 * ==============================================================================
 *  WAVELET.SLANG - Continuous Wavelet KAN Kernel with Autodiff
 * ==============================================================================
 *  Supports Mexican Hat, Morlet, and Derivative of Gaussian (DOG) wavelets.
 * ==============================================================================
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
[numthreads(16, 16, 1)]
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
    uint j = threadId.y;

    if (b >= (uint)batch_size || j >= (uint)out_features)
        return;

    float acc = bias[j];

    for (int i = 0; i < in_features; i++)
    {
        float x_val = input[b * in_features + i];
        int w_base = (i * out_features + (int)j) * num_wavelets;
        int param_base = i * num_wavelets;

        for (int w = 0; w < num_wavelets; w++)
        {
            float trans = translations[param_base + w];
            float sc = scales[param_base + w];
            float phi = eval_wavelet(x_val, trans, sc, wavelet_type);
            acc += phi * weights[w_base + w];
        }
    }

    output[b * out_features + j] = acc;
}
