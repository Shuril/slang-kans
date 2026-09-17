/**
 * ==============================================================================
 *  FOURIER.SLANG - Fourier Series Harmonic KAN Kernel with Autodiff
 * ==============================================================================
 *  Bases: 1.0, cos(k * pi * x), sin(k * pi * x) for k = 1..K. Total bases = 2*K + 1.
 * ==============================================================================
 */

module fourier;

static const float PI = 3.14159265358979323846f;

[shader("compute")]
[numthreads(16, 16, 1)]
public void fourier_kan_forward(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> output,     // [B, D_out]
    uniform StructuredBuffer<float> input,       // [B, D_in]
    uniform StructuredBuffer<float> weights,     // [D_in, D_out, 2*K + 1]
    uniform StructuredBuffer<float> bias,        // [D_out]
    uniform int batch_size,
    uniform int in_features,
    uniform int out_features,
    uniform int num_frequencies
)
{
    uint b = threadId.x;
    uint j = threadId.y;

    if (b >= (uint)batch_size || j >= (uint)out_features)
        return;

    float acc = bias[j];
    int num_bases = 2 * num_frequencies + 1;

    for (int i = 0; i < in_features; i++)
    {
        float x_val = input[b * in_features + i];
        int w_base = (i * out_features + (int)j) * num_bases;

        // Constant term (basis 0)
        acc += 1.0f * weights[w_base + 0];

        // Harmonic terms
        for (int k = 1; k <= num_frequencies; k++)
        {
            float arg = (float)k * PI * x_val;
            float cos_term = cos(arg);
            float sin_term = sin(arg);

            acc += cos_term * weights[w_base + k];
            acc += sin_term * weights[w_base + num_frequencies + k];
        }
    }

    output[b * out_features + j] = acc;
}
