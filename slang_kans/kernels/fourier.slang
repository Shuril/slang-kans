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
[numthreads(16, 4, 1)]
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
    uint j_base = threadId.y * 4;

    if (b >= (uint)batch_size)
        return;

    float acc0 = (j_base + 0 < (uint)out_features) ? bias[j_base + 0] : 0.0f;
    float acc1 = (j_base + 1 < (uint)out_features) ? bias[j_base + 1] : 0.0f;
    float acc2 = (j_base + 2 < (uint)out_features) ? bias[j_base + 2] : 0.0f;
    float acc3 = (j_base + 3 < (uint)out_features) ? bias[j_base + 3] : 0.0f;

    int num_bases = 2 * num_frequencies + 1;

    for (int i = 0; i < in_features; i++)
    {
        float x_val = input[b * in_features + i];

        float cos_terms[16];
        float sin_terms[16];
        for (int k = 1; k <= num_frequencies && k < 16; k++)
        {
            float arg = (float)k * PI * x_val;
            cos_terms[k] = cos(arg);
            sin_terms[k] = sin(arg);
        }

        for (int sub = 0; sub < 4; sub++)
        {
            uint j = j_base + (uint)sub;
            if (j < (uint)out_features)
            {
                int w_base = (i * out_features + (int)j) * num_bases;
                float val = weights[w_base + 0];
                for (int k = 1; k <= num_frequencies && k < 16; k++)
                {
                    val += cos_terms[k] * weights[w_base + k];
                    val += sin_terms[k] * weights[w_base + num_frequencies + k];
                }
                if (sub == 0) acc0 += val;
                else if (sub == 1) acc1 += val;
                else if (sub == 2) acc2 += val;
                else acc3 += val;
            }
        }
    }

    if (j_base + 0 < (uint)out_features) output[b * out_features + j_base + 0] = acc0;
    if (j_base + 1 < (uint)out_features) output[b * out_features + j_base + 1] = acc1;
    if (j_base + 2 < (uint)out_features) output[b * out_features + j_base + 2] = acc2;
    if (j_base + 3 < (uint)out_features) output[b * out_features + j_base + 3] = acc3;
}

[shader("compute")]
[numthreads(16, 16, 1)]
public void eval_fourier_basis(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> phi,
    uniform StructuredBuffer<float> input,
    uniform int batch_size,
    uniform int in_features,
    uniform int num_frequencies
) {
    uint b = threadId.x;
    uint din = threadId.y;
    if (b >= (uint)batch_size || din >= (uint)in_features) return;
    float x_val = input[b * in_features + din];
    int num_bases = 2 * num_frequencies + 1;
    uint out_offset = b * (in_features * num_bases) + din * num_bases;

    phi[out_offset + 0] = 1.0f;
    for (int k = 1; k <= num_frequencies; k++) {
        float arg = (float)k * PI * x_val;
        phi[out_offset + k]                   = cos(arg);
        phi[out_offset + num_frequencies + k] = sin(arg);
    }
}
