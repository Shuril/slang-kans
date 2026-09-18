/**
 * ==============================================================================
 *  CHEBY.SLANG - Orthogonal Chebyshev Polynomial KAN Kernel with Autodiff
 * ==============================================================================
 *  Computes y_j = bias_j + sum_{i=0}^{d_in-1} Clenshaw(x_i, weights_{i, j, :})
 *  with [Differentiable] source-to-source automatic differentiation.
 * ==============================================================================
 */

module cheby;

/**
 * Evaluates Chebyshev polynomial series on [-1, 1] using Clenshaw recurrence.
 * Entirely register-based FMA arithmetic.
 */
[Differentiable]
public float clenshaw_eval(float x, float coeffs[8], int deg)
{
    float s = clamp(x, -1.0f, 1.0f);
    float two_s = 2.0f * s;
    float b2 = 0.0f;
    float b1 = 0.0f;
    float b0 = 0.0f;

    [unroll]
    for (int k = 7; k >= 1; k--)
    {
        if (k <= deg)
        {
            b0 = two_s * b1 - b2 + coeffs[k];
            b2 = b1;
            b1 = b0;
        }
    }
    return s * b1 - b2 + coeffs[0];
}

/**
 * High-performance 4-output tiled ChebyKAN Layer forward invocation.
 * Dispatched across Batch B and (D_out + 3) / 4.
 */
[shader("compute")]
[numthreads(16, 4, 1)]
public void cheby_kan_forward(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> output,     // [B, D_out]
    uniform StructuredBuffer<float> input,       // [B, D_in]
    uniform StructuredBuffer<float> weights,     // [D_in, D_out, Deg+1]
    uniform StructuredBuffer<float> bias,        // [D_out]
    uniform int batch_size,
    uniform int in_features,
    uniform int out_features,
    uniform int degree
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

    int deg_stride = degree + 1;

    if (degree == 4)
    {
        for (int i = 0; i < in_features; i++)
        {
            float x_val = input[b * in_features + i];
            float s = clamp(x_val, -1.0f, 1.0f);
            float s2 = s * s;
            float t0 = 1.0f;
            float t1 = s;
            float t2 = 2.0f * s2 - 1.0f;
            float t3 = s * (4.0f * s2 - 3.0f);
            float t4 = 8.0f * s2 * (s2 - 1.0f) + 1.0f;

            if (j_base + 0 < (uint)out_features) {
                int w = (i * out_features + (int)j_base + 0) * 5;
                acc0 += t0 * weights[w] + t1 * weights[w+1] + t2 * weights[w+2] + t3 * weights[w+3] + t4 * weights[w+4];
            }
            if (j_base + 1 < (uint)out_features) {
                int w = (i * out_features + (int)j_base + 1) * 5;
                acc1 += t0 * weights[w] + t1 * weights[w+1] + t2 * weights[w+2] + t3 * weights[w+3] + t4 * weights[w+4];
            }
            if (j_base + 2 < (uint)out_features) {
                int w = (i * out_features + (int)j_base + 2) * 5;
                acc2 += t0 * weights[w] + t1 * weights[w+1] + t2 * weights[w+2] + t3 * weights[w+3] + t4 * weights[w+4];
            }
            if (j_base + 3 < (uint)out_features) {
                int w = (i * out_features + (int)j_base + 3) * 5;
                acc3 += t0 * weights[w] + t1 * weights[w+1] + t2 * weights[w+2] + t3 * weights[w+3] + t4 * weights[w+4];
            }
        }
    }
    else
    {
        for (int i = 0; i < in_features; i++)
        {
            float x_val = input[b * in_features + i];

            for (int sub = 0; sub < 4; sub++)
            {
                uint j = j_base + (uint)sub;
                if (j < (uint)out_features)
                {
                    float c[8];
                    for (int k = 0; k < 8; k++) c[k] = 0.0f;
                    for (int k = 0; k <= degree; k++)
                    {
                        int w_idx = (i * out_features + (int)j) * deg_stride + k;
                        c[k] = weights[w_idx];
                    }
                    float val = clenshaw_eval(x_val, c, degree);
                    if (sub == 0) acc0 += val;
                    else if (sub == 1) acc1 += val;
                    else if (sub == 2) acc2 += val;
                    else acc3 += val;
                }
            }
        }
    }

    if (j_base + 0 < (uint)out_features) output[b * out_features + j_base + 0] = acc0;
    if (j_base + 1 < (uint)out_features) output[b * out_features + j_base + 1] = acc1;
    if (j_base + 2 < (uint)out_features) output[b * out_features + j_base + 2] = acc2;
    if (j_base + 3 < (uint)out_features) output[b * out_features + j_base + 3] = acc3;
}
