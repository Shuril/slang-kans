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
 * Fused ChebyKAN Layer forward invocation.
 * Dispatched across Batch B and Output dimension D_out.
 */
[shader("compute")]
[numthreads(16, 16, 1)]
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
    uint j = threadId.y;

    if (b >= (uint)batch_size || j >= (uint)out_features)
        return;

    float acc = bias[j];
    int deg_stride = degree + 1;

    for (int i = 0; i < in_features; i++)
    {
        float x_val = input[b * in_features + i];

        // Load local polynomial coefficients
        float c[8];
        [unroll]
        for (int k = 0; k <= 7; k++)
        {
            if (k <= degree)
            {
                int w_idx = (i * out_features + (int)j) * deg_stride + k;
                c[k] = weights[w_idx];
            }
            else
            {
                c[k] = 0.0f;
            }
        }

        acc += clenshaw_eval(x_val, c, degree);
    }

    output[b * out_features + j] = acc;
}
