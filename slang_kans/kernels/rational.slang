/**
 * ==============================================================================
 *  RATIONAL.SLANG - Padé-Chebyshev Rational KAN Kernel with Autodiff
 * ==============================================================================
 *  Computes phi(x) = P_p(x) / (1 + |Q_q(x)|) using Chebyshev series for P and Q.
 * ==============================================================================
 */

module rational;

[Differentiable]
public float clenshaw(float x, float coeffs[8], int deg)
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

[shader("compute")]
[numthreads(16, 16, 1)]
public void rational_kan_forward(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> output,     // [B, D_out]
    uniform StructuredBuffer<float> input,       // [B, D_in]
    uniform StructuredBuffer<float> p_weights,   // [D_in, D_out, p_degree + 1]
    uniform StructuredBuffer<float> q_weights,   // [D_in, D_out, q_degree + 1]
    uniform StructuredBuffer<float> bias,        // [D_out]
    uniform int batch_size,
    uniform int in_features,
    uniform int out_features,
    uniform int p_degree,
    uniform int q_degree
)
{
    uint b = threadId.x;
    uint j = threadId.y;

    if (b >= (uint)batch_size || j >= (uint)out_features)
        return;

    float acc = bias[j];
    int p_stride = p_degree + 1;
    int q_stride = q_degree + 1;

    for (int i = 0; i < in_features; i++)
    {
        float x_val = input[b * in_features + i];

        // Load P coeffs
        float cp[8];
        [unroll]
        for (int k = 0; k <= 7; k++)
        {
            if (k <= p_degree)
            {
                int idx = (i * out_features + (int)j) * p_stride + k;
                cp[k] = p_weights[idx];
            }
            else
            {
                cp[k] = 0.0f;
            }
        }
        float p_val = clenshaw(x_val, cp, p_degree);

        // Load Q coeffs
        float cq[8];
        [unroll]
        for (int m = 0; m <= 7; m++)
        {
            if (m <= q_degree)
            {
                int idx = (i * out_features + (int)j) * q_stride + m;
                cq[m] = q_weights[idx];
            }
            else
            {
                cq[m] = 0.0f;
            }
        }
        float q_val = clenshaw(x_val, cq, q_degree);

        acc += p_val / (1.0f + abs(q_val));
    }

    output[b * out_features + j] = acc;
}
