/**
 * Orthogonal Chebyshev polynomial KAN kernel with autodiff.
 * Computes y_j = bias_j + sum_{i=0}^{d_in-1} Clenshaw(x_i, weights_{i, j, :})
 * with [Differentiable] source-to-source automatic differentiation.
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
            float s = clamp(x_val, -1.0f, 1.0f);

            int w0 = (i * out_features + (int)j_base + 0) * deg_stride;
            int w1 = (i * out_features + (int)j_base + 1) * deg_stride;
            int w2 = (i * out_features + (int)j_base + 2) * deg_stride;
            int w3 = (i * out_features + (int)j_base + 3) * deg_stride;

            bool has0 = (j_base + 0 < (uint)out_features);
            bool has1 = (j_base + 1 < (uint)out_features);
            bool has2 = (j_base + 2 < (uint)out_features);
            bool has3 = (j_base + 3 < (uint)out_features);

            // T_0(x) = 1.0
            float t_prev2 = 1.0f;
            if (has0) acc0 += weights[w0 + 0];
            if (has1) acc1 += weights[w1 + 0];
            if (has2) acc2 += weights[w2 + 0];
            if (has3) acc3 += weights[w3 + 0];

            if (degree >= 1)
            {
                // T_1(x) = s
                float t_prev1 = s;
                if (has0) acc0 += s * weights[w0 + 1];
                if (has1) acc1 += s * weights[w1 + 1];
                if (has2) acc2 += s * weights[w2 + 1];
                if (has3) acc3 += s * weights[w3 + 1];

                float two_s = 2.0f * s;
                for (int k = 2; k <= degree; k++)
                {
                    float t_curr = two_s * t_prev1 - t_prev2;
                    if (has0) acc0 += t_curr * weights[w0 + k];
                    if (has1) acc1 += t_curr * weights[w1 + k];
                    if (has2) acc2 += t_curr * weights[w2 + k];
                    if (has3) acc3 += t_curr * weights[w3 + k];
                    t_prev2 = t_prev1;
                    t_prev1 = t_curr;
                }
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
public void eval_cheby_basis(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> phi,
    uniform StructuredBuffer<float> input,
    uniform int batch_size,
    uniform int in_features,
    uniform int num_bases
) {
    uint b = threadId.x;
    uint din = threadId.y;
    if (b >= (uint)batch_size || din >= (uint)in_features) return;
    float x = clamp(input[b * in_features + din], -1.0f, 1.0f);
    uint out_offset = b * (in_features * num_bases) + din * num_bases;

    phi[out_offset + 0] = 1.0f;
    if (num_bases > 1) phi[out_offset + 1] = x;
    float t_prev2 = 1.0f;
    float t_prev1 = x;
    for (int k = 2; k < num_bases; k++) {
        float t_curr = 2.0f * x * t_prev1 - t_prev2;
        phi[out_offset + k] = t_curr;
        t_prev2 = t_prev1;
        t_prev1 = t_curr;
    }
}
