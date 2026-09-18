/**
 * ==============================================================================
 *  JACOBI.SLANG - Orthogonal Jacobi Polynomial KAN Kernel with Autodiff
 * ==============================================================================
 *  General orthogonal polynomials on [-1, 1] parameterized by (alpha, beta).
 *  Special cases include Legendre (0, 0) and Chebyshev (-0.5, -0.5).
 * ==============================================================================
 */

module jacobi;

[Differentiable]
public void compute_jacobi_bases(float x, int degree, float alpha, float beta, out float bases[16])
{
    float xc = clamp(x, -1.0f, 1.0f);
    bases[0] = 1.0f;

    if (degree > 1)
    {
        bases[1] = 0.5f * (alpha - beta + (alpha + beta + 2.0f) * xc);
        float a_b = alpha + beta;

        [unroll]
        for (int n = 2; n < 16; n++)
        {
            if (n < degree)
            {
                float fn = (float)n;
                float an = 2.0f * fn * (fn + a_b) * (2.0f * fn + a_b - 2.0f);
                float bn_term1 = (2.0f * fn + a_b - 1.0f) * (2.0f * fn + a_b) * (2.0f * fn + a_b - 2.0f);
                float bn_term2 = (2.0f * fn + a_b - 1.0f) * (alpha * alpha - beta * beta);
                float bn = bn_term1 * xc + bn_term2;
                float cn = 2.0f * (fn + alpha - 1.0f) * (fn + beta - 1.0f) * (2.0f * fn + a_b);

                bases[n] = (bn * bases[n - 1] - cn * bases[n - 2]) / an;
            }
            else
            {
                bases[n] = 0.0f;
            }
        }
    }
}

[shader("compute")]
[numthreads(16, 4, 1)]
public void jacobi_kan_forward(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> output,     // [B, D_out]
    uniform StructuredBuffer<float> input,       // [B, D_in]
    uniform StructuredBuffer<float> weights,     // [D_in, D_out, degree]
    uniform StructuredBuffer<float> bias,        // [D_out]
    uniform int batch_size,
    uniform int in_features,
    uniform int out_features,
    uniform int degree,
    uniform float alpha,
    uniform float beta
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

        float bases[16];
        compute_jacobi_bases(x_val, degree, alpha, beta, bases);

        if (j_base + 0 < (uint)out_features) {
            int w0 = (i * out_features + (int)j_base + 0) * degree;
            for (int d = 0; d < degree; d++) acc0 += bases[d] * weights[w0 + d];
        }
        if (j_base + 1 < (uint)out_features) {
            int w1 = (i * out_features + (int)j_base + 1) * degree;
            for (int d = 0; d < degree; d++) acc1 += bases[d] * weights[w1 + d];
        }
        if (j_base + 2 < (uint)out_features) {
            int w2 = (i * out_features + (int)j_base + 2) * degree;
            for (int d = 0; d < degree; d++) acc2 += bases[d] * weights[w2 + d];
        }
        if (j_base + 3 < (uint)out_features) {
            int w3 = (i * out_features + (int)j_base + 3) * degree;
            for (int d = 0; d < degree; d++) acc3 += bases[d] * weights[w3 + d];
        }
    }

    if (j_base + 0 < (uint)out_features) output[b * out_features + j_base + 0] = acc0;
    if (j_base + 1 < (uint)out_features) output[b * out_features + j_base + 1] = acc1;
    if (j_base + 2 < (uint)out_features) output[b * out_features + j_base + 2] = acc2;
    if (j_base + 3 < (uint)out_features) output[b * out_features + j_base + 3] = acc3;
}
