/**
 * Orthogonal Jacobi polynomial KAN kernel with autodiff.
 * General orthogonal polynomials on [-1, 1] parameterized by (alpha, beta).
 * Special cases include Legendre (0, 0) and Chebyshev (-0.5, -0.5).
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

    float a_b = alpha + beta;

    for (int i = 0; i < in_features; i++)
    {
        float x_val = input[b * in_features + i];
        float xc = clamp(x_val, -1.0f, 1.0f);

        int w0 = (i * out_features + (int)j_base + 0) * degree;
        int w1 = (i * out_features + (int)j_base + 1) * degree;
        int w2 = (i * out_features + (int)j_base + 2) * degree;
        int w3 = (i * out_features + (int)j_base + 3) * degree;

        bool has0 = (j_base + 0 < (uint)out_features);
        bool has1 = (j_base + 1 < (uint)out_features);
        bool has2 = (j_base + 2 < (uint)out_features);
        bool has3 = (j_base + 3 < (uint)out_features);

        // P_0(x) = 1.0
        float p_prev2 = 1.0f;
        if (has0) acc0 += weights[w0 + 0];
        if (has1) acc1 += weights[w1 + 0];
        if (has2) acc2 += weights[w2 + 0];
        if (has3) acc3 += weights[w3 + 0];

        if (degree > 1)
        {
            // P_1(x) = 0.5 * (alpha - beta + (a_b + 2.0) * xc)
            float p_prev1 = 0.5f * (alpha - beta + (a_b + 2.0f) * xc);
            if (has0) acc0 += p_prev1 * weights[w0 + 1];
            if (has1) acc1 += p_prev1 * weights[w1 + 1];
            if (has2) acc2 += p_prev1 * weights[w2 + 1];
            if (has3) acc3 += p_prev1 * weights[w3 + 1];

            for (int n = 2; n < degree; n++)
            {
                float fn = (float)n;
                float an = 2.0f * fn * (fn + a_b) * (2.0f * fn + a_b - 2.0f);
                if (abs(an) > 1e-7f)
                {
                    float bn_term1 = (2.0f * fn + a_b - 1.0f) * (2.0f * fn + a_b) * (2.0f * fn + a_b - 2.0f);
                    float bn_term2 = (2.0f * fn + a_b - 1.0f) * (alpha * alpha - beta * beta);
                    float bn = bn_term1 * xc + bn_term2;
                    float cn = 2.0f * (fn + alpha - 1.0f) * (fn + beta - 1.0f) * (2.0f * fn + a_b);

                    float p_curr = (bn * p_prev1 - cn * p_prev2) / an;
                    if (has0) acc0 += p_curr * weights[w0 + n];
                    if (has1) acc1 += p_curr * weights[w1 + n];
                    if (has2) acc2 += p_curr * weights[w2 + n];
                    if (has3) acc3 += p_curr * weights[w3 + n];

                    p_prev2 = p_prev1;
                    p_prev1 = p_curr;
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
public void eval_jacobi_basis(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> phi,
    uniform StructuredBuffer<float> input,
    uniform int batch_size,
    uniform int in_features,
    uniform int degree,
    uniform float alpha,
    uniform float beta
) {
    uint b = threadId.x;
    uint din = threadId.y;
    if (b >= (uint)batch_size || din >= (uint)in_features) return;
    float x_val = input[b * in_features + din];
    float bases[16];
    compute_jacobi_bases(x_val, degree, alpha, beta, bases);
    uint out_offset = b * (in_features * degree) + din * degree;
    for (int d = 0; d < degree; d++) {
        phi[out_offset + d] = bases[d];
    }
}
