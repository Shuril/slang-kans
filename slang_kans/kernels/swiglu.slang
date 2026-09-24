/**
 * SwiGLU gating and add kernels for GatedKAN in Slang.
 * Computes SiLU(gate) * up forward and analytical derivatives backward.
 */

module swiglu;

static float sigmoid_fast(float x) {
    float c = clamp(x, -20.0f, 20.0f);
    return 1.0f / (1.0f + exp(-c));
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void swiglu_forward_f32(
    uint3 threadId : SV_DispatchThreadID,
    uniform StructuredBuffer<float> gate,
    uniform StructuredBuffer<float> up,
    uniform RWStructuredBuffer<float> output,
    uniform int n_elems
) {
    uint idx = threadId.x;
    if (idx >= (uint)n_elems) return;

    float g = gate[idx];
    float u = up[idx];
    float silu_g = g * sigmoid_fast(g);
    output[idx] = silu_g * u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void swiglu_backward_f32(
    uint3 threadId : SV_DispatchThreadID,
    uniform StructuredBuffer<float> d_hidden,
    uniform StructuredBuffer<float> gate,
    uniform StructuredBuffer<float> up,
    uniform RWStructuredBuffer<float> d_gate,
    uniform RWStructuredBuffer<float> d_up,
    uniform int n_elems
) {
    uint idx = threadId.x;
    if (idx >= (uint)n_elems) return;

    float dh = d_hidden[idx];
    float g = gate[idx];
    float u = up[idx];

    float sig = sigmoid_fast(g);
    float silu_g = g * sig;
    float d_silu = sig * (1.0f + g * (1.0f - sig));

    d_up[idx] = dh * silu_g;
    d_gate[idx] = dh * u * d_silu;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void elementwise_add_f32(
    uint3 threadId : SV_DispatchThreadID,
    uniform StructuredBuffer<float> a,
    uniform StructuredBuffer<float> b,
    uniform RWStructuredBuffer<float> output,
    uniform int n_elems
) {
    uint idx = threadId.x;
    if (idx >= (uint)n_elems) return;

    output[idx] = a[idx] + b[idx];
}
