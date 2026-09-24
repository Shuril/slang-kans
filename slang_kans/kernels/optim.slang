/**
 * GPU compute shaders for optimizers in Slang.
 * Computes SGD, Adam, AdamW, Lion, RMSprop, Muon, and MSE loss gradients.
 */

module optim;

[shader("compute")]
[numthreads(64, 1, 1)]
public void adamw_step(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> param,
    uniform StructuredBuffer<float> grad,
    uniform RWStructuredBuffer<float> m,
    uniform RWStructuredBuffer<float> v,
    uniform float lr,
    uniform float beta1,
    uniform float beta2,
    uniform float eps,
    uniform float weight_decay,
    uniform float lr_t,
    uniform int n_elems
) {
    uint i = threadId.x;
    if (i >= (uint)n_elems) return;

    float p = param[i];
    if (weight_decay != 0.0f) {
        p -= lr * weight_decay * p;
    }

    float g = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * (g * g);

    m[i] = mi;
    v[i] = vi;

    float denom = sqrt(vi) + eps;
    param[i] = p - lr_t * (mi / denom);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void adam_step(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> param,
    uniform StructuredBuffer<float> grad,
    uniform RWStructuredBuffer<float> m,
    uniform RWStructuredBuffer<float> v,
    uniform float lr,
    uniform float beta1,
    uniform float beta2,
    uniform float eps,
    uniform float weight_decay,
    uniform float lr_t,
    uniform int n_elems
) {
    uint i = threadId.x;
    if (i >= (uint)n_elems) return;

    float p = param[i];
    float g = grad[i];
    if (weight_decay != 0.0f) {
        g += weight_decay * p;
    }

    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * (g * g);

    m[i] = mi;
    v[i] = vi;

    float denom = sqrt(vi) + eps;
    param[i] = p - lr_t * (mi / denom);
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void sgd_step(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> param,
    uniform StructuredBuffer<float> grad,
    uniform RWStructuredBuffer<float> v,
    uniform float lr,
    uniform float momentum,
    uniform float weight_decay,
    uniform int nesterov,
    uniform int n_elems
) {
    uint i = threadId.x;
    if (i >= (uint)n_elems) return;

    float p = param[i];
    float g = grad[i];
    if (weight_decay != 0.0f) {
        g += weight_decay * p;
    }

    float update = g;
    if (momentum != 0.0f) {
        float vi = momentum * v[i] + g;
        v[i] = vi;
        if (nesterov != 0) {
            update = g + momentum * vi;
        } else {
            update = vi;
        }
    }

    param[i] = p - lr * update;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void lion_step(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> param,
    uniform StructuredBuffer<float> grad,
    uniform RWStructuredBuffer<float> m,
    uniform float lr,
    uniform float beta1,
    uniform float beta2,
    uniform float weight_decay,
    uniform int n_elems
) {
    uint i = threadId.x;
    if (i >= (uint)n_elems) return;

    float p = param[i];
    if (weight_decay != 0.0f) {
        p -= lr * weight_decay * p;
    }

    float g = grad[i];
    float mi = m[i];

    float dir = beta1 * mi + (1.0f - beta1) * g;
    float s = (dir > 0.0f) ? 1.0f : ((dir < 0.0f) ? -1.0f : 0.0f);

    param[i] = p - lr * s;
    m[i] = beta2 * mi + (1.0f - beta2) * g;
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void rmsprop_step(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> param,
    uniform StructuredBuffer<float> grad,
    uniform RWStructuredBuffer<float> v,
    uniform RWStructuredBuffer<float> buf,
    uniform float lr,
    uniform float alpha,
    uniform float eps,
    uniform float weight_decay,
    uniform float momentum,
    uniform int has_momentum,
    uniform int n_elems
) {
    uint i = threadId.x;
    if (i >= (uint)n_elems) return;

    float p = param[i];
    float g = grad[i];
    if (weight_decay != 0.0f) {
        g += weight_decay * p;
    }

    float vi = alpha * v[i] + (1.0f - alpha) * (g * g);
    v[i] = vi;

    float avg = g / (sqrt(vi) + eps);

    if (has_momentum != 0) {
        float bi = momentum * buf[i] + avg;
        buf[i] = bi;
        param[i] = p - lr * bi;
    } else {
        param[i] = p - lr * avg;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void muon_momentum_update(
    uint3 threadId : SV_DispatchThreadID,
    uniform StructuredBuffer<float> param,
    uniform StructuredBuffer<float> grad,
    uniform RWStructuredBuffer<float> v,
    uniform RWStructuredBuffer<float> update,
    uniform float momentum,
    uniform float weight_decay,
    uniform int nesterov,
    uniform int n_elems
) {
    uint i = threadId.x;
    if (i >= (uint)n_elems) return;

    float g = grad[i];
    if (weight_decay != 0.0f) {
        g += weight_decay * param[i];
    }

    float vi = momentum * v[i] + (1.0f - momentum) * g;
    v[i] = vi;

    if (nesterov != 0) {
        update[i] = (1.0f - momentum) * g + momentum * vi;
    } else {
        update[i] = vi;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void muon_param_update(
    uint3 threadId : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> param,
    uniform StructuredBuffer<float> ortho,
    uniform float step_lr,
    uniform int n_elems
) {
    uint i = threadId.x;
    if (i >= (uint)n_elems) return;

    param[i] -= step_lr * ortho[i];
}

[shader("compute")]
[numthreads(64, 1, 1)]
public void calc_mse_loss_backward(
    uint3 threadId : SV_DispatchThreadID,
    uniform StructuredBuffer<float> pred,
    uniform StructuredBuffer<float> target,
    uniform RWStructuredBuffer<float> dY,
    uniform float scale,
    uniform int n_elems
) {
    uint i = threadId.x;
    if (i >= (uint)n_elems) return;

    dY[i] = scale * (pred[i] - target[i]);
}
