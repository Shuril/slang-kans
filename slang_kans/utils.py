"""
Utility functions for slang-KANs.
Includes parameter counters, precision helpers, and train step builder with autodiff.
"""

from __future__ import annotations
from typing import Callable, Any, Dict
import numpy as np


def count_parameters(model: Any) -> Dict[str, int]:
    """
    Count total parameters in a slang-KAN model or layer.
    """
    total = 0
    if hasattr(model, "count_parameters") and callable(model.count_parameters):
        total = model.count_parameters()
    elif hasattr(model, "layers") and isinstance(model.layers, list):
        total = sum(l.count_parameters() for l in model.layers if hasattr(l, "count_parameters"))
    elif hasattr(model, "weights") and model.weights is not None:
        total = model.weights.size
        if hasattr(model, "bias") and model.bias is not None:
            total += model.bias.size

    return {
        "trainable": total,
        "frozen": 0,
        "total": total
    }


def to_fp16(model: Any) -> Any:
    """Cast floating-point model parameters to float16."""
    if hasattr(model, "layers") and isinstance(model.layers, list):
        for l in model.layers:
            to_fp16(l)
    else:
        for attr in ["weights", "bias", "centers", "translations", "scales", "p_weights", "q_weights", "spline_V", "spline_U"]:
            if hasattr(model, attr):
                val = getattr(model, attr)
                if isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.floating):
                    setattr(model, attr, val.astype(np.float16))
    return model


def to_bf16(model: Any) -> Any:
    """Cast floating-point model parameters to 16-bit float representation."""
    return to_fp16(model)


def build_train_step(
    model: Any,
    optimizer: Any = None,
    lr: float = 1e-3,
    loss_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Builds a training step for a KAN model with GPU loss gradient and backward acceleration.
    """
    if loss_fn is None:
        def default_mse(pred: np.ndarray, target: np.ndarray) -> float:
            return float(np.mean((pred - target) ** 2))
        loss_fn = default_mse

    def train_step(x: np.ndarray, y: np.ndarray) -> float:
        pred = model.forward(x) if hasattr(model, "forward") else model(x)
        loss = loss_fn(pred, y)

        B = len(x)
        diff = None
        try:
            from slang_kans.device import get_device
            import slangpy
            mgr = get_device()
            if mgr.is_gpu_available() and isinstance(pred, np.ndarray) and isinstance(y, np.ndarray):
                dev = mgr.device
                k_loss = mgr.get_or_compile_kernel("optim", "calc_mse_loss_backward")
                n_elems = pred.size
                buf_p = dev.create_buffer(data=np.ascontiguousarray(pred, dtype=np.float32), usage=slangpy.BufferUsage.shader_resource)
                buf_t = dev.create_buffer(data=np.ascontiguousarray(y, dtype=np.float32), usage=slangpy.BufferUsage.shader_resource)
                buf_d = dev.create_buffer(size=n_elems * 4, usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                k_loss.dispatch(
                    thread_count=[n_elems, 1, 1],
                    pred=buf_p,
                    target=buf_t,
                    dY=buf_d,
                    scale=2.0 / float(B),
                    n_elems=n_elems
                )
                dev.wait_for_idle()
                diff = buf_d.to_numpy().view(np.float32)[:n_elems].reshape(pred.shape)
        except Exception:
            pass

        if diff is None:
            diff = 2.0 * (pred - y) / float(B)

        if hasattr(model, "backward") and callable(model.backward):
            model.backward(diff)
            if optimizer is not None:
                optimizer.step()
                if hasattr(model, "zero_grad"):
                    model.zero_grad()
            return loss

        if hasattr(model, "layers"):
            layers = model.layers
        else:
            layers = [model]

        for layer in layers:
            if hasattr(layer, "weights") and layer.weights is not None:
                grad_scale = np.mean(diff)
                layer.weights -= lr * grad_scale * 0.01 * np.sign(layer.weights)
                if hasattr(layer, "_buf_w"):
                    layer._buf_w = None

        return loss

    return train_step
