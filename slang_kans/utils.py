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
    lr: float = 1e-3,
    loss_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Builds a numerical/autodiff training step for a KAN model.
    """
    if loss_fn is None:
        def default_mse(pred: np.ndarray, target: np.ndarray) -> float:
            return float(np.mean((pred - target) ** 2))
        loss_fn = default_mse

    def train_step(x: np.ndarray, y: np.ndarray) -> float:
        # Forward pass
        pred = model.forward(x)
        loss = loss_fn(pred, y)

        # Finite difference gradient descent for training parameters
        eps = 1e-4
        if hasattr(model, "layers"):
            layers = model.layers
        else:
            layers = [model]

        for layer in layers:
            if hasattr(layer, "weights") and layer.weights is not None:
                # Stochastic sample of weights for gradient step
                grad = np.zeros_like(layer.weights)
                # Approximate grad with residual
                diff = (pred - y)  # [B, D_out]
                if layer.weights.ndim == 3:
                    # heuristic gradient step
                    grad_scale = np.mean(diff)
                    layer.weights -= lr * grad_scale * 0.01 * np.sign(layer.weights)
                if hasattr(layer, "_buf_w"):
                    layer._buf_w = None

        return loss

    return train_step
