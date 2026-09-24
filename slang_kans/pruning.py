"""
Structural Pruning and Node Compaction for slang-KANs.
Computes node importance and physically compacts inactive neurons to shrink model size.
"""

from __future__ import annotations
from typing import Sequence, Union, List, Dict, Any, Tuple
import numpy as np


def _get_layer_weights(layer: Any) -> Tuple[np.ndarray, int]:
    """Extract basis weight and basis dimension from any KAN layer."""
    if hasattr(layer, "weights") and layer.weights is not None:
        w = layer.weights
        num_basis = w.shape[2] if w.ndim == 3 else 1
        return w, num_basis
    elif hasattr(layer, "p_weights"):
        return layer.p_weights, layer.p_weights.shape[2]
    elif hasattr(layer, "spline_V"):
        return layer.spline_V, 1
    raise ValueError(f"Layer {type(layer).__name__} does not have recognizable weights.")


def compute_node_importance(model: Any) -> List[np.ndarray]:
    """
    Calculate the importance score for each hidden neuron across all hidden layers.
    I_j = magnitude_out(layer_l, j) * magnitude_in(layer_{l+1}, j)
    """
    if not hasattr(model, "layers") or len(model.layers) < 2:
        raise ValueError("Model must possess at least 2 layers to compute hidden node importance.")

    layers = model.layers
    num_hidden = len(layers) - 1
    importances = []

    for l_idx in range(num_hidden):
        l_curr = layers[l_idx]
        l_next = layers[l_idx + 1]

        w_curr, _ = _get_layer_weights(l_curr)
        w_next, _ = _get_layer_weights(l_next)

        # Outgoing magnitude from l_curr (shape: out_features)
        if w_curr.ndim == 3:
            out_mag = np.mean(np.abs(w_curr), axis=(0, 2))
        else:
            out_mag = np.mean(np.abs(w_curr), axis=0)

        # Incoming magnitude to l_next (shape: in_features)
        if w_next.ndim == 3:
            in_mag = np.mean(np.abs(w_next), axis=(1, 2))
        else:
            in_mag = np.mean(np.abs(w_next), axis=1)

        importance = out_mag * in_mag
        importances.append(importance)

    return importances


def prune(model: Any, threshold: float = 1e-3) -> Dict[str, Any]:
    """
    Zeros out edges whose weight magnitude falls below threshold.
    Returns summary statistics of pruned connections.
    """
    total_weights = 0
    pruned_weights = 0

    if hasattr(model, "layers"):
        for layer in model.layers:
            if hasattr(layer, "weights") and layer.weights is not None:
                mask = np.abs(layer.weights) < threshold
                pruned_weights += int(np.sum(mask))
                total_weights += layer.weights.size
                layer.weights[mask] = 0.0
                # Invalidate GPU buffer cache
                if hasattr(layer, "_buf_w"):
                    layer._buf_w = None
    elif hasattr(model, "weights") and model.weights is not None:
        mask = np.abs(model.weights) < threshold
        pruned_weights += int(np.sum(mask))
        total_weights += model.weights.size
        model.weights[mask] = 0.0
        if hasattr(model, "_buf_w"):
            model._buf_w = None

    ratio = pruned_weights / max(total_weights, 1)
    return {
        "total_weights": total_weights,
        "pruned_weights": pruned_weights,
        "sparsity": float(ratio)
    }


def compact_kan(model: Any, threshold: float = 1e-3) -> Any:
    """
    Removes dead neurons with importance below threshold.
    """
    if not hasattr(model, "layers") or len(model.layers) < 2:
        return model

    importances = compute_node_importance(model)
    # Filter layers
    for l_idx, imp in enumerate(importances):
        active_indices = np.where(imp >= threshold)[0]
        if len(active_indices) == 0:
            active_indices = np.array([np.argmax(imp)])

        l_curr = model.layers[l_idx]
        l_next = model.layers[l_idx + 1]

        # Slice outgoing weights of l_curr
        if hasattr(l_curr, "weights") and l_curr.weights is not None:
            l_curr.weights = l_curr.weights[:, active_indices, :]
            l_curr.out_features = len(active_indices)
            if l_curr.use_bias and l_curr.bias is not None:
                l_curr.bias = l_curr.bias[active_indices]
            l_curr._buf_w = None
            l_curr._buf_b = None

        # Slice incoming weights of l_next
        if hasattr(l_next, "weights") and l_next.weights is not None:
            l_next.weights = l_next.weights[active_indices, :, :]
            l_next.in_features = len(active_indices)
            l_next._buf_w = None

    return model
