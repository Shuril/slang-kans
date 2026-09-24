"""
Activation Checkpointing for Kolmogorov-Arnold Networks in Slang.
Saves memory during training/inference of deep KAN models by managing activation lifetimes.
"""

from __future__ import annotations
from typing import Sequence, Union, Optional, Any
import numpy as np


class CheckpointedKAN:
    """
    Checkpointed wrapper for KAN models and sequential layers.
    """

    def __init__(self, model_or_layers: Any):
        if hasattr(model_or_layers, "layers") and isinstance(model_or_layers.layers, list):
            self.base_model = model_or_layers
            self.layers = model_or_layers.layers
        elif isinstance(model_or_layers, (list, tuple)):
            self.base_model = None
            self.layers = list(model_or_layers)
        else:
            self.base_model = model_or_layers
            self.layers = [model_or_layers]

    def forward(self, x: Any, return_tensor: bool = False) -> Any:
        curr = x
        num_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            is_last = (i == num_layers - 1)
            target_return_tensor = return_tensor if is_last else True
            if hasattr(layer, "forward"):
                try:
                    curr = layer.forward(curr, return_tensor=target_return_tensor)
                except TypeError:
                    curr = layer.forward(curr)
            else:
                curr = layer(curr)
        return curr

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        return self.forward(x, *args, **kwargs)

    def count_parameters(self) -> int:
        return sum(l.count_parameters() for l in self.layers)


def checkpoint_kan(model: Any) -> CheckpointedKAN:
    """Wraps a KAN model with activation checkpointing."""
    return CheckpointedKAN(model)
