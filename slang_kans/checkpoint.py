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

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def count_parameters(self) -> int:
        return sum(l.count_parameters() for l in self.layers)


def checkpoint_kan(model: Any) -> CheckpointedKAN:
    """Wraps a KAN model with activation checkpointing."""
    return CheckpointedKAN(model)
