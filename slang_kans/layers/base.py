"""
Base Layer for slang-KANs Architectures.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Any
import numpy as np
from slang_kans.device import get_device, SlangDeviceManager


class SlangKANLayerBase(ABC):
    """Abstract base class for all Slang-accelerated KAN layers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.mgr: SlangDeviceManager = get_device()

        # Weight tensors stored in NumPy (with optional zero-copy GPU mapping)
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        if self.use_bias:
            self.bias = np.zeros(out_features, dtype=np.float32)

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes forward pass across batch inputs."""
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        total = 0
        if self.weights is not None:
            total += self.weights.size
        if self.bias is not None:
            total += self.bias.size
        return total
