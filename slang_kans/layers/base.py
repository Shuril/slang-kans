"""
Base Layer for slang-KANs Architectures.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Any, Dict
import time
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

        self._buffer_pool: Dict[str, Any] = {}

    def _get_buffer(self, name: str, size_bytes: int, usage: int):
        dev = self.mgr.device
        if dev is None:
            return None
        buf = self._buffer_pool.get(name)
        if buf is None or buf.size < size_bytes:
            alloc_size = max(size_bytes, 1024 * 64 * 4)
            buf = dev.create_buffer(size=alloc_size, usage=usage)
            self._buffer_pool[name] = buf
        return buf

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executes forward pass across batch inputs."""
        pass

    def benchmark(self, x: np.ndarray, warmup: int = 10, iters: int = 50) -> float:
        """Benchmarks forward execution in milliseconds on GPU."""
        dev = self.mgr.device
        if dev is None or not self.mgr.is_gpu_available():
            raise RuntimeError("GPU device not available for slang benchmark.")

        for _ in range(warmup):
            self.forward(x)
        dev.wait_for_idle()

        t0 = time.perf_counter()
        for _ in range(iters):
            self.forward(x)
        dev.wait_for_idle()
        t1 = time.perf_counter()

        return ((t1 - t0) / iters) * 1000.0

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
