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

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        return self.forward(x, *args, **kwargs)

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        total = 0
        if self.weights is not None:
            total += self.weights.size
        if self.bias is not None:
            total += self.bias.size
        return total

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Returns layer parameters as dictionary of NumPy arrays."""
        sd = {}
        if hasattr(self, "weights") and self.weights is not None:
            sd["weights"] = np.array(self.weights, copy=True)
        if hasattr(self, "bias") and self.bias is not None:
            sd["bias"] = np.array(self.bias, copy=True)
        if hasattr(self, "base_weights") and self.base_weights is not None:
            sd["base_weights"] = np.array(self.base_weights, copy=True)
        if hasattr(self, "centers") and self.centers is not None:
            sd["centers"] = np.array(self.centers, copy=True)
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads layer parameters from state dict."""
        if "weights" in state_dict and hasattr(self, "weights"):
            self.weights = np.ascontiguousarray(state_dict["weights"], dtype=np.float32)
            if hasattr(self, "_sync_gpu_weights"):
                self._sync_gpu_weights()
        if "bias" in state_dict and hasattr(self, "bias"):
            self.bias = np.ascontiguousarray(state_dict["bias"], dtype=np.float32)
            if hasattr(self, "_sync_gpu_weights"):
                self._sync_gpu_weights()
        if "base_weights" in state_dict and hasattr(self, "base_weights"):
            self.base_weights = np.ascontiguousarray(state_dict["base_weights"], dtype=np.float32)
            if hasattr(self, "_sync_gpu_weights"):
                self._sync_gpu_weights()

    def save_weights(self, filepath: str):
        """Saves layer weights to .safetensors (if available) or .npz."""
        sd = self.state_dict()
        if filepath.endswith(".safetensors"):
            try:
                import safetensors.numpy as st
                st.save_file(sd, filepath)
                return
            except ImportError:
                filepath = filepath[:-12] + ".npz"
        np.savez(filepath, **sd)

    def load_weights(self, filepath: str):
        """Loads layer weights from .safetensors or .npz."""
        if filepath.endswith(".safetensors"):
            try:
                import safetensors.numpy as st
                sd = st.load_file(filepath)
                self.load_state_dict(sd)
                return
            except ImportError:
                filepath = filepath[:-12] + ".npz"
        with np.load(filepath) as data:
            sd = {k: data[k] for k in data.files}
            self.load_state_dict(sd)
