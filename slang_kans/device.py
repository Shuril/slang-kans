"""
Device and Runtime Manager for slang-KANs.
Manages cross-platform GPU selection (Vulkan / Metal / CUDA / CPU) and kernel compilation.
On Apple Silicon macOS, leverages Vulkan (MoltenVK) for zero-dependency native GPU acceleration.
"""

import os
import platform
from typing import Optional, Dict, Any
import numpy as np

_SLANG_AVAILABLE = False
_SLANG_ERROR = None
try:
    import slangpy
    _SLANG_AVAILABLE = True
except ImportError as e:
    slangpy = None
    _SLANG_ERROR = str(e)


def is_slang_available() -> bool:
    """Returns True if slangpy runtime is installed and importable."""
    return _SLANG_AVAILABLE


class SlangTensor:
    """
    GPU-resident tensor for slang-KANs.
    Enables zero-copy chaining across multiple layers on GPU without host roundtrips.
    """
    def __init__(self, buffer: Any, shape: tuple[int, ...], dtype=np.float32):
        self.buffer = buffer
        self.shape = tuple(shape)
        self.dtype = dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numpy(self) -> np.ndarray:
        """Transfers buffer data from GPU to CPU NumPy array."""
        count = int(np.prod(self.shape))
        return self.buffer.to_numpy().view(self.dtype)[:count].reshape(self.shape)

    to_numpy = numpy

    def __array__(self) -> np.ndarray:
        return self.numpy()

    def to_mlx(self) -> Any:
        """Converts to Apple Silicon MLX array."""
        import mlx.core as mx
        return mx.array(self.numpy())

    def to_torch(self) -> Any:
        """Converts to PyTorch tensor."""
        import torch
        return torch.from_numpy(self.numpy())

    def __dlpack__(self, stream: Optional[Any] = None) -> Any:
        """DLPack protocol for zero-copy interop with PyTorch, JAX, and MLX."""
        arr = self.numpy()
        if hasattr(arr, "__dlpack__"):
            return arr.__dlpack__(stream=stream)
        raise NotImplementedError("DLPack protocol not supported by underlying array.")

    def __dlpack_device__(self) -> tuple[int, int]:
        """DLPack device type identifier."""
        return (1, 0)

    def __repr__(self) -> str:
        return f"SlangTensor(shape={self.shape}, dtype={self.dtype.__name__ if hasattr(self.dtype, '__name__') else str(self.dtype)}, device=GPU)"


def to_tensor(x: Any) -> SlangTensor:
    """Uploads NumPy array, MLX array, PyTorch tensor, or array-like to GPU SlangTensor."""
    if isinstance(x, SlangTensor):
        return x
    mgr = get_device()
    dev = mgr.device
    if dev is None:
        raise RuntimeError("Slang GPU device is not available.")
    if hasattr(x, "numpy") and callable(x.numpy):
        arr = np.ascontiguousarray(x.numpy(), dtype=np.float32)
    elif hasattr(x, "__array__"):
        arr = np.ascontiguousarray(np.array(x), dtype=np.float32)
    else:
        arr = np.ascontiguousarray(x, dtype=np.float32)
    buf = dev.create_buffer(data=arr, usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
    return SlangTensor(buf, arr.shape, arr.dtype)


class SlangDeviceManager:
    """
    Singleton Manager for Slang GPU / CPU device, module cache, and compute kernels.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SlangDeviceManager, cls).__new__(cls)
            cls._instance._device = None
            cls._instance._module_cache = {}
            cls._instance._kernel_cache = {}
            cls._instance._init_device()
        return cls._instance

    def _init_device(self):
        if not _SLANG_AVAILABLE:
            return

        # On macOS, Vulkan (MoltenVK) compiles and executes directly on Apple Silicon GPU
        # without requiring standalone Xcode command-line tool 'metal'.
        backends_to_try = []
        if platform.system() == "Darwin":
            backends_to_try = [slangpy.DeviceType.vulkan, slangpy.DeviceType.metal, slangpy.DeviceType.automatic]
        else:
            backends_to_try = [slangpy.DeviceType.automatic, slangpy.DeviceType.cuda, slangpy.DeviceType.vulkan]

        for b in backends_to_try:
            try:
                dev = slangpy.create_device(type=b)
                slangpy.push_current_device(dev)
                self._device = dev
                break
            except Exception:
                continue

        if self._device is None:
            try:
                self._device = slangpy.create_device(type=slangpy.DeviceType.cpu)
                slangpy.push_current_device(self._device)
            except Exception:
                self._device = None

    @property
    def device(self) -> Any:
        return self._device

    @property
    def device_type(self) -> str:
        if self._device is None:
            return "none"
        return str(self._device.info.type)

    def is_gpu_available(self) -> bool:
        if self._device is None:
            return False
        dt = str(self._device.info.type).lower()
        return "cpu" not in dt and "none" not in dt

    def load_kernel_source(self, kernel_name: str) -> str:
        """Loads Slang shader source code from kernels/ directory."""
        kernel_dir = os.path.join(os.path.dirname(__file__), "kernels")
        path = os.path.join(kernel_dir, f"{kernel_name}.slang")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Slang kernel '{kernel_name}.slang' not found at {path}")
        with open(path, "r") as f:
            return f.read()

    def get_or_compile_module(self, kernel_name: str) -> Any:
        """Loads and compiles Slang module for the active target device."""
        if self._device is None:
            raise RuntimeError(f"Slang device is not available: {_SLANG_ERROR}")

        if kernel_name in self._module_cache:
            return self._module_cache[kernel_name]

        src = self.load_kernel_source(kernel_name)
        mod = self._device.load_module_from_source(kernel_name, src)
        self._module_cache[kernel_name] = mod
        return mod

    def get_or_compile_kernel(self, kernel_name: str, entry_point_name: Optional[str] = None) -> Any:
        """Returns cached ComputeKernel compiled for active device."""
        cache_key = f"{kernel_name}::{entry_point_name or 'default'}"
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        mod = self.get_or_compile_module(kernel_name)
        ep = mod.entry_points[0]
        if entry_point_name:
            for candidate in mod.entry_points:
                if candidate.name == entry_point_name:
                    ep = candidate
                    break

        prog = self._device.link_program([mod], [ep])
        kernel = self._device.create_compute_kernel(prog)
        self._kernel_cache[cache_key] = kernel
        return kernel


def get_device() -> SlangDeviceManager:
    """Returns the global SlangDeviceManager instance."""
    return SlangDeviceManager()
