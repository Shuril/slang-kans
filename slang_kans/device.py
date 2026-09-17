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
