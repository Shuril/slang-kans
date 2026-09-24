"""
GatedKAN: Drop-in Replacement for SwiGLU / MLP Transformer Blocks in Slang.
Accelerated on GPU with fused SwiGLU compute shaders and elementwise addition.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Union, Tuple, Sequence

from slang_kans.layers.cheby_kan import ChebyKANLinear
from slang_kans.quantized import to_int8, to_int4, to_ternary, to_int2
from slang_kans.device import get_device

try:
    import slangpy
except ImportError:
    slangpy = None


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _d_silu(x: np.ndarray) -> np.ndarray:
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))
    return sig * (1.0 + x * (1.0 - sig))


class GatedKAN:
    """
    Gated Kolmogorov-Arnold Network (GatedKAN) block.
    Computes:
        gate = Gate_KAN(x)
        up = Up_KAN(x)
        hidden = SiLU(gate) * up
        out = Down_KAN(hidden)

    Accelerated with Slang compute shaders (swiglu_forward_f32, swiglu_backward_f32, elementwise_add_f32).
    """
    def __init__(
        self,
        d_model: int,
        d_ffn: Optional[int] = None,
        degree: int = 4,
        has_bias: bool = False,
        use_gpu: bool = True,
        dtype: np.dtype = np.float32,
    ):
        self.d_model = int(d_model)
        self.d_ffn = int(d_ffn if d_ffn is not None else 2 * d_model)
        self.degree = int(degree)
        self.has_bias = bool(has_bias)
        self.use_gpu = bool(use_gpu)
        self.dtype = np.dtype(dtype)

        self.gate_kan = ChebyKANLinear(
            self.d_model,
            self.d_ffn,
            degree=self.degree,
            bias=self.has_bias,
            use_gpu=self.use_gpu
        )
        self.up_kan = ChebyKANLinear(
            self.d_model,
            self.d_ffn,
            degree=self.degree,
            bias=self.has_bias,
            use_gpu=self.use_gpu
        )
        self.down_kan = ChebyKANLinear(
            self.d_ffn,
            self.d_model,
            degree=self.degree,
            bias=self.has_bias,
            use_gpu=self.use_gpu
        )

        self._saved_x: Optional[np.ndarray] = None
        self._saved_gate: Optional[np.ndarray] = None
        self._saved_up: Optional[np.ndarray] = None
        self._saved_hidden: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        orig_shape = x.shape
        if len(orig_shape) > 2:
            x_2d = x.reshape(-1, self.d_model)
        else:
            x_2d = x

        self._saved_x = x_2d

        gate = self.gate_kan.forward(x_2d)
        up = self.up_kan.forward(x_2d)

        self._saved_gate = gate
        self._saved_up = up

        mgr = None
        if self.use_gpu:
            try:
                mgr = get_device()
                if not mgr.is_gpu_available():
                    mgr = None
            except Exception:
                mgr = None

        if mgr is not None and self.dtype == np.float32 and gate.dtype == np.float32 and up.dtype == np.float32:
            try:
                dev = mgr.device
                kernel = mgr.get_or_compile_kernel("swiglu", "swiglu_forward_f32")
                n_elems = gate.size

                buf_g = dev.create_buffer(data=np.ascontiguousarray(gate), usage=slangpy.BufferUsage.shader_resource)
                buf_u = dev.create_buffer(data=np.ascontiguousarray(up), usage=slangpy.BufferUsage.shader_resource)
                buf_out = dev.create_buffer(size=n_elems * 4, usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)

                kernel.dispatch(
                    thread_count=[n_elems, 1, 1],
                    gate=buf_g,
                    up=buf_u,
                    output=buf_out,
                    n_elems=n_elems
                )
                dev.wait_for_idle()
                hidden = buf_out.to_numpy().view(np.float32)[:n_elems].reshape(gate.shape)
                self._saved_hidden = hidden
                out = self.down_kan.forward(hidden)
                if len(orig_shape) > 2:
                    return out.reshape(*orig_shape[:-1], self.d_model)
                return out
            except Exception:
                pass

        # CPU Fallback
        silu_gate = _silu(gate)
        hidden = silu_gate * up
        self._saved_hidden = hidden

        out = self.down_kan.forward(hidden)
        if len(orig_shape) > 2:
            return out.reshape(*orig_shape[:-1], self.d_model)
        return out

    __call__ = forward

    def backward(self, dY: np.ndarray) -> np.ndarray:
        if self._saved_x is None:
            raise RuntimeError("Cannot run backward before forward() has been called.")

        orig_shape = dY.shape
        if len(orig_shape) > 2:
            dY_2d = dY.reshape(-1, self.d_model)
        else:
            dY_2d = dY

        d_hidden = self.down_kan.backward(dY_2d)

        mgr = None
        if self.use_gpu:
            try:
                mgr = get_device()
                if not mgr.is_gpu_available():
                    mgr = None
            except Exception:
                mgr = None

        if (
            mgr is not None
            and self.dtype == np.float32
            and d_hidden.dtype == np.float32
            and self._saved_gate.dtype == np.float32
            and self._saved_up.dtype == np.float32
        ):
            try:
                dev = mgr.device
                k_bwd = mgr.get_or_compile_kernel("swiglu", "swiglu_backward_f32")
                k_add = mgr.get_or_compile_kernel("swiglu", "elementwise_add_f32")
                n_elems = d_hidden.size

                buf_dh = dev.create_buffer(data=np.ascontiguousarray(d_hidden), usage=slangpy.BufferUsage.shader_resource)
                buf_g = dev.create_buffer(data=np.ascontiguousarray(self._saved_gate), usage=slangpy.BufferUsage.shader_resource)
                buf_u = dev.create_buffer(data=np.ascontiguousarray(self._saved_up), usage=slangpy.BufferUsage.shader_resource)
                buf_dg = dev.create_buffer(size=n_elems * 4, usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                buf_du = dev.create_buffer(size=n_elems * 4, usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)

                k_bwd.dispatch(
                    thread_count=[n_elems, 1, 1],
                    d_hidden=buf_dh,
                    gate=buf_g,
                    up=buf_u,
                    d_gate=buf_dg,
                    d_up=buf_du,
                    n_elems=n_elems
                )
                dev.wait_for_idle()

                d_gate = buf_dg.to_numpy().view(np.float32)[:n_elems].reshape(self._saved_gate.shape)
                d_up = buf_du.to_numpy().view(np.float32)[:n_elems].reshape(self._saved_up.shape)

                dX_gate = self.gate_kan.backward(d_gate)
                dX_up = self.up_kan.backward(d_up)

                n_x = dX_gate.size
                buf_dxg = dev.create_buffer(data=np.ascontiguousarray(dX_gate), usage=slangpy.BufferUsage.shader_resource)
                buf_dxu = dev.create_buffer(data=np.ascontiguousarray(dX_up), usage=slangpy.BufferUsage.shader_resource)
                buf_dx = dev.create_buffer(size=n_x * 4, usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)

                k_add.dispatch(
                    thread_count=[n_x, 1, 1],
                    a=buf_dxg,
                    b=buf_dxu,
                    output=buf_dx,
                    n_elems=n_x
                )
                dev.wait_for_idle()
                dX = buf_dx.to_numpy().view(np.float32)[:n_x].reshape(dX_gate.shape)

                if len(orig_shape) > 2:
                    return dX.reshape(orig_shape)
                return dX
            except Exception:
                pass

        # CPU Fallback
        gate = self._saved_gate
        up = self._saved_up
        silu_gate = _silu(gate)
        d_silu_gate = _d_silu(gate)

        d_up = d_hidden * silu_gate
        d_gate = d_hidden * up * d_silu_gate

        dX_gate = self.gate_kan.backward(d_gate)
        dX_up = self.up_kan.backward(d_up)
        dX = dX_gate + dX_up

        if len(orig_shape) > 2:
            return dX.reshape(orig_shape)
        return dX

    def zero_grad(self) -> None:
        self.gate_kan.zero_grad()
        self.up_kan.zero_grad()
        self.down_kan.zero_grad()

    def get_trainable_params(self) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        params = []
        for kan in (self.gate_kan, self.up_kan, self.down_kan):
            params.append((kan.weights, kan.grad_weights))
            if kan.use_bias and kan.bias is not None and kan.grad_bias is not None:
                params.append((kan.bias, kan.grad_bias))
        return params

    def to_int8(self, group_size: int = 64) -> GatedKAN:
        to_int8(self.gate_kan, group_size=group_size)
        to_int8(self.up_kan, group_size=group_size)
        to_int8(self.down_kan, group_size=group_size)
        return self

    def to_int4(self, group_size: int = 64) -> GatedKAN:
        to_int4(self.gate_kan, group_size=group_size)
        to_int4(self.up_kan, group_size=group_size)
        to_int4(self.down_kan, group_size=group_size)
        return self

    def to_ternary(self, group_size: int = 32) -> GatedKAN:
        to_ternary(self.gate_kan, group_size=group_size)
        to_ternary(self.up_kan, group_size=group_size)
        to_ternary(self.down_kan, group_size=group_size)
        return self

    def to_int2(self, group_size: int = 32) -> GatedKAN:
        to_int2(self.gate_kan, group_size=group_size)
        to_int2(self.up_kan, group_size=group_size)
        to_int2(self.down_kan, group_size=group_size)
        return self

    def count_parameters(self) -> int:
        return self.gate_kan.count_parameters() + self.up_kan.count_parameters() + self.down_kan.count_parameters()

    def __repr__(self) -> str:
        return f"GatedKAN(d_model={self.d_model}, d_ffn={self.d_ffn}, degree={self.degree}, params={self.count_parameters():,})"
