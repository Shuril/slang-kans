"""
FourierKAN: Kolmogorov-Arnold Networks with Fourier Series in Slang.
Harmonic decomposition with trigonometric basis cos(k*pi*x), sin(k*pi*x).
Accelerated on GPU via Slang compute kernel with CPU vectorized fallback.
"""

from typing import Optional, Sequence, Union, Any, List
import numpy as np
import slangpy
from slang_kans.layers.base import SlangKANLayerBase


class FourierKANLinear(SlangKANLayerBase):
    """
    FourierKAN Linear Layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_frequencies: int = 4,
        bias: bool = True,
        use_gpu: bool = True
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.num_frequencies = num_frequencies
        self.num_bases = 2 * num_frequencies + 1
        self.use_gpu = use_gpu and self.mgr.is_gpu_available()

        scale = 1.0 / np.sqrt(in_features * self.num_bases)
        self.weights = np.random.uniform(-scale, scale, (in_features, out_features, self.num_bases)).astype(np.float32)

        # GPU cached buffers and kernel
        self._kernel = None
        self._buf_w = None
        self._buf_b = None

    def _sync_gpu_weights(self):
        dev = self.mgr.device
        if dev is None:
            return
        if self._kernel is None:
            self._kernel = self.mgr.get_or_compile_kernel("fourier", "fourier_kan_forward")
        self._buf_w = dev.create_buffer(data=self.weights, usage=slangpy.BufferUsage.shader_resource)
        bias_arr = self.bias if (self.use_bias and self.bias is not None) else np.zeros(self.out_features, dtype=np.float32)
        self._buf_b = dev.create_buffer(data=bias_arr, usage=slangpy.BufferUsage.shader_resource)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward FourierKAN evaluation."""
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        is_1d = x_arr.ndim == 1
        if is_1d:
            x_arr = x_arr.reshape(1, -1)

        B, D_in = x_arr.shape
        D_out = self.out_features

        if self.use_gpu and self.mgr.is_gpu_available():
            dev = self.mgr.device
            if self._buf_w is None or self._kernel is None:
                self._sync_gpu_weights()

            buf_x = self._get_buffer("x", B * D_in * 4, slangpy.BufferUsage.shader_resource)
            buf_x.copy_from_numpy(x_arr)

            buf_out = self._get_buffer("out", B * D_out * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)

            self._kernel.dispatch(
                thread_count=[B, (D_out + 3) // 4, 1],
                output=buf_out,
                input=buf_x,
                weights=self._buf_w,
                bias=self._buf_b,
                batch_size=B,
                in_features=D_in,
                out_features=D_out,
                num_frequencies=self.num_frequencies
            )
            dev.wait_for_idle()
            out = buf_out.to_numpy().view(np.float32)[:B * D_out].reshape(B, D_out)
            return out[0] if is_1d else out

        # CPU Fallback: 1, cos(k*pi*x), sin(k*pi*x)
        k_indices = np.arange(1, self.num_frequencies + 1, dtype=np.float32)
        # arg: [B, D_in, num_frequencies]
        arg = (np.pi * x_arr)[..., None] * k_indices[None, None, :]
        cos_terms = np.cos(arg)
        sin_terms = np.sin(arg)
        ones_term = np.ones_like(x_arr[..., None])

        bases = np.concatenate([ones_term, cos_terms, sin_terms], axis=-1)  # [B, D_in, 2K+1]
        out = np.einsum("bik,ijk->bj", bases, self.weights)

        if self.use_bias and self.bias is not None:
            out += self.bias[None, :]

        return out[0] if is_1d else out

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> float:
        l1 = np.mean(np.abs(self.weights), axis=-1)
        reg_l1 = float(np.sum(l1))
        p = l1 / (reg_l1 + 1e-8)
        entropy = float(-np.sum(p * np.log(p + 1e-8)))
        return regularize_activation * reg_l1 + regularize_entropy * entropy


class FourierKAN:
    """Multi-layer FourierKAN Network."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        num_frequencies: int = 4,
        bias: bool = True,
        use_gpu: bool = True
    ):
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            FourierKANLinear(
                in_f,
                out_f,
                num_frequencies=num_frequencies,
                bias=bias,
                use_gpu=use_gpu
            )
            for in_f, out_f in zip(self.layers_hidden, self.layers_hidden[1:])
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def count_parameters(self) -> int:
        return sum(l.count_parameters() for l in self.layers)
