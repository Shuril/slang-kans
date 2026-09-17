"""
RationalKAN: Kolmogorov-Arnold Networks with Padé-Chebyshev Rational Functions in Slang.
phi(x) = P_p(x) / (1 + |Q_q(x)|) where P and Q are Chebyshev series.
Accelerated on GPU via Slang compute kernel with CPU vectorized fallback.
"""

from typing import Optional, Sequence, Union, Any, List
import numpy as np
import slangpy
from slang_kans.layers.base import SlangKANLayerBase


class RationalKANLinear(SlangKANLayerBase):
    """
    Rational Padé-Chebyshev KAN Layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        p_degree: int = 4,
        q_degree: int = 2,
        bias: bool = True,
        use_gpu: bool = True
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.p_degree = p_degree
        self.q_degree = q_degree
        self.use_gpu = use_gpu and self.mgr.is_gpu_available()

        # Numerator weights P and denominator weights Q
        scale_p = 1.0 / np.sqrt(in_features * (p_degree + 1))
        scale_q = 0.1 / np.sqrt(in_features * (q_degree + 1))
        self.p_weights = np.random.uniform(-scale_p, scale_p, (in_features, out_features, p_degree + 1)).astype(np.float32)
        self.q_weights = np.random.uniform(-scale_q, scale_q, (in_features, out_features, q_degree + 1)).astype(np.float32)

        # For base class weights tracking
        self.weights = self.p_weights

        # GPU cached buffers and kernel
        self._kernel = None
        self._buf_p = None
        self._buf_q = None
        self._buf_b = None

    def count_parameters(self) -> int:
        total = self.p_weights.size + self.q_weights.size
        if self.use_bias and self.bias is not None:
            total += self.bias.size
        return total

    def _sync_gpu_weights(self):
        dev = self.mgr.device
        if dev is None:
            return
        if self._kernel is None:
            self._kernel = self.mgr.get_or_compile_kernel("rational", "rational_kan_forward")
        self._buf_p = dev.create_buffer(data=self.p_weights, usage=slangpy.BufferUsage.shader_resource)
        self._buf_q = dev.create_buffer(data=self.q_weights, usage=slangpy.BufferUsage.shader_resource)
        bias_arr = self.bias if (self.use_bias and self.bias is not None) else np.zeros(self.out_features, dtype=np.float32)
        self._buf_b = dev.create_buffer(data=bias_arr, usage=slangpy.BufferUsage.shader_resource)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward RationalKAN evaluation."""
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        is_1d = x_arr.ndim == 1
        if is_1d:
            x_arr = x_arr.reshape(1, -1)

        B, D_in = x_arr.shape
        D_out = self.out_features

        if self.use_gpu and self.mgr.is_gpu_available():
            dev = self.mgr.device
            if self._buf_p is None or self._kernel is None:
                self._sync_gpu_weights()

            buf_x = dev.create_buffer(data=x_arr, usage=slangpy.BufferUsage.shader_resource)
            out_arr = np.zeros((B, D_out), dtype=np.float32)
            buf_out = dev.create_buffer(
                data=out_arr,
                usage=slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access
            )

            self._kernel.dispatch(
                thread_count=[B, D_out, 1],
                output=buf_out,
                input=buf_x,
                p_weights=self._buf_p,
                q_weights=self._buf_q,
                bias=self._buf_b,
                batch_size=B,
                in_features=D_in,
                out_features=D_out,
                p_degree=self.p_degree,
                q_degree=self.q_degree
            )
            dev.wait_for_idle()
            out = buf_out.to_numpy().view(np.float32).reshape(B, D_out)
            return out[0] if is_1d else out

        # CPU Fallback via Clenshaw
        s = np.clip(x_arr, -1.0, 1.0)
        two_s = 2.0 * s[..., None]

        # Evaluate P(s)
        b2 = np.zeros((B, D_in, D_out), dtype=np.float32)
        b1 = np.zeros((B, D_in, D_out), dtype=np.float32)
        for k in range(self.p_degree, 0, -1):
            w_k = self.p_weights[:, :, k]
            b0 = two_s * b1 - b2 + w_k[None, :, :]
            b2 = b1
            b1 = b0
        p_val = s[..., None] * b1 - b2 + self.p_weights[:, :, 0][None, :, :]

        # Evaluate Q(s)
        b2_q = np.zeros((B, D_in, D_out), dtype=np.float32)
        b1_q = np.zeros((B, D_in, D_out), dtype=np.float32)
        for k in range(self.q_degree, 0, -1):
            w_k = self.q_weights[:, :, k]
            b0 = two_s * b1_q - b2_q + w_k[None, :, :]
            b2_q = b1_q
            b1_q = b0
        q_val = s[..., None] * b1_q - b2_q + self.q_weights[:, :, 0][None, :, :]

        phi = p_val / (1.0 + np.abs(q_val))
        out = np.sum(phi, axis=1)

        if self.use_bias and self.bias is not None:
            out += self.bias[None, :]

        return out[0] if is_1d else out

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> float:
        l1 = np.mean(np.abs(self.p_weights), axis=-1)
        reg_l1 = float(np.sum(l1))
        p = l1 / (reg_l1 + 1e-8)
        entropy = float(-np.sum(p * np.log(p + 1e-8)))
        return regularize_activation * reg_l1 + regularize_entropy * entropy


class RationalKAN:
    """Multi-layer RationalKAN Network."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        p_degree: int = 4,
        q_degree: int = 2,
        bias: bool = True,
        use_gpu: bool = True
    ):
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            RationalKANLinear(
                in_f,
                out_f,
                p_degree=p_degree,
                q_degree=q_degree,
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
