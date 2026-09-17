"""
ChebyKAN: Orthogonal Chebyshev Polynomial KAN Layer for Slang.
Evaluates y_j = bias_j + sum_{i} Clenshaw(x_i, w_{i, j, :}).
Accelerated on GPU via Slang compute kernel with seamless CPU fallback.
"""

from typing import Optional, Sequence, Union, Any, List
import numpy as np
import slangpy
from slang_kans.layers.base import SlangKANLayerBase


class ChebyKANLinear(SlangKANLayerBase):
    """
    Chebyshev Polynomial KAN Layer powered by Slang Clenshaw recurrence.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 4,
        bias: bool = True,
        use_gpu: bool = True
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.degree = degree
        self.kernel_name = "cheby"
        self.use_gpu = use_gpu and self.mgr.is_gpu_available()

        scale = 1.0 / np.sqrt(in_features * (degree + 1))
        self.weights = np.random.uniform(-scale, scale, (in_features, out_features, degree + 1)).astype(np.float32)

        self._kernel = None
        self._buf_w = None
        self._buf_b = None

    def _sync_gpu_weights(self):
        dev = self.mgr.device
        if dev is None:
            return
        if self._kernel is None:
            self._kernel = self.mgr.get_or_compile_kernel("cheby", "cheby_kan_forward")
        self._buf_w = dev.create_buffer(data=self.weights, usage=slangpy.BufferUsage.shader_resource)
        bias_arr = self.bias if (self.use_bias and self.bias is not None) else np.zeros(self.out_features, dtype=np.float32)
        self._buf_b = dev.create_buffer(data=bias_arr, usage=slangpy.BufferUsage.shader_resource)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        is_1d = x_arr.ndim == 1
        if is_1d:
            x_arr = x_arr.reshape(1, -1)

        B, D_in = x_arr.shape
        D_out = self.out_features
        deg = self.degree

        if self.use_gpu and self.mgr.is_gpu_available():
            dev = self.mgr.device
            if self._buf_w is None or self._kernel is None:
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
                weights=self._buf_w,
                bias=self._buf_b,
                batch_size=B,
                in_features=D_in,
                out_features=D_out,
                degree=deg
            )
            dev.wait_for_idle()
            out = buf_out.to_numpy().view(np.float32).reshape(B, D_out)
            return out[0] if is_1d else out

        # CPU Fallback: Clenshaw recurrence across batch and features
        s = np.clip(x_arr, -1.0, 1.0)
        b2 = np.zeros((B, D_in, D_out), dtype=np.float32)
        b1 = np.zeros((B, D_in, D_out), dtype=np.float32)
        two_s = 2.0 * s[..., None]

        for k in range(deg, 0, -1):
            w_k = self.weights[:, :, k]
            b0 = two_s * b1 - b2 + w_k[None, :, :]
            b2 = b1
            b1 = b0

        w_0 = self.weights[:, :, 0]
        phi = s[..., None] * b1 - b2 + w_0[None, :, :]
        out = np.sum(phi, axis=1)

        if self.use_bias and self.bias is not None:
            out += self.bias[None, :]

        return out[0] if is_1d else out

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> float:
        l1 = np.mean(np.abs(self.weights), axis=-1)
        reg_l1 = float(np.sum(l1))
        p = l1 / (reg_l1 + 1e-8)
        entropy = float(-np.sum(p * np.log(p + 1e-8)))
        return regularize_activation * reg_l1 + regularize_entropy * entropy


class ChebyKAN:
    """
    ChebyKAN wrapper: behaves as ChebyKANLinear when passed (in_f, out_f),
    or as multi-layer ChebyKAN network when passed layers_hidden: Sequence[int].
    """

    def __new__(cls, *args, **kwargs):
        if len(args) >= 1 and isinstance(args[0], (list, tuple)):
            # Multi-layer network
            instance = super().__new__(cls)
            instance.__init_net__(*args, **kwargs)
            return instance
        # Single layer
        return ChebyKANLinear(*args, **kwargs)

    def __init_net__(
        self,
        layers_hidden: Sequence[int],
        degree: int = 4,
        bias: bool = True,
        use_gpu: bool = True
    ):
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            ChebyKANLinear(
                in_f,
                out_f,
                degree=degree,
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
