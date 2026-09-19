"""
ReLUKAN: Kolmogorov-Arnold Networks with Piecewise Linear Tent Functions in Slang.
Replaces splines with piecewise-linear tent functions max(0, 1 - |x - c| * inv_h).
Accelerated on GPU via Slang compute kernel with CPU vectorized fallback.
"""

from typing import Optional, Sequence, Union, Any, List
import numpy as np
import slangpy
from slang_kans.layers.base import SlangKANLayerBase


class ReLUKANLinear(SlangKANLayerBase):
    """
    ReLUKAN Linear Layer using piecewise linear tent functions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_grids: int = 8,
        grid_range: tuple = (-1.0, 1.0),
        bias: bool = True,
        use_gpu: bool = True
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.num_grids = num_grids
        self.grid_min, self.grid_max = grid_range
        self.use_gpu = use_gpu and self.mgr.is_gpu_available()

        # Centers
        self.centers = np.linspace(self.grid_min, self.grid_max, num_grids, dtype=np.float32)
        h = (self.grid_max - self.grid_min) / max(num_grids - 1, 1)
        self.inv_h = np.float32(1.0 / max(h, 1e-6))

        scale = 1.0 / np.sqrt(in_features * num_grids)
        self.weights = np.random.uniform(-scale, scale, (in_features, out_features, num_grids)).astype(np.float32)

        # GPU cached buffers and kernel
        self._kernel = None
        self._kernel_basis = None
        self._kernel_gemm = None
        self._buf_w = None
        self._buf_w_gemm = None
        self._buf_b = None

    def _sync_gpu_weights(self):
        dev = self.mgr.device
        if dev is None:
            return
        if self._kernel is None:
            self._kernel = self.mgr.get_or_compile_kernel("relu", "relu_kan_forward")
            self._kernel_basis = self.mgr.get_or_compile_kernel("relu", "eval_relu_basis")
            self._kernel_gemm = self.mgr.get_or_compile_kernel("gemm", "gemm_tiled")

        self._buf_w = dev.create_buffer(data=self.weights, usage=slangpy.BufferUsage.shader_resource)
        w_mat = np.ascontiguousarray(self.weights.transpose(1, 0, 2).reshape(self.out_features, -1))
        self._buf_w_gemm = dev.create_buffer(data=w_mat, usage=slangpy.BufferUsage.shader_resource)

        bias_arr = self.bias if (self.use_bias and self.bias is not None) else np.zeros(self.out_features, dtype=np.float32)
        self._buf_b = dev.create_buffer(data=bias_arr, usage=slangpy.BufferUsage.shader_resource)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward ReLUKAN tent basis evaluation."""
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

            if B >= 64:
                K_dim = D_in * self.num_grids
                buf_phi = self._get_buffer("phi", B * K_dim * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                self._kernel_basis.dispatch(
                    thread_count=[B, D_in, 1],
                    phi=buf_phi,
                    input=buf_x,
                    batch_size=B,
                    in_features=D_in,
                    num_grids=self.num_grids,
                    grid_min=float(self.grid_min),
                    grid_max=float(self.grid_max)
                )
                self._kernel_gemm.dispatch(
                    thread_count=[(D_out + 15) // 16 * 16, (B + 15) // 16 * 16, 1],
                    C=buf_out,
                    A=buf_phi,
                    B_mat=self._buf_w_gemm,
                    bias=self._buf_b,
                    M=B,
                    N=D_out,
                    K=K_dim,
                    has_bias=1 if self.use_bias else 0
                )
            else:
                self._kernel.dispatch(
                    thread_count=[B, (D_out + 3) // 4, 1],
                    output=buf_out,
                    input=buf_x,
                    weights=self._buf_w,
                    bias=self._buf_b,
                    batch_size=B,
                    in_features=D_in,
                    out_features=D_out,
                    num_grids=self.num_grids,
                    grid_min=float(self.grid_min),
                    grid_max=float(self.grid_max)
                )

            dev.wait_for_idle()
            out = buf_out.to_numpy().view(np.float32)[:B * D_out].reshape(B, D_out)
            return out[0] if is_1d else out

        # CPU fallback: max(0, 1 - |x - c| * inv_h)
        diff = np.abs(x_arr[..., None] - self.centers[None, None, :]) * self.inv_h
        bases = np.maximum(0.0, 1.0 - diff)
        out = np.einsum("bic,ijc->bj", bases, self.weights)

        if self.use_bias and self.bias is not None:
            out += self.bias[None, :]

        return out[0] if is_1d else out

    def _benchmark_gpu(self, x: np.ndarray, warmup: int = 10, iters: int = 50) -> float:
        dev = self.mgr.device
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        B, D_in = x_arr.shape
        D_out = self.out_features

        if self._buf_w is None or self._kernel is None:
            self._sync_gpu_weights()

        buf_x = self._get_buffer("x", B * D_in * 4, slangpy.BufferUsage.shader_resource)
        buf_x.copy_from_numpy(x_arr)
        buf_out = self._get_buffer("out", B * D_out * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
        K_dim = D_in * self.num_grids
        buf_phi = self._get_buffer("phi", B * K_dim * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)

        for _ in range(warmup):
            if B >= 64:
                self._kernel_basis.dispatch(thread_count=[B, D_in, 1], phi=buf_phi, input=buf_x, batch_size=B, in_features=D_in, num_grids=self.num_grids, grid_min=float(self.grid_min), grid_max=float(self.grid_max))
                self._kernel_gemm.dispatch(thread_count=[(D_out + 15) // 16 * 16, (B + 15) // 16 * 16, 1], C=buf_out, A=buf_phi, B_mat=self._buf_w_gemm, bias=self._buf_b, M=B, N=D_out, K=K_dim, has_bias=1 if self.use_bias else 0)
            else:
                self._kernel.dispatch(thread_count=[B, (D_out + 3) // 4, 1], output=buf_out, input=buf_x, weights=self._buf_w, bias=self._buf_b, batch_size=B, in_features=D_in, out_features=D_out, num_grids=self.num_grids, grid_min=float(self.grid_min), grid_max=float(self.grid_max))
        dev.wait_for_idle()

        import time
        t0 = time.perf_counter()
        for _ in range(iters):
            if B >= 64:
                self._kernel_basis.dispatch(thread_count=[B, D_in, 1], phi=buf_phi, input=buf_x, batch_size=B, in_features=D_in, num_grids=self.num_grids, grid_min=float(self.grid_min), grid_max=float(self.grid_max))
                self._kernel_gemm.dispatch(thread_count=[(D_out + 15) // 16 * 16, (B + 15) // 16 * 16, 1], C=buf_out, A=buf_phi, B_mat=self._buf_w_gemm, bias=self._buf_b, M=B, N=D_out, K=K_dim, has_bias=1 if self.use_bias else 0)
            else:
                self._kernel.dispatch(thread_count=[B, (D_out + 3) // 4, 1], output=buf_out, input=buf_x, weights=self._buf_w, bias=self._buf_b, batch_size=B, in_features=D_in, out_features=D_out, num_grids=self.num_grids, grid_min=float(self.grid_min), grid_max=float(self.grid_max))
        dev.wait_for_idle()
        t1 = time.perf_counter()
        return ((t1 - t0) / iters) * 1000.0

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> float:
        l1 = np.mean(np.abs(self.weights), axis=-1)
        reg_l1 = float(np.sum(l1))
        p = l1 / (reg_l1 + 1e-8)
        entropy = float(-np.sum(p * np.log(p + 1e-8)))
        return regularize_activation * reg_l1 + regularize_entropy * entropy


class ReLUKAN:
    """Multi-layer ReLUKAN Network."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        num_grids: int = 8,
        grid_range: tuple = (-1.0, 1.0),
        bias: bool = True,
        use_gpu: bool = True
    ):
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            ReLUKANLinear(
                in_f,
                out_f,
                num_grids=num_grids,
                grid_range=grid_range,
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
