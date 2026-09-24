"""
FastKAN: Kolmogorov-Arnold Network with Gaussian Radial Basis Functions (RBF).
Accelerated on GPU via Slang compute kernel with seamless CPU fallback.
"""

from typing import Optional, Sequence, Union, Any, List
import numpy as np
import slangpy
from slang_kans.layers.base import SlangKANLayerBase
from slang_kans.device import SlangTensor


class FastKANLinear(SlangKANLayerBase):
    """
    FastKAN Layer using Gaussian RBF basis functions: exp(-((x - c)/sigma)^2).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_centers: int = 8,
        grid_range: tuple = (-1.0, 1.0),
        bias: bool = True,
        use_gpu: bool = True
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.num_centers = num_centers
        self.grid_min, self.grid_max = grid_range
        self.use_gpu = use_gpu and self.mgr.is_gpu_available()

        self.centers = np.linspace(self.grid_min, self.grid_max, num_centers, dtype=np.float32)
        step = (self.grid_max - self.grid_min) / max(num_centers - 1, 1)
        self.inv_sigma = np.float32(1.0 / max(step * 1.5, 1e-6))

        scale = 1.0 / np.sqrt(in_features * num_centers)
        self.weights = np.random.uniform(-scale, scale, (in_features, out_features, num_centers)).astype(np.float32)

        self._kernel = None
        self._kernel_basis = None
        self._kernel_gemm = None
        self._buf_w = None
        self._buf_w_gemm = None
        self._buf_b = None
        self._saved_x = None
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias) if self.use_bias else None

    def _sync_gpu_weights(self):
        dev = self.mgr.device
        if dev is None:
            return
        if self._kernel is None:
            self._kernel = self.mgr.get_or_compile_kernel("fast_rbf", "fast_kan_forward")
            self._kernel_basis = self.mgr.get_or_compile_kernel("fast_rbf", "eval_rbf_basis")
            self._kernel_gemm = self.mgr.get_or_compile_kernel("gemm", "gemm_tiled")

        self._buf_w = dev.create_buffer(data=self.weights, usage=slangpy.BufferUsage.shader_resource)
        w_mat = np.ascontiguousarray(self.weights.transpose(1, 0, 2).reshape(self.out_features, -1))
        self._buf_w_gemm = dev.create_buffer(data=w_mat, usage=slangpy.BufferUsage.shader_resource)

        bias_arr = self.bias if (self.use_bias and self.bias is not None) else np.zeros(self.out_features, dtype=np.float32)
        self._buf_b = dev.create_buffer(data=bias_arr, usage=slangpy.BufferUsage.shader_resource)

    def forward(self, x: Any, return_tensor: bool = False) -> Any:
        buf_x = None
        if isinstance(x, SlangTensor):
            buf_x = x.buffer
            is_1d = len(x.shape) == 1
            B = 1 if is_1d else x.shape[0]
            D_in = x.shape[0] if is_1d else x.shape[1]
            x_arr = None
        else:
            x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
            is_1d = x_arr.ndim == 1
            if is_1d:
                x_arr = x_arr.reshape(1, -1)
            B, D_in = x_arr.shape

        self._saved_x = x_arr if x_arr is not None else (x.numpy().reshape(1, -1) if len(x.shape) == 1 else x.numpy())

        D_out = self.out_features

        if self.use_gpu and self.mgr.is_gpu_available():
            dev = self.mgr.device
            if self._buf_w is None or self._kernel is None:
                self._sync_gpu_weights()

            if buf_x is None:
                buf_x = self._get_buffer("x", B * D_in * 4, slangpy.BufferUsage.shader_resource)
                buf_x.copy_from_numpy(x_arr)

            buf_out = self._get_buffer("out", B * D_out * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)

            if B >= 64:
                K_dim = D_in * self.num_centers
                buf_phi = self._get_buffer("phi", B * K_dim * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                self._kernel_basis.dispatch(
                    thread_count=[B, D_in, 1],
                    phi=buf_phi,
                    input=buf_x,
                    batch_size=B,
                    in_features=D_in,
                    num_centers=self.num_centers,
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
                    num_centers=self.num_centers,
                    grid_min=float(self.grid_min),
                    grid_max=float(self.grid_max)
                )

            if return_tensor:
                return SlangTensor(buf_out, shape=(B, D_out) if not is_1d else (D_out,))

            dev.wait_for_idle()
            out = buf_out.to_numpy().view(np.float32)[:B * D_out].reshape(B, D_out)
            return out[0] if is_1d else out

        # CPU Fallback
        if x_arr is None and isinstance(x, SlangTensor):
            x_arr = x.numpy()
            if is_1d:
                x_arr = x_arr.reshape(1, -1)
        diff = (x_arr[..., None] - self.centers[None, None, :]) * self.inv_sigma
        rbf = np.exp(-diff * diff)
        out = np.einsum("bic,ijc->bj", rbf, self.weights)

        if self.use_bias and self.bias is not None:
            out += self.bias[None, :]

        return out[0] if is_1d else out

    def zero_grad(self) -> None:
        """Zeros parameter gradients."""
        if self.grad_weights is not None:
            self.grad_weights.fill(0.0)
        if self.grad_bias is not None:
            self.grad_bias.fill(0.0)

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Analytical backward pass for Gaussian RBF KAN.
        Accumulates dW and dBias in-place and returns dX gradient.
        """
        if self._saved_x is None:
            raise RuntimeError("Cannot call backward before forward()")
        x_arr = self._saved_x
        dY_arr = np.ascontiguousarray(np.asarray(dY, dtype=np.float32))
        is_1d = dY_arr.ndim == 1
        if is_1d:
            dY_arr = dY_arr.reshape(1, -1)

        B, D_in = x_arr.shape
        D_out = self.out_features

        diff = (x_arr[..., None] - self.centers[None, None, :]) * self.inv_sigma
        phi = np.exp(-diff * diff)

        dW = np.einsum("bic,bj->ijc", phi, dY_arr)
        if self.grad_weights is not None:
            self.grad_weights += dW

        if self.use_bias and self.grad_bias is not None:
            self.grad_bias += np.sum(dY_arr, axis=0)

        dphi_dx = (-2.0 * self.inv_sigma) * diff * phi
        dL_dphi = np.einsum("bj,ijc->bic", dY_arr, self.weights)
        dX = np.sum(dL_dphi * dphi_dx, axis=-1)

        return dX[0] if is_1d else dX

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
        K_dim = D_in * self.num_centers
        buf_phi = self._get_buffer("phi", B * K_dim * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)

        for _ in range(warmup):
            if B >= 64:
                self._kernel_basis.dispatch(thread_count=[B, D_in, 1], phi=buf_phi, input=buf_x, batch_size=B, in_features=D_in, num_centers=self.num_centers, grid_min=float(self.grid_min), grid_max=float(self.grid_max))
                self._kernel_gemm.dispatch(thread_count=[(D_out + 15) // 16 * 16, (B + 15) // 16 * 16, 1], C=buf_out, A=buf_phi, B_mat=self._buf_w_gemm, bias=self._buf_b, M=B, N=D_out, K=K_dim, has_bias=1 if self.use_bias else 0)
            else:
                self._kernel.dispatch(thread_count=[B, (D_out + 3) // 4, 1], output=buf_out, input=buf_x, weights=self._buf_w, bias=self._buf_b, batch_size=B, in_features=D_in, out_features=D_out, num_centers=self.num_centers, grid_min=float(self.grid_min), grid_max=float(self.grid_max))
        dev.wait_for_idle()

        import time
        t0 = time.perf_counter()
        for _ in range(iters):
            if B >= 64:
                self._kernel_basis.dispatch(thread_count=[B, D_in, 1], phi=buf_phi, input=buf_x, batch_size=B, in_features=D_in, num_centers=self.num_centers, grid_min=float(self.grid_min), grid_max=float(self.grid_max))
                self._kernel_gemm.dispatch(thread_count=[(D_out + 15) // 16 * 16, (B + 15) // 16 * 16, 1], C=buf_out, A=buf_phi, B_mat=self._buf_w_gemm, bias=self._buf_b, M=B, N=D_out, K=K_dim, has_bias=1 if self.use_bias else 0)
            else:
                self._kernel.dispatch(thread_count=[B, (D_out + 3) // 4, 1], output=buf_out, input=buf_x, weights=self._buf_w, bias=self._buf_b, batch_size=B, in_features=D_in, out_features=D_out, num_centers=self.num_centers, grid_min=float(self.grid_min), grid_max=float(self.grid_max))
        dev.wait_for_idle()
        t1 = time.perf_counter()
        return ((t1 - t0) / iters) * 1000.0

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> float:
        l1 = np.mean(np.abs(self.weights), axis=-1)
        reg_l1 = float(np.sum(l1))
        p = l1 / (reg_l1 + 1e-8)
        entropy = float(-np.sum(p * np.log(p + 1e-8)))
        return regularize_activation * reg_l1 + regularize_entropy * entropy


class FastKAN:
    """
    FastKAN wrapper: behaves as FastKANLinear when passed (in_f, out_f),
    or as multi-layer FastKAN network when passed layers_hidden: Sequence[int].
    """

    def __new__(cls, *args, **kwargs):
        if (len(args) >= 1 and isinstance(args[0], (list, tuple))) or "layers_hidden" in kwargs:
            instance = super().__new__(cls)
            instance.__init_net__(*args, **kwargs)
            return instance
        return FastKANLinear(*args, **kwargs)

    def __init_net__(
        self,
        layers_hidden: Sequence[int],
        num_centers: int = 8,
        grid_range: tuple = (-1.0, 1.0),
        bias: bool = True,
        use_gpu: bool = True
    ):
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            FastKANLinear(
                in_f,
                out_f,
                num_centers=num_centers,
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
