"""
JacobiKAN: Kolmogorov-Arnold Networks with Orthogonal Jacobi Polynomials in Slang.
Parameterized by (alpha, beta). Special cases include Legendre (0, 0) and Chebyshev (-0.5, -0.5).
Accelerated on GPU via Slang compute kernel with CPU vectorized fallback.
"""

from typing import Optional, Sequence, Union, Any, List
import numpy as np
import slangpy
from slang_kans.layers.base import SlangKANLayerBase
from slang_kans.device import SlangTensor


class JacobiKANLinear(SlangKANLayerBase):
    """
    JacobiKAN Linear Layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int = 4,
        alpha: float = 0.0,
        beta: float = 0.0,
        bias: bool = True,
        use_gpu: bool = True
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.degree = degree
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.use_gpu = use_gpu and self.mgr.is_gpu_available()

        scale = 1.0 / np.sqrt(in_features * degree)
        self.weights = np.random.uniform(-scale, scale, (in_features, out_features, degree)).astype(np.float32)

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
            self._kernel = self.mgr.get_or_compile_kernel("jacobi", "jacobi_kan_forward")
            self._kernel_basis = self.mgr.get_or_compile_kernel("jacobi", "eval_jacobi_basis")
            self._kernel_gemm = self.mgr.get_or_compile_kernel("gemm", "gemm_tiled")

        self._buf_w = dev.create_buffer(data=self.weights, usage=slangpy.BufferUsage.shader_resource)
        w_mat = np.ascontiguousarray(self.weights.transpose(1, 0, 2).reshape(self.out_features, -1))
        self._buf_w_gemm = dev.create_buffer(data=w_mat, usage=slangpy.BufferUsage.shader_resource)

        bias_arr = self.bias if (self.use_bias and self.bias is not None) else np.zeros(self.out_features, dtype=np.float32)
        self._buf_b = dev.create_buffer(data=bias_arr, usage=slangpy.BufferUsage.shader_resource)

    def forward(self, x: Any, return_tensor: bool = False) -> Any:
        """Forward JacobiKAN evaluation."""
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

        D_out = self.out_features
        deg = self.degree

        if self.use_gpu and self.mgr.is_gpu_available():
            dev = self.mgr.device
            if self._buf_w is None or self._kernel is None:
                self._sync_gpu_weights()

            if buf_x is None:
                buf_x = self._get_buffer("x", B * D_in * 4, slangpy.BufferUsage.shader_resource)
                buf_x.copy_from_numpy(x_arr)

            buf_out = self._get_buffer("out", B * D_out * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)

            if B >= 64:
                K_dim = D_in * deg
                buf_phi = self._get_buffer("phi", B * K_dim * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                self._kernel_basis.dispatch(
                    thread_count=[B, D_in, 1],
                    phi=buf_phi,
                    input=buf_x,
                    batch_size=B,
                    in_features=D_in,
                    degree=deg,
                    alpha=self.alpha,
                    beta=self.beta
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
                    degree=deg,
                    alpha=self.alpha,
                    beta=self.beta
                )

            if return_tensor:
                return SlangTensor(buf_out, shape=(B, D_out) if not is_1d else (D_out,))

            dev.wait_for_idle()
            out = buf_out.to_numpy().view(np.float32)[:B * D_out].reshape(B, D_out)
            return out[0] if is_1d else out

        # CPU Fallback: Recurrence relation for Jacobi polynomials
        if x_arr is None and isinstance(x, SlangTensor):
            x_arr = x.numpy()
            if is_1d:
                x_arr = x_arr.reshape(1, -1)
        xc = np.clip(x_arr, -1.0, 1.0)
        bases = [np.ones_like(xc[..., None])]
        if deg > 1:
            p1 = 0.5 * (self.alpha - self.beta + (self.alpha + self.beta + 2.0) * xc[..., None])
            bases.append(p1)
            a_b = self.alpha + self.beta
            for n in range(2, deg):
                an = 2.0 * n * (n + a_b) * (2.0 * n + a_b - 2.0)
                bn_term1 = (2.0 * n + a_b - 1.0) * (2.0 * n + a_b) * (2.0 * n + a_b - 2.0)
                bn_term2 = (2.0 * n + a_b - 1.0) * (self.alpha**2 - self.beta**2)
                bn = bn_term1 * xc[..., None] + bn_term2
                cn = 2.0 * (n + self.alpha - 1.0) * (n + self.beta - 1.0) * (2.0 * n + a_b)
                pn = (bn * bases[-1] - cn * bases[-2]) / an
                bases.append(pn)

        jacobi_bases = np.concatenate(bases, axis=-1)  # [B, D_in, degree]
        out = np.einsum("bid,ijd->bj", jacobi_bases, self.weights)

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
        deg = self.degree

        if self._buf_w is None or self._kernel is None:
            self._sync_gpu_weights()

        buf_x = self._get_buffer("x", B * D_in * 4, slangpy.BufferUsage.shader_resource)
        buf_x.copy_from_numpy(x_arr)
        buf_out = self._get_buffer("out", B * D_out * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)

        use_gemm = (B >= 64)
        if use_gemm:
            K_dim = D_in * deg
            buf_phi = self._get_buffer("phi", B * K_dim * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)

        for _ in range(warmup):
            if use_gemm:
                self._kernel_basis.dispatch(
                    thread_count=[B, D_in, 1],
                    phi=buf_phi,
                    input=buf_x,
                    batch_size=B,
                    in_features=D_in,
                    degree=deg,
                    alpha=self.alpha,
                    beta=self.beta
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
                    degree=deg,
                    alpha=self.alpha,
                    beta=self.beta
                )
        dev.wait_for_idle()

        import time
        start = time.perf_counter()
        for _ in range(iters):
            if use_gemm:
                self._kernel_basis.dispatch(
                    thread_count=[B, D_in, 1],
                    phi=buf_phi,
                    input=buf_x,
                    batch_size=B,
                    in_features=D_in,
                    degree=deg,
                    alpha=self.alpha,
                    beta=self.beta
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
                    degree=deg,
                    alpha=self.alpha,
                    beta=self.beta
                )
        dev.wait_for_idle()
        end = time.perf_counter()
        return (end - start) / iters * 1000.0

    def benchmark(self, x: np.ndarray, warmup: int = 10, iters: int = 50) -> float:
        if self.use_gpu and self.mgr.is_gpu_available():
            return self._benchmark_gpu(x, warmup, iters)
        import time
        for _ in range(warmup):
            self.forward(x)
        start = time.perf_counter()
        for _ in range(iters):
            self.forward(x)
        end = time.perf_counter()
        return (end - start) / iters * 1000.0

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> float:
        l1 = np.mean(np.abs(self.weights), axis=-1)
        reg_l1 = float(np.sum(l1))
        p = l1 / (reg_l1 + 1e-8)
        entropy = float(-np.sum(p * np.log(p + 1e-8)))
        return regularize_activation * reg_l1 + regularize_entropy * entropy


class JacobiKAN:
    """Multi-layer JacobiKAN Network."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        degree: int = 4,
        alpha: float = 0.0,
        beta: float = 0.0,
        bias: bool = True,
        use_gpu: bool = True
    ):
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            JacobiKANLinear(
                in_f,
                out_f,
                degree=degree,
                alpha=alpha,
                beta=beta,
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
