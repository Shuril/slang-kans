"""
WavKAN: Wavelet Kolmogorov-Arnold Networks in Slang.
Uses continuous wavelets: Mexican Hat (Ricker), Morlet, and Derivative of Gaussian (DOG).
Accelerated on GPU via Slang compute kernel with CPU vectorized fallback.
"""

from typing import Optional, Sequence, Union, Any, List
import numpy as np
import slangpy
from slang_kans.layers.base import SlangKANLayerBase


class WavKANLinear(SlangKANLayerBase):
    """
    Wavelet KAN Linear Layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_wavelets: int = 8,
        wavelet_type: str = "mexican_hat",
        bias: bool = True,
        use_gpu: bool = True
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.num_wavelets = num_wavelets
        self.wavelet_type = wavelet_type
        self.use_gpu = use_gpu and self.mgr.is_gpu_available()

        # Wavelet type int code: 0=mexican_hat, 1=morlet, 2=dog
        type_map = {"mexican_hat": 0, "morlet": 1, "dog": 2}
        self.wavelet_code = type_map.get(wavelet_type.lower(), 0)

        # Translations and scales
        self.translations = np.linspace(-1.0, 1.0, num_wavelets, dtype=np.float32)
        self.translations = np.broadcast_to(self.translations[None, :], (in_features, num_wavelets)).copy()

        step = 2.0 / max(num_wavelets - 1, 1)
        scale_val = np.float32(step * 1.5)
        self.scales = np.full((in_features, num_wavelets), scale_val, dtype=np.float32)

        scale = 1.0 / np.sqrt(in_features * num_wavelets)
        self.weights = np.random.uniform(-scale, scale, (in_features, out_features, num_wavelets)).astype(np.float32)

        # GPU cached buffers and kernel
        self._kernel = None
        self._kernel_basis = None
        self._kernel_gemm = None
        self._buf_w = None
        self._buf_w_gemm = None
        self._buf_t = None
        self._buf_s = None
        self._buf_b = None

    def _sync_gpu_weights(self):
        dev = self.mgr.device
        if dev is None:
            return
        if self._kernel is None:
            self._kernel = self.mgr.get_or_compile_kernel("wavelet", "wavelet_kan_forward")
            self._kernel_basis = self.mgr.get_or_compile_kernel("wavelet", "eval_wavelet_basis")
            self._kernel_gemm = self.mgr.get_or_compile_kernel("gemm", "gemm_tiled")

        self._buf_w = dev.create_buffer(data=self.weights, usage=slangpy.BufferUsage.shader_resource)
        w_mat = np.ascontiguousarray(self.weights.transpose(1, 0, 2).reshape(self.out_features, -1))
        self._buf_w_gemm = dev.create_buffer(data=w_mat, usage=slangpy.BufferUsage.shader_resource)

        self._buf_t = dev.create_buffer(data=self.translations, usage=slangpy.BufferUsage.shader_resource)
        self._buf_s = dev.create_buffer(data=self.scales, usage=slangpy.BufferUsage.shader_resource)
        bias_arr = self.bias if (self.use_bias and self.bias is not None) else np.zeros(self.out_features, dtype=np.float32)
        self._buf_b = dev.create_buffer(data=bias_arr, usage=slangpy.BufferUsage.shader_resource)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward WavKAN evaluation."""
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
                K_dim = D_in * self.num_wavelets
                buf_phi = self._get_buffer("phi", B * K_dim * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                self._kernel_basis.dispatch(
                    thread_count=[B, D_in, 1],
                    phi=buf_phi,
                    input=buf_x,
                    translations=self._buf_t,
                    scales=self._buf_s,
                    batch_size=B,
                    in_features=D_in,
                    num_wavelets=self.num_wavelets,
                    wavelet_type=self.wavelet_code
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
                    translations=self._buf_t,
                    scales=self._buf_s,
                    bias=self._buf_b,
                    batch_size=B,
                    in_features=D_in,
                    out_features=D_out,
                    num_wavelets=self.num_wavelets,
                    wavelet_type=self.wavelet_code
                )

            dev.wait_for_idle()
            out = buf_out.to_numpy().view(np.float32)[:B * D_out].reshape(B, D_out)
            return out[0] if is_1d else out

        # CPU Fallback
        z = (x_arr[..., None] - self.translations[None, ...]) / (np.abs(self.scales[None, ...]) + 1e-4)
        z2 = z * z
        gauss = np.exp(-0.5 * z2)

        if self.wavelet_code == 1:
            phi = np.cos(5.0 * z) * gauss
        elif self.wavelet_code == 2:
            phi = -z * gauss
        else:
            phi = (1.0 - z2) * gauss

        out = np.einsum("biw,ijw->bj", phi, self.weights)

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
        K_dim = D_in * self.num_wavelets
        buf_phi = self._get_buffer("phi", B * K_dim * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)

        for _ in range(warmup):
            if B >= 64:
                self._kernel_basis.dispatch(thread_count=[B, D_in, 1], phi=buf_phi, input=buf_x, translations=self._buf_t, scales=self._buf_s, batch_size=B, in_features=D_in, num_wavelets=self.num_wavelets, wavelet_type=self.wavelet_code)
                self._kernel_gemm.dispatch(thread_count=[(D_out + 15) // 16 * 16, (B + 15) // 16 * 16, 1], C=buf_out, A=buf_phi, B_mat=self._buf_w_gemm, bias=self._buf_b, M=B, N=D_out, K=K_dim, has_bias=1 if self.use_bias else 0)
            else:
                self._kernel.dispatch(thread_count=[B, (D_out + 3) // 4, 1], output=buf_out, input=buf_x, weights=self._buf_w, translations=self._buf_t, scales=self._buf_s, bias=self._buf_b, batch_size=B, in_features=D_in, out_features=D_out, num_wavelets=self.num_wavelets, wavelet_type=self.wavelet_code)
        dev.wait_for_idle()

        import time
        t0 = time.perf_counter()
        for _ in range(iters):
            if B >= 64:
                self._kernel_basis.dispatch(thread_count=[B, D_in, 1], phi=buf_phi, input=buf_x, translations=self._buf_t, scales=self._buf_s, batch_size=B, in_features=D_in, num_wavelets=self.num_wavelets, wavelet_type=self.wavelet_code)
                self._kernel_gemm.dispatch(thread_count=[(D_out + 15) // 16 * 16, (B + 15) // 16 * 16, 1], C=buf_out, A=buf_phi, B_mat=self._buf_w_gemm, bias=self._buf_b, M=B, N=D_out, K=K_dim, has_bias=1 if self.use_bias else 0)
            else:
                self._kernel.dispatch(thread_count=[B, (D_out + 3) // 4, 1], output=buf_out, input=buf_x, weights=self._buf_w, translations=self._buf_t, scales=self._buf_s, bias=self._buf_b, batch_size=B, in_features=D_in, out_features=D_out, num_wavelets=self.num_wavelets, wavelet_type=self.wavelet_code)
        dev.wait_for_idle()
        t1 = time.perf_counter()
        return ((t1 - t0) / iters) * 1000.0

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> float:
        l1 = np.mean(np.abs(self.weights), axis=-1)
        reg_l1 = float(np.sum(l1))
        p = l1 / (reg_l1 + 1e-8)
        entropy = float(-np.sum(p * np.log(p + 1e-8)))
        return regularize_activation * reg_l1 + regularize_entropy * entropy


class WavKAN:
    """Multi-layer WavKAN Network."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        num_wavelets: int = 8,
        wavelet_type: str = "mexican_hat",
        bias: bool = True,
        use_gpu: bool = True
    ):
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            WavKANLinear(
                in_f,
                out_f,
                num_wavelets=num_wavelets,
                wavelet_type=wavelet_type,
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
