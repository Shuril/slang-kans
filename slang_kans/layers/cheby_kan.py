"""
ChebyKAN: Orthogonal Chebyshev Polynomial KAN Layer for Slang.
Evaluates y_j = bias_j + sum_{i} Clenshaw(x_i, w_{i, j, :}).
Accelerated on GPU via Slang compute kernel with seamless CPU fallback.
"""

from typing import Optional, Sequence, Union, Any, List
import numpy as np
import slangpy
from slang_kans.device import SlangTensor
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

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros(out_features, dtype=np.float32) if self.use_bias else None
        self._saved_x = None

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
            self._kernel = self.mgr.get_or_compile_kernel("cheby", "cheby_kan_forward")
            self._kernel_basis = self.mgr.get_or_compile_kernel("cheby", "eval_cheby_basis")
            self._kernel_gemm = self.mgr.get_or_compile_kernel("gemm", "gemm_tiled")

        self._buf_w = dev.create_buffer(data=self.weights, usage=slangpy.BufferUsage.shader_resource)
        w_mat = np.ascontiguousarray(self.weights.transpose(1, 0, 2).reshape(self.out_features, -1))
        self._buf_w_gemm = dev.create_buffer(data=w_mat, usage=slangpy.BufferUsage.shader_resource)

        bias_arr = self.bias if (self.use_bias and self.bias is not None) else np.zeros(self.out_features, dtype=np.float32)
        self._buf_b = dev.create_buffer(data=bias_arr, usage=slangpy.BufferUsage.shader_resource)

    def forward(self, x: Union[np.ndarray, SlangTensor], return_tensor: bool = False) -> Union[np.ndarray, SlangTensor]:
        if isinstance(x, SlangTensor):
            buf_x = x.buffer
            is_1d = len(x.shape) == 1
            B, D_in = (1, x.shape[0]) if is_1d else x.shape
            x_arr = None
            self._saved_x = None
        else:
            x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
            is_1d = x_arr.ndim == 1
            if is_1d:
                x_arr = x_arr.reshape(1, -1)
            self._saved_x = x_arr
            B, D_in = x_arr.shape
            buf_x = None

        D_out = self.out_features
        deg = self.degree
        num_bases = deg + 1

        if self.use_gpu and self.mgr.is_gpu_available():
            dev = self.mgr.device
            if self._buf_w is None or self._kernel is None:
                self._sync_gpu_weights()

            if buf_x is None:
                buf_x = self._get_buffer("x", B * D_in * 4, slangpy.BufferUsage.shader_resource)
                buf_x.copy_from_numpy(x_arr)

            buf_out = self._get_buffer("out", B * D_out * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)

            if B >= 64:
                K_dim = D_in * num_bases
                buf_phi = self._get_buffer("phi", B * K_dim * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)
                self._kernel_basis.dispatch(
                    thread_count=[B, D_in, 1],
                    phi=buf_phi,
                    input=buf_x,
                    batch_size=B,
                    in_features=D_in,
                    num_bases=num_bases
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
                    degree=deg
                )

            if return_tensor:
                return SlangTensor(buf_out, shape=(B, D_out) if not is_1d else (D_out,))

            dev.wait_for_idle()
            out = buf_out.to_numpy().view(np.float32)[:B * D_out].reshape(B, D_out)
            return out[0] if is_1d else out

        # CPU Fallback: Clenshaw recurrence across batch and features
        if x_arr is None and isinstance(x, SlangTensor):
            x_arr = x.numpy()
            if is_1d:
                x_arr = x_arr.reshape(1, -1)
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

    def zero_grad(self) -> None:
        if self.grad_weights is not None:
            self.grad_weights.fill(0.0)
        if self.grad_bias is not None:
            self.grad_bias.fill(0.0)

    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Analytical backward pass for Chebyshev polynomial series.
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
        deg = self.degree
        num_bases = deg + 1

        s = np.clip(x_arr, -1.0, 1.0)
        T = [np.ones_like(s), s]
        dT = [np.zeros_like(s), np.ones_like(s)]
        for k in range(2, num_bases):
            T_curr = 2.0 * s * T[-1] - T[-2]
            dT_curr = 2.0 * T[-1] + 2.0 * s * dT[-1] - dT[-2]
            T.append(T_curr)
            dT.append(dT_curr)

        Phi = np.stack(T, axis=-1)
        dPhi_dx = np.stack(dT, axis=-1)

        dW = np.einsum("bik,bj->ijk", Phi, dY_arr)
        if self.grad_weights is not None:
            self.grad_weights += dW

        if self.use_bias and self.grad_bias is not None:
            self.grad_bias += np.sum(dY_arr, axis=0)

        dPhi = np.einsum("bj,ijk->bik", dY_arr, self.weights)
        dX = np.sum(dPhi * dPhi_dx, axis=-1) * (np.abs(x_arr) <= 1.0).astype(np.float32)

        return dX[0] if is_1d else dX

    def _benchmark_gpu(self, x: np.ndarray, warmup: int = 10, iters: int = 50) -> float:
        dev = self.mgr.device
        x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        B, D_in = x_arr.shape
        D_out = self.out_features
        deg = self.degree
        num_bases = deg + 1

        if self._buf_w is None or self._kernel is None:
            self._sync_gpu_weights()

        buf_x = self._get_buffer("x", B * D_in * 4, slangpy.BufferUsage.shader_resource)
        buf_x.copy_from_numpy(x_arr)
        buf_out = self._get_buffer("out", B * D_out * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)

        use_gemm = (B >= 64)
        if use_gemm:
            K_dim = D_in * num_bases
            buf_phi = self._get_buffer("phi", B * K_dim * 4, slangpy.BufferUsage.shader_resource | slangpy.BufferUsage.unordered_access)

        for _ in range(warmup):
            if use_gemm:
                self._kernel_basis.dispatch(
                    thread_count=[B, D_in, 1],
                    phi=buf_phi,
                    input=buf_x,
                    batch_size=B,
                    in_features=D_in,
                    num_bases=num_bases
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
                    degree=deg
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
                    num_bases=num_bases
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
                    degree=deg
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

    def state_dict(self) -> dict[str, np.ndarray]:
        """Returns model parameters as dictionary of NumPy arrays."""
        sd = {}
        for idx, layer in enumerate(self.layers):
            for k, v in layer.state_dict().items():
                sd[f"layer_{idx}.{k}"] = v
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Loads model parameters from state dict."""
        for idx, layer in enumerate(self.layers):
            layer_sd = {}
            prefix = f"layer_{idx}."
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    layer_sd[k[len(prefix):]] = v
            if layer_sd:
                layer.load_state_dict(layer_sd)

    def save_weights(self, filepath: str):
        """Saves model weights to .safetensors (if available) or .npz."""
        sd = self.state_dict()
        if filepath.endswith(".safetensors"):
            try:
                import safetensors.numpy as st
                st.save_file(sd, filepath)
                return
            except ImportError:
                filepath = filepath[:-12] + ".npz"
        np.savez(filepath, **sd)

    def load_weights(self, filepath: str):
        """Loads model weights from .safetensors or .npz."""
        if filepath.endswith(".safetensors"):
            try:
                import safetensors.numpy as st
                sd = st.load_file(filepath)
                self.load_state_dict(sd)
                return
            except ImportError:
                filepath = filepath[:-12] + ".npz"
        with np.load(filepath) as data:
            sd = {k: data[k] for k in data.files}
            self.load_state_dict(sd)
