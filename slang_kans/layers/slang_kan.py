"""
SlangKAN: High-Throughput Multi-Layer Kolmogorov-Arnold Network for Slang.
Executes multi-layer networks entirely in GPU VRAM with zero host-device roundtrips
between intermediate layers.
"""

from typing import List, Sequence, Union, Optional, Any
import numpy as np

from slang_kans.device import SlangTensor, to_tensor, get_device
from slang_kans.layers.base import SlangKANLayerBase
from slang_kans.layers.cheby_kan import ChebyKANLinear
from slang_kans.layers.fast_kan import FastKANLinear
from slang_kans.layers.relu_kan import ReLUKANLinear
from slang_kans.layers.kan import KANLinear
from slang_kans.layers.wav_kan import WavKANLinear
from slang_kans.layers.fourier_kan import FourierKANLinear
from slang_kans.layers.jacobi_kan import JacobiKANLinear
from slang_kans.layers.rational_kan import RationalKANLinear


class SlangKAN:
    """
    Multi-layer Pure Slang KAN Network.
    Executes sequentially on GPU with zero-copy buffer chaining.
    """

    def __init__(
        self,
        layers: Sequence[Union[int, SlangKANLayerBase]],
        basis_type: str = "cheby",
        degree: int = 4,
        bias: bool = True,
        use_gpu: bool = True,
    ):
        if len(layers) > 0 and isinstance(layers[0], SlangKANLayerBase):
            self.layers = list(layers)
            self.layers_hidden = [l.in_features for l in self.layers] + [self.layers[-1].out_features]
            self.basis_type = "custom"
            self.degree = degree
            self.bias = bias
            self.use_gpu = use_gpu
            return

        self.layers_hidden = list(layers)
        self.basis_type = basis_type.lower()
        self.degree = degree
        self.bias = bias
        self.use_gpu = use_gpu
        self.layers: List[Any] = []

        for i in range(len(self.layers_hidden) - 1):
            in_f = self.layers_hidden[i]
            out_f = self.layers_hidden[i + 1]

            if self.basis_type == "cheby":
                layer = ChebyKANLinear(in_f, out_f, degree=degree, bias=bias, use_gpu=use_gpu)
            elif self.basis_type in ("fastkan", "rbf"):
                layer = FastKANLinear(in_f, out_f, num_centers=degree, bias=bias, use_gpu=use_gpu)
            elif self.basis_type in ("relukan", "relu"):
                layer = ReLUKANLinear(in_f, out_f, num_grids=degree, bias=bias, use_gpu=use_gpu)
            elif self.basis_type in ("bspline", "kan"):
                layer = KANLinear(in_f, out_f, grid_size=degree, bias=bias, use_gpu=use_gpu)
            elif self.basis_type in ("wav", "wavelet"):
                layer = WavKANLinear(in_f, out_f, num_wavelets=degree, bias=bias, use_gpu=use_gpu)
            elif self.basis_type == "fourier":
                layer = FourierKANLinear(in_f, out_f, num_frequencies=degree, bias=bias, use_gpu=use_gpu)
            elif self.basis_type == "jacobi":
                layer = JacobiKANLinear(in_f, out_f, degree=degree, bias=bias, use_gpu=use_gpu)
            elif self.basis_type == "rational":
                layer = RationalKANLinear(in_f, out_f, p_degree=degree, bias=bias, use_gpu=use_gpu)
            else:
                raise ValueError(f"Unknown basis_type: {basis_type}")

            self.layers.append(layer)

    def forward(self, x: Union[np.ndarray, SlangTensor], return_tensor: bool = False) -> Union[np.ndarray, SlangTensor]:
        """
        Executes sequential forward pass on GPU.
        Chains intermediate layers without transferring tensors back to CPU host.
        """
        curr = x
        num_layers = len(self.layers)

        for i, layer in enumerate(self.layers):
            is_last = (i == num_layers - 1)
            # Intermediate layers always output SlangTensor in GPU memory
            need_tensor = True if not is_last else return_tensor

            if hasattr(layer, "forward"):
                # Check if layer supports return_tensor
                try:
                    curr = layer.forward(curr, return_tensor=need_tensor)
                except TypeError:
                    curr = layer.forward(curr)
                    if not is_last and isinstance(curr, np.ndarray) and self.use_gpu:
                        curr = to_tensor(curr)
            else:
                curr = layer(curr)

        return curr

    __call__ = forward

    def state_dict(self) -> dict[str, np.ndarray]:
        """Returns model weights as a dictionary of NumPy arrays."""
        sd = {}
        for idx, layer in enumerate(self.layers):
            if hasattr(layer, "weights") and layer.weights is not None:
                sd[f"layer_{idx}.weights"] = np.array(layer.weights, copy=True)
            if hasattr(layer, "bias") and layer.bias is not None:
                sd[f"layer_{idx}.bias"] = np.array(layer.bias, copy=True)
            if hasattr(layer, "base_weights") and layer.base_weights is not None:
                sd[f"layer_{idx}.base_weights"] = np.array(layer.base_weights, copy=True)
            if hasattr(layer, "centers") and layer.centers is not None:
                sd[f"layer_{idx}.centers"] = np.array(layer.centers, copy=True)
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Loads weights from state dict into model layers."""
        for idx, layer in enumerate(self.layers):
            w_key = f"layer_{idx}.weights"
            if w_key in state_dict and hasattr(layer, "weights"):
                layer.weights = np.ascontiguousarray(state_dict[w_key], dtype=np.float32)
                if hasattr(layer, "_sync_gpu_weights"):
                    layer._sync_gpu_weights()
            b_key = f"layer_{idx}.bias"
            if b_key in state_dict and hasattr(layer, "bias"):
                layer.bias = np.ascontiguousarray(state_dict[b_key], dtype=np.float32)
                if hasattr(layer, "_sync_gpu_weights"):
                    layer._sync_gpu_weights()
            bw_key = f"layer_{idx}.base_weights"
            if bw_key in state_dict and hasattr(layer, "base_weights"):
                layer.base_weights = np.ascontiguousarray(state_dict[bw_key], dtype=np.float32)
                if hasattr(layer, "_sync_gpu_weights"):
                    layer._sync_gpu_weights()

    def save_weights(self, filepath: str):
        """
        Saves weights to disk. Supports .safetensors (if safetensors is installed) or .npz.
        """
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
        """Loads weights from .safetensors or .npz file."""
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

    def count_parameters(self) -> int:
        return sum(layer.count_parameters() for layer in self.layers)

    def __repr__(self) -> str:
        layers_str = " -> ".join(map(str, self.layers_hidden))
        return f"SlangKAN(layers=[{layers_str}], basis={self.basis_type!r})"
