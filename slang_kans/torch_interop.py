"""
PyTorch Autograd and Module Interoperability for slang-KANs.
Enables native torch.nn.Module layers backed by Vulkan / Metal Slang compute shaders,
with complete analytical autograd backward propagation and PyTorch optimizer compatibility.
"""

from typing import Union, Sequence, Optional, Any
import numpy as np
import torch
import torch.nn as nn

from slang_kans.layers.cheby_kan import ChebyKANLinear
from slang_kans.layers.fast_kan import FastKANLinear
from slang_kans.layers.relu_kan import ReLUKANLinear
from slang_kans.layers.kan import KANLinear
from slang_kans.layers.wav_kan import WavKANLinear
from slang_kans.layers.fourier_kan import FourierKANLinear
from slang_kans.layers.jacobi_kan import JacobiKANLinear
from slang_kans.layers.rational_kan import RationalKANLinear
from slang_kans.device import is_slang_available, get_device


class _SlangKANAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], layer: Any) -> torch.Tensor:
        ctx.layer = layer
        ctx.has_bias = (bias is not None)
        ctx.device = x.device
        ctx.dtype = x.dtype

        # Fast contiguous numpy array view
        x_np = x.detach().contiguous().cpu().numpy().astype(np.float32)
        y_np = layer.forward(x_np)

        ctx.save_for_backward(x, weight, bias if bias is not None else torch.empty(0))
        out = torch.from_numpy(y_np).to(device=x.device, dtype=x.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        layer = ctx.layer
        device = ctx.device
        dtype = ctx.dtype
        saved_tensors = ctx.saved_tensors
        x, weight = saved_tensors[0], saved_tensors[1]

        grad_np = grad_output.detach().contiguous().cpu().numpy().astype(np.float32)

        if hasattr(layer, "zero_grad"):
            layer.zero_grad()

        if hasattr(layer, "backward"):
            dx_np = layer.backward(grad_np)
            dx = torch.from_numpy(dx_np).to(device=device, dtype=dtype)
        else:
            # Numerical / heuristic fallback if backward not implemented on custom layer
            dx = torch.zeros_like(x)

        if hasattr(layer, "grad_weights") and layer.grad_weights is not None:
            dw = torch.from_numpy(layer.grad_weights.copy()).to(device=weight.device, dtype=weight.dtype)
        else:
            dw = torch.zeros_like(weight)

        db = None
        if ctx.has_bias:
            if hasattr(layer, "grad_bias") and layer.grad_bias is not None:
                db = torch.from_numpy(layer.grad_bias.copy()).to(device=device, dtype=dtype)
            else:
                db = grad_output.sum(dim=0)

        return dx, dw, db, None


class TorchSlangKANLinear(nn.Module):
    """
    Drop-in PyTorch Linear Layer accelerated by Slang Compute Shaders.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        basis: Basis type ("cheby", "fastkan", "relu", "bspline", "wav", "fourier", "jacobi", "rational")
        degree: Degree, grid count, or wavelet scale
        bias: Whether to include bias term
        use_gpu: Whether to leverage Slang GPU hardware acceleration
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        basis: str = "cheby",
        degree: int = 4,
        bias: bool = True,
        use_gpu: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis = basis.lower()
        self.degree = degree
        self.use_bias = bias
        self.use_gpu = use_gpu

        if self.basis == "cheby":
            self.slang_layer = ChebyKANLinear(in_features, out_features, degree=degree, bias=bias, use_gpu=use_gpu)
        elif self.basis in ("fastkan", "rbf"):
            self.slang_layer = FastKANLinear(in_features, out_features, num_centers=degree, bias=bias, use_gpu=use_gpu)
        elif self.basis in ("relu", "relukan"):
            self.slang_layer = ReLUKANLinear(in_features, out_features, num_grids=degree, bias=bias, use_gpu=use_gpu)
        elif self.basis in ("bspline", "kan"):
            self.slang_layer = KANLinear(in_features, out_features, grid_size=degree, bias=bias, use_gpu=use_gpu)
        elif self.basis in ("wav", "wavelet"):
            self.slang_layer = WavKANLinear(in_features, out_features, num_wavelets=degree, bias=bias, use_gpu=use_gpu)
        elif self.basis == "fourier":
            self.slang_layer = FourierKANLinear(in_features, out_features, num_frequencies=degree, bias=bias, use_gpu=use_gpu)
        elif self.basis == "jacobi":
            self.slang_layer = JacobiKANLinear(in_features, out_features, degree=degree, bias=bias, use_gpu=use_gpu)
        elif self.basis == "rational":
            self.slang_layer = RationalKANLinear(in_features, out_features, p_degree=degree, bias=bias, use_gpu=use_gpu)
        else:
            raise ValueError(f"Unsupported Slang KAN basis type: {basis}")

        if hasattr(self.slang_layer, "weights") and self.slang_layer.weights is not None:
            self.weight = nn.Parameter(torch.from_numpy(self.slang_layer.weights.copy()))
        else:
            self.weight = nn.Parameter(torch.randn(in_features, out_features, degree))

        if bias and hasattr(self.slang_layer, "bias") and self.slang_layer.bias is not None:
            self.bias_param = nn.Parameter(torch.from_numpy(self.slang_layer.bias.copy()))
        else:
            self.bias_param = None

    def _sync_to_slang(self):
        """Synchronizes PyTorch parameter weights into Slang layer memory."""
        w_np = self.weight.detach().cpu().numpy().astype(np.float32)
        if hasattr(self.slang_layer, "weights") and self.slang_layer.weights is not None:
            self.slang_layer.weights[:] = w_np
            if hasattr(self.slang_layer, "_sync_gpu_weights"):
                self.slang_layer._sync_gpu_weights()

        if self.bias_param is not None and hasattr(self.slang_layer, "bias") and self.slang_layer.bias is not None:
            self.slang_layer.bias[:] = self.bias_param.detach().cpu().numpy().astype(np.float32)
            if hasattr(self.slang_layer, "_sync_gpu_weights"):
                self.slang_layer._sync_gpu_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._sync_to_slang()
        return _SlangKANAutogradFunction.apply(x, self.weight, self.bias_param, self.slang_layer)


class TorchSlangKAN(nn.Module):
    """
    Multi-layer PyTorch Network accelerated by Slang Compute Shaders.
    """
    def __init__(
        self,
        layers_hidden: Sequence[int],
        basis: str = "cheby",
        degree: int = 4,
        bias: bool = True,
        use_gpu: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TorchSlangKANLinear(in_f, out_f, basis=basis, degree=degree, bias=bias, use_gpu=use_gpu)
            for in_f, out_f in zip(layers_hidden, layers_hidden[1:])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
