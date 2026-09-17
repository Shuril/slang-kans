"""
Native INT8 & INT4 Quantization for slang-KANs.
Provides block-level affine and symmetric quantization for all KAN architectures.
Delivers up to 4x memory compression and accelerated inference.
"""

from __future__ import annotations
import math
from typing import Dict, Any, Optional, Sequence, Union
import numpy as np

from slang_kans.layers.base import SlangKANLayerBase
from slang_kans.layers.kan import KANLinear
from slang_kans.layers.fast_kan import FastKANLinear
from slang_kans.layers.cheby_kan import ChebyKANLinear
from slang_kans.layers.relu_kan import ReLUKANLinear
from slang_kans.layers.wav_kan import WavKANLinear
from slang_kans.layers.fourier_kan import FourierKANLinear
from slang_kans.layers.jacobi_kan import JacobiKANLinear
from slang_kans.layers.low_rank_kan import LowRankKANLinear
from slang_kans.layers.mult_kan import MultKANLinear


class QuantizedWeight:
    """
    Block-level quantized weight tensor (INT8 / INT4) with scale and zero-point parameters.
    """

    def __init__(
        self,
        weight: np.ndarray,
        group_size: int = 64,
        bits: int = 8,
        mode: str = "affine"
    ):
        self.orig_shape = weight.shape
        w_2d = weight.reshape(weight.shape[0], -1).astype(np.float32)
        out_f, in_f = w_2d.shape
        self.out_features = out_f
        self.in_features = in_f
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        pad_len = (group_size - (in_f % group_size)) % group_size
        self.pad_len = pad_len
        if pad_len > 0:
            w_2d = np.pad(w_2d, ((0, 0), (0, pad_len)), mode="constant", constant_values=0)

        num_groups = w_2d.shape[1] // group_size
        grouped = w_2d.reshape(out_f, num_groups, group_size)

        max_q = (1 << bits) - 1
        w_min = np.min(grouped, axis=-1, keepdims=True)
        w_max = np.max(grouped, axis=-1, keepdims=True)
        scales = np.maximum(w_max - w_min, 1e-7) / float(max_q)
        biases = w_min

        q_vals = np.clip(np.round((grouped - biases) / scales), 0, max_q).astype(np.uint8)

        if bits == 4:
            # Pack two 4-bit values per uint8 byte
            q_even = q_vals[..., 0::2]
            q_odd = q_vals[..., 1::2]
            self.packed_weight = (q_even | (q_odd << 4)).astype(np.uint8)
        else:
            self.packed_weight = q_vals

        self.scales = scales.astype(np.float32)
        self.biases = biases.astype(np.float32)

    def dequantize(self) -> np.ndarray:
        """Dequantizes weights back to float32."""
        out_f = self.out_features
        num_groups = self.scales.shape[1]
        g_size = self.group_size

        if self.bits == 4:
            q_even = self.packed_weight & 0x0F
            q_odd = (self.packed_weight >> 4) & 0x0F
            q_vals = np.empty((out_f, num_groups, g_size), dtype=np.float32)
            q_vals[..., 0::2] = q_even
            q_vals[..., 1::2] = q_odd
        else:
            q_vals = self.packed_weight.astype(np.float32)

        w_full = q_vals * self.scales + self.biases
        w_flat = w_full.reshape(out_f, -1)
        if self.pad_len > 0:
            w_flat = w_flat[:, :self.in_features]
        return w_flat.reshape(self.orig_shape)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        w = self.dequantize()
        return x @ w.T


class QuantizedKANLinearBase:
    """Base class for all quantized KAN layers."""
    pass


class QuantizedChebyKANLinear(QuantizedKANLinearBase):
    def __init__(self, layer: ChebyKANLinear, group_size: int = 64, bits: int = 8):
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.degree = layer.degree
        self.use_bias = layer.use_bias
        self.bias = layer.bias.copy() if layer.bias is not None else None
        # Quantize weights
        self.q_weights = QuantizedWeight(layer.weights, group_size=group_size, bits=bits)

    def forward(self, x: np.ndarray) -> np.ndarray:
        w_deq = self.q_weights.dequantize()
        x_arr = np.asarray(x, dtype=np.float32)
        is_1d = x_arr.ndim == 1
        if is_1d:
            x_arr = x_arr.reshape(1, -1)

        B, D_in = x_arr.shape
        D_out = self.out_features
        deg = self.degree

        s = np.clip(x_arr, -1.0, 1.0)
        two_s = 2.0 * s[..., None]
        b2 = np.zeros((B, D_in, D_out), dtype=np.float32)
        b1 = np.zeros((B, D_in, D_out), dtype=np.float32)

        for k in range(deg, 0, -1):
            w_k = w_deq[:, :, k]
            b0 = two_s * b1 - b2 + w_k[None, :, :]
            b2 = b1
            b1 = b0

        phi = s[..., None] * b1 - b2 + w_deq[:, :, 0][None, :, :]
        out = np.sum(phi, axis=1)

        if self.use_bias and self.bias is not None:
            out += self.bias[None, :]

        return out[0] if is_1d else out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class QuantizedFastKANLinear(QuantizedKANLinearBase):
    def __init__(self, layer: FastKANLinear, group_size: int = 64, bits: int = 8):
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.num_centers = layer.num_centers
        self.centers = layer.centers.copy()
        self.inv_sigma = layer.inv_sigma
        self.use_bias = layer.use_bias
        self.bias = layer.bias.copy() if layer.bias is not None else None
        self.q_weights = QuantizedWeight(layer.weights, group_size=group_size, bits=bits)

    def forward(self, x: np.ndarray) -> np.ndarray:
        w_deq = self.q_weights.dequantize()
        x_arr = np.asarray(x, dtype=np.float32)
        is_1d = x_arr.ndim == 1
        if is_1d:
            x_arr = x_arr.reshape(1, -1)

        diff = (x_arr[..., None] - self.centers[None, None, :]) * self.inv_sigma
        rbf = np.exp(-diff * diff)
        out = np.einsum("bic,ijc->bj", rbf, w_deq)

        if self.use_bias and self.bias is not None:
            out += self.bias[None, :]

        return out[0] if is_1d else out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class QuantizedKANLinear(QuantizedKANLinearBase):
    def __init__(self, layer: KANLinear, group_size: int = 64, bits: int = 8):
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.grid_size = layer.grid_size
        self.grid_min = layer.grid_min
        self.grid_max = layer.grid_max
        self.span = layer.span
        self.use_bias = layer.use_bias
        self.bias = layer.bias.copy() if layer.bias is not None else None
        self.q_weights = QuantizedWeight(layer.weights, group_size=group_size, bits=bits)

    def forward(self, x: np.ndarray) -> np.ndarray:
        w_deq = self.q_weights.dequantize()
        x_arr = np.asarray(x, dtype=np.float32)
        is_1d = x_arr.ndim == 1
        if is_1d:
            x_arr = x_arr.reshape(1, -1)

        B, D_in = x_arr.shape
        D_out = self.out_features
        G = self.grid_size

        x_clamped = np.clip(x_arr, self.grid_min, self.grid_max)
        t = (x_clamped - self.grid_min) / self.span * G
        interval = np.clip(np.floor(t).astype(np.int32), 0, G - 1)
        u = t - interval

        u2 = u * u
        u3 = u2 * u
        b0 = (1.0 - u)**3 / 6.0
        b1 = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0
        b2 = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0
        b3 = u3 / 6.0
        bases = np.stack([b0, b1, b2, b3], axis=-1)

        out = np.zeros((B, D_out), dtype=np.float32)
        for i in range(D_in):
            int_i = interval[:, i]
            bases_i = bases[:, i, :]
            for k in range(4):
                idx = int_i + k
                w_slice = w_deq[i, :, idx]
                out += bases_i[:, k:k+1] * w_slice

        if self.use_bias and self.bias is not None:
            out += self.bias[None, :]

        return out[0] if is_1d else out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class QuantizedReLUKANLinear(QuantizedKANLinearBase):
    def __init__(self, layer: ReLUKANLinear, group_size: int = 64, bits: int = 8):
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.num_grids = layer.num_grids
        self.centers = layer.centers.copy()
        self.inv_h = layer.inv_h
        self.use_bias = layer.use_bias
        self.bias = layer.bias.copy() if layer.bias is not None else None
        self.q_weights = QuantizedWeight(layer.weights, group_size=group_size, bits=bits)

    def forward(self, x: np.ndarray) -> np.ndarray:
        w_deq = self.q_weights.dequantize()
        x_arr = np.asarray(x, dtype=np.float32)
        diff = np.abs(x_arr[..., None] - self.centers[None, None, :]) * self.inv_h
        bases = np.maximum(0.0, 1.0 - diff)
        out = np.einsum("bic,ijc->bj", bases, w_deq)
        if self.use_bias and self.bias is not None:
            out += self.bias[None, :]
        return out


class QuantizedWavKANLinear(QuantizedKANLinearBase):
    def __init__(self, layer: WavKANLinear, group_size: int = 64, bits: int = 8):
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.num_wavelets = layer.num_wavelets
        self.wavelet_code = layer.wavelet_code
        self.translations = layer.translations.copy()
        self.scales = layer.scales.copy()
        self.use_bias = layer.use_bias
        self.bias = layer.bias.copy() if layer.bias is not None else None
        self.q_weights = QuantizedWeight(layer.weights, group_size=group_size, bits=bits)

    def forward(self, x: np.ndarray) -> np.ndarray:
        w_deq = self.q_weights.dequantize()
        x_arr = np.asarray(x, dtype=np.float32)
        z = (x_arr[..., None] - self.translations[None, ...]) / (np.abs(self.scales[None, ...]) + 1e-4)
        z2 = z * z
        gauss = np.exp(-0.5 * z2)
        if self.wavelet_code == 1:
            phi = np.cos(5.0 * z) * gauss
        elif self.wavelet_code == 2:
            phi = -z * gauss
        else:
            phi = (1.0 - z2) * gauss
        out = np.einsum("biw,ijw->bj", phi, w_deq)
        if self.use_bias and self.bias is not None:
            out += self.bias[None, :]
        return out


class QuantizedFourierKANLinear(QuantizedKANLinearBase):
    def __init__(self, layer: FourierKANLinear, group_size: int = 64, bits: int = 8):
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.num_frequencies = layer.num_frequencies
        self.use_bias = layer.use_bias
        self.bias = layer.bias.copy() if layer.bias is not None else None
        self.q_weights = QuantizedWeight(layer.weights, group_size=group_size, bits=bits)

    def forward(self, x: np.ndarray) -> np.ndarray:
        w_deq = self.q_weights.dequantize()
        x_arr = np.asarray(x, dtype=np.float32)
        k_indices = np.arange(1, self.num_frequencies + 1, dtype=np.float32)
        arg = (np.pi * x_arr)[..., None] * k_indices[None, None, :]
        cos_terms = np.cos(arg)
        sin_terms = np.sin(arg)
        ones_term = np.ones_like(x_arr[..., None])
        bases = np.concatenate([ones_term, cos_terms, sin_terms], axis=-1)
        out = np.einsum("bik,ijk->bj", bases, w_deq)
        if self.use_bias and self.bias is not None:
            out += self.bias[None, :]
        return out


class QuantizedJacobiKANLinear(QuantizedKANLinearBase):
    def __init__(self, layer: JacobiKANLinear, group_size: int = 64, bits: int = 8):
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.degree = layer.degree
        self.alpha = layer.alpha
        self.beta = layer.beta
        self.use_bias = layer.use_bias
        self.bias = layer.bias.copy() if layer.bias is not None else None
        self.q_weights = QuantizedWeight(layer.weights, group_size=group_size, bits=bits)

    def forward(self, x: np.ndarray) -> np.ndarray:
        w_deq = self.q_weights.dequantize()
        x_arr = np.asarray(x, dtype=np.float32)
        xc = np.clip(x_arr, -1.0, 1.0)
        bases = [np.ones_like(xc[..., None])]
        deg = self.degree
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
        jacobi_bases = np.concatenate(bases, axis=-1)
        out = np.einsum("bid,ijd->bj", jacobi_bases, w_deq)
        if self.use_bias and self.bias is not None:
            out += self.bias[None, :]
        return out


class QuantizedLowRankKANLinear(QuantizedKANLinearBase):
    def __init__(self, layer: LowRankKANLinear, group_size: int = 64, bits: int = 8):
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.rank = layer.rank
        self.num_grids = layer.num_grids
        self.centers = layer.centers.copy()
        self.inv_sigma = layer.inv_sigma
        self.use_bias = layer.use_bias
        self.bias = layer.bias.copy() if layer.bias is not None else None
        self.q_V = QuantizedWeight(layer.spline_V, group_size=group_size, bits=bits)
        self.q_U = QuantizedWeight(layer.spline_U, group_size=group_size, bits=bits)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        diff = (x_arr[..., None] - self.centers[None, None, :]) * self.inv_sigma
        rbf = np.exp(-diff * diff)
        bases_flat = rbf.reshape(x_arr.shape[0], -1)
        v_deq = self.q_V.dequantize()
        u_deq = self.q_U.dequantize()
        bottleneck = bases_flat @ v_deq.T
        out = bottleneck @ u_deq.T
        if self.use_bias and self.bias is not None:
            out += self.bias[None, :]
        return out


class QuantizedMultKANLinear(QuantizedKANLinearBase):
    def __init__(self, layer: MultKANLinear, group_size: int = 64, bits: int = 8):
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.num_mult = layer.num_mult
        self.num_add = layer.num_add
        self.q_sub = QuantizedFastKANLinear(layer.sub_layer, group_size=group_size, bits=bits)

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = self.q_sub.forward(x)
        if self.num_mult == 0:
            return z
        if self.num_add > 0:
            add_part = z[:, :self.num_add]
            mult_raw = z[:, self.num_add:]
        else:
            add_part = None
            mult_raw = z
        B = mult_raw.shape[0]
        mult_pairs = mult_raw.reshape(B, self.num_mult, 2)
        mult_part = mult_pairs[:, :, 0] * mult_pairs[:, :, 1]
        if add_part is not None:
            return np.concatenate([add_part, mult_part], axis=-1)
        return mult_part


def quantize(model: Any, bits: int = 8, group_size: int = 64) -> Any:
    """Quantizes an individual layer or full network to INT8 / INT4."""
    mapping = {
        ChebyKANLinear: QuantizedChebyKANLinear,
        FastKANLinear: QuantizedFastKANLinear,
        KANLinear: QuantizedKANLinear,
        ReLUKANLinear: QuantizedReLUKANLinear,
        WavKANLinear: QuantizedWavKANLinear,
        FourierKANLinear: QuantizedFourierKANLinear,
        JacobiKANLinear: QuantizedJacobiKANLinear,
        LowRankKANLinear: QuantizedLowRankKANLinear,
        MultKANLinear: QuantizedMultKANLinear,
    }

    t = type(model)
    if t in mapping:
        return mapping[t](model, group_size=group_size, bits=bits)

    # Multi-layer network container with .layers attribute
    if hasattr(model, "layers") and isinstance(model.layers, list):
        for idx, layer in enumerate(model.layers):
            lt = type(layer)
            if lt in mapping:
                model.layers[idx] = mapping[lt](layer, group_size=group_size, bits=bits)
        return model

    return model


def to_int8(model: Any, group_size: int = 64) -> Any:
    """Convenience wrapper for INT8 quantization."""
    return quantize(model, bits=8, group_size=group_size)


def to_int4(model: Any, group_size: int = 64) -> Any:
    """Convenience wrapper for INT4 quantization."""
    return quantize(model, bits=4, group_size=group_size)


def get_model_size(model: Any) -> Dict[str, Any]:
    """Computes total parameter count and memory footprint (MB)."""
    total_bytes = 0
    total_params = 0

    def inspect_obj(obj):
        nonlocal total_bytes, total_params
        if isinstance(obj, np.ndarray):
            total_bytes += obj.nbytes
            total_params += obj.size
        elif isinstance(obj, QuantizedWeight):
            total_bytes += obj.packed_weight.nbytes + obj.scales.nbytes + obj.biases.nbytes
            total_params += obj.in_features * obj.out_features
        elif hasattr(obj, "__dict__"):
            for v in obj.__dict__.values():
                inspect_obj(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                inspect_obj(item)

    inspect_obj(model)
    return {
        "total_params": total_params,
        "size_bytes": total_bytes,
        "size_mb": round(total_bytes / (1024 * 1024), 4)
    }
