"""
LowRankKAN (Bottleneck KAN / LoRA-KAN):
Kolmogorov-Arnold Networks with Low-Rank Factorized Spline Projections in Slang.
Factorizes weights into U @ V (rank r << min(d_in, d_out)) for significant parameter reduction.
"""

from typing import Optional, Sequence, Union, Any, List
import numpy as np
from slang_kans.layers.base import SlangKANLayerBase


class LowRankKANLinear(SlangKANLayerBase):
    """
    LowRankKAN Layer with rank-factorized spline projection.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        num_grids: int = 8,
        grid_range: tuple = (-1.0, 1.0),
        bias: bool = True,
        use_gpu: bool = False
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.rank = min(rank, out_features, in_features * num_grids)
        self.num_grids = num_grids
        self.grid_min, self.grid_max = grid_range

        self.centers = np.linspace(self.grid_min, self.grid_max, num_grids, dtype=np.float32)
        step = (self.grid_max - self.grid_min) / max(num_grids - 1, 1)
        self.inv_sigma = np.float32(1.0 / max(step * 1.5, 1e-6))

        in_total = in_features * num_grids
        scale_v = 1.0 / np.sqrt(in_total)
        scale_u = 1.0 / np.sqrt(self.rank)
        self.spline_V = np.random.uniform(-scale_v, scale_v, (self.rank, in_total)).astype(np.float32)
        self.spline_U = np.random.uniform(-scale_u, scale_u, (out_features, self.rank)).astype(np.float32)

    def count_parameters(self) -> int:
        total = self.spline_V.size + self.spline_U.size
        if self.use_bias and self.bias is not None:
            total += self.bias.size
        return total

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        is_1d = x_arr.ndim == 1
        if is_1d:
            x_arr = x_arr.reshape(1, -1)

        B, D_in = x_arr.shape

        diff = (x_arr[..., None] - self.centers[None, None, :]) * self.inv_sigma
        rbf = np.exp(-diff * diff)  # [B, D_in, num_grids]
        bases_flat = rbf.reshape(B, -1)  # [B, D_in * num_grids]

        # Low-rank factorized projection: (bases @ V.T) @ U.T
        bottleneck = bases_flat @ self.spline_V.T  # [B, rank]
        out = bottleneck @ self.spline_U.T  # [B, D_out]

        if self.use_bias and self.bias is not None:
            out += self.bias[None, :]

        return out[0] if is_1d else out

    def benchmark(self, x: np.ndarray, warmup: int = 10, iters: int = 50) -> float:
        for _ in range(warmup):
            self.forward(x)
        import time
        start = time.perf_counter()
        for _ in range(iters):
            self.forward(x)
        end = time.perf_counter()
        return (end - start) / iters * 1000.0

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0) -> float:
        l1 = np.mean(np.abs(self.spline_U @ self.spline_V), axis=-1)
        reg_l1 = float(np.sum(l1))
        p = l1 / (reg_l1 + 1e-8)
        entropy = float(-np.sum(p * np.log(p + 1e-8)))
        return regularize_activation * reg_l1 + regularize_entropy * entropy


class LowRankKAN:
    """Multi-layer LowRankKAN Network."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        rank: int = 8,
        num_grids: int = 8,
        grid_range: tuple = (-1.0, 1.0),
        bias: bool = True
    ):
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            LowRankKANLinear(
                in_f,
                out_f,
                rank=rank,
                num_grids=num_grids,
                grid_range=grid_range,
                bias=bias
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
