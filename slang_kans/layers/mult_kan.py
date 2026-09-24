"""
MultKAN (KAN 2.0): Kolmogorov-Arnold Networks with Multiplication Nodes in Slang.
Introduces multiplicative interaction nodes (u * v) alongside standard additive nodes.
"""

from typing import Optional, Sequence, Union, Any, List
import numpy as np
from slang_kans.layers.base import SlangKANLayerBase
from slang_kans.layers.fast_kan import FastKANLinear


class MultKANLinear(SlangKANLayerBase):
    """
    MultKAN (KAN 2.0) Layer with additive and multiplicative channels.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_mult: Optional[int] = None,
        num_grids: int = 8,
        bias: bool = True,
        use_gpu: bool = True
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features

        if num_mult is None:
            self.num_mult = max(1, out_features // 2) if out_features > 1 else 0
        else:
            self.num_mult = min(num_mult, out_features)

        self.num_add = out_features - self.num_mult
        internal_out = self.num_add + 2 * self.num_mult

        self.sub_layer = FastKANLinear(
            in_features=in_features,
            out_features=internal_out,
            num_centers=num_grids,
            bias=bias,
            use_gpu=use_gpu
        )

    def count_parameters(self) -> int:
        return self.sub_layer.count_parameters()

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        is_1d = x_arr.ndim == 1
        if is_1d:
            x_arr = x_arr.reshape(1, -1)

        z = self.sub_layer.forward(x_arr)

        if self.num_mult == 0:
            return z[0] if is_1d else z

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
            out = np.concatenate([add_part, mult_part], axis=-1)
        else:
            out = mult_part

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


class MultKAN:
    """Multi-layer MultKAN Network."""

    def __init__(
        self,
        layers_hidden: Sequence[int],
        num_mult: Optional[int] = None,
        num_grids: int = 8,
        bias: bool = True,
        use_gpu: bool = True
    ):
        self.layers_hidden = list(layers_hidden)
        self.layers = [
            MultKANLinear(
                in_f,
                out_f,
                num_mult=num_mult,
                num_grids=num_grids,
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
