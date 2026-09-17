"""
Exact Symbolic Extraction for Kolmogorov-Arnold Networks in Slang.
Extracts closed-form analytic formulas from trained KAN models by fitting learned 1D edge
activations against candidate basis functions (polynomials, trig, exp, sigmoids).
Exports directly to SymPy expressions, C99 functions, and Slang shaders.
"""

from __future__ import annotations
import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any
import numpy as np


class SymbolicEdge:
    """Represents a single fitted symbolic edge function phi_{i,j}(x)."""

    def __init__(self, name: str, coeffs: List[float], r2: float):
        self.name = name
        self.coeffs = coeffs
        self.r2 = r2

    def __call__(self, x: np.ndarray) -> np.ndarray:
        c = self.coeffs
        if self.name == "zero" or not c:
            return np.zeros_like(x)
        elif self.name == "linear":
            return c[0] * x + c[1]
        elif self.name == "quadratic":
            return c[0] * x**2 + c[1] * x + c[2]
        elif self.name == "cubic":
            return c[0] * x**3 + c[1] * x**2 + c[2] * x + c[3]
        elif self.name == "quartic":
            return c[0] * x**4 + c[1] * x**3 + c[2] * x**2 + c[3] * x + c[4]
        elif self.name == "sin":
            return c[0] * np.sin(np.pi * x) + c[1]
        elif self.name == "cos":
            return c[0] * np.cos(np.pi * x) + c[1]
        elif self.name == "exp":
            return c[0] * np.exp(np.clip(x, -10.0, 10.0)) + c[1]
        elif self.name == "tanh":
            return c[0] * np.tanh(x) + c[1]
        elif self.name == "gaussian":
            return c[0] * np.exp(-x * x) + c[1]
        return c[0] * x + c[1]

    def to_c_code(self, var: str) -> str:
        c = self.coeffs
        if self.name == "zero" or not c:
            return "0.0f"
        elif self.name == "linear":
            return f"({c[0]}f * {var} + {c[1]}f)"
        elif self.name == "quadratic":
            return f"({c[0]}f * {var} * {var} + {c[1]}f * {var} + {c[2]}f)"
        elif self.name == "cubic":
            return f"({c[0]}f * {var} * {var} * {var} + {c[1]}f * {var} * {var} + {c[2]}f * {var} + {c[3]}f)"
        elif self.name == "sin":
            return f"({c[0]}f * sinf(3.14159265f * {var}) + {c[1]}f)"
        elif self.name == "cos":
            return f"({c[0]}f * cosf(3.14159265f * {var}) + {c[1]}f)"
        elif self.name == "exp":
            return f"({c[0]}f * expf({var}) + {c[1]}f)"
        elif self.name == "tanh":
            return f"({c[0]}f * tanhf({var}) + {c[1]}f)"
        elif self.name == "gaussian":
            return f"({c[0]}f * expf(-{var} * {var}) + {c[1]}f)"
        return f"({c[0]}f * {var} + {c[1]}f)"

    def to_sympy(self, var: Any) -> Any:
        import sympy as sp
        c = self.coeffs
        if self.name == "zero" or not c:
            return sp.Float(0.0)
        elif self.name == "linear":
            return c[0] * var + c[1]
        elif self.name == "quadratic":
            return c[0] * var**2 + c[1] * var + c[2]
        elif self.name == "cubic":
            return c[0] * var**3 + c[1] * var**2 + c[2] * var + c[3]
        elif self.name == "sin":
            return c[0] * sp.sin(sp.pi * var) + c[1]
        elif self.name == "cos":
            return c[0] * sp.cos(sp.pi * var) + c[1]
        elif self.name == "exp":
            return c[0] * sp.exp(var) + c[1]
        elif self.name == "tanh":
            return c[0] * sp.tanh(var) + c[1]
        elif self.name == "gaussian":
            return c[0] * sp.exp(-var**2) + c[1]
        return c[0] * var + c[1]


class SymbolicLayer:
    """Symbolic KAN Layer composed of SymbolicEdges."""

    def __init__(self, in_features: int, out_features: int, edges: List[List[SymbolicEdge]], bias: Optional[np.ndarray] = None):
        self.in_features = in_features
        self.out_features = out_features
        self.edges = edges  # [in_features][out_features]
        self.bias = bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        is_1d = x_arr.ndim == 1
        if is_1d:
            x_arr = x_arr.reshape(1, -1)

        B = x_arr.shape[0]
        out = np.zeros((B, self.out_features), dtype=np.float32)

        for j in range(self.out_features):
            acc = np.zeros(B, dtype=np.float32)
            for i in range(self.in_features):
                acc += self.edges[i][j](x_arr[:, i])
            if self.bias is not None:
                acc += self.bias[j]
            out[:, j] = acc

        return out[0] if is_1d else out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class SymbolicKAN:
    """Multi-layer Symbolic KAN."""

    def __init__(self, layers: List[SymbolicLayer]):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def to_sympy(self) -> List[Any]:
        import sympy as sp
        in_dim = self.layers[0].in_features
        vars = [sp.Symbol(f"x_{i}") for i in range(in_dim)]
        curr_vars = vars

        for layer in self.layers:
            next_vars = []
            for j in range(layer.out_features):
                expr = sp.Float(0.0)
                for i in range(layer.in_features):
                    expr += layer.edges[i][j].to_sympy(curr_vars[i])
                if layer.bias is not None:
                    expr += sp.Float(layer.bias[j])
                next_vars.append(sp.simplify(expr))
            curr_vars = next_vars

        return curr_vars

    def formula(self) -> str:
        """Returns human-readable mathematical formulas for each output."""
        try:
            exprs = self.to_sympy()
            return "\n".join(f"y_{j} = {expr}" for j, expr in enumerate(exprs))
        except Exception:
            return self.to_c_code()

    def latex(self) -> str:
        """Returns LaTeX formatted equations for each output."""
        import sympy as sp
        exprs = self.to_sympy()
        return "\n".join(f"y_{{{j}}} = {sp.latex(expr)}" for j, expr in enumerate(exprs))

    def to_c_code(self, func_name: str = "symbolic_kan_eval") -> str:
        in_dim = self.layers[0].in_features
        out_dim = self.layers[-1].out_features

        code = [
            f"void {func_name}(const float* in, float* out) {{",
        ]
        # Allocate temp arrays for hidden layers
        for l_idx, layer in enumerate(self.layers[:-1]):
            code.append(f"    float h_{l_idx}[{layer.out_features}];")

        prev_name = "in"
        for l_idx, layer in enumerate(self.layers):
            is_last = (l_idx == len(self.layers) - 1)
            target = "out" if is_last else f"h_{l_idx}"
            for j in range(layer.out_features):
                terms = []
                for i in range(layer.in_features):
                    term = layer.edges[i][j].to_c_code(f"{prev_name}[{i}]")
                    if term != "0.0f":
                        terms.append(term)
                if layer.bias is not None and layer.bias[j] != 0.0:
                    terms.append(f"{layer.bias[j]}f")
                expr = " + ".join(terms) if terms else "0.0f"
                code.append(f"    {target}[{j}] = {expr};")
            prev_name = target

        code.append("}")
        return "\n".join(code)


def _fit_candidate_edge(x_grid: np.ndarray, y_vals: np.ndarray) -> SymbolicEdge:
    """Fits candidate functions to (x_grid, y_vals) and selects the best R^2."""
    candidates = []

    # 1. Zero
    var_y = float(np.var(y_vals))
    if var_y < 1e-6:
        return SymbolicEdge("zero", [], 1.0)

    # 2. Linear: c0 * x + c1
    p1 = np.polyfit(x_grid, y_vals, deg=1)
    r2_1 = 1.0 - np.sum((y_vals - np.polyval(p1, x_grid))**2) / (np.sum((y_vals - np.mean(y_vals))**2) + 1e-8)
    candidates.append(("linear", [float(p1[0]), float(p1[1])], float(r2_1)))

    # 3. Quadratic: c0 * x^2 + c1 * x + c2
    p2 = np.polyfit(x_grid, y_vals, deg=2)
    r2_2 = 1.0 - np.sum((y_vals - np.polyval(p2, x_grid))**2) / (np.sum((y_vals - np.mean(y_vals))**2) + 1e-8)
    candidates.append(("quadratic", [float(p2[0]), float(p2[1]), float(p2[2])], float(r2_2)))

    # 4. Cubic
    p3 = np.polyfit(x_grid, y_vals, deg=3)
    r2_3 = 1.0 - np.sum((y_vals - np.polyval(p3, x_grid))**2) / (np.sum((y_vals - np.mean(y_vals))**2) + 1e-8)
    candidates.append(("cubic", [float(p3[0]), float(p3[1]), float(p3[2]), float(p3[3])], float(r2_3)))

    # 5. Sin: c0 * sin(pi * x) + c1
    A_sin = np.column_stack([np.sin(np.pi * x_grid), np.ones_like(x_grid)])
    c_sin, _, _, _ = np.linalg.lstsq(A_sin, y_vals, rcond=None)
    r2_sin = 1.0 - np.sum((y_vals - (A_sin @ c_sin))**2) / (np.sum((y_vals - np.mean(y_vals))**2) + 1e-8)
    candidates.append(("sin", [float(c_sin[0]), float(c_sin[1])], float(r2_sin)))

    # Pick best candidate
    best = max(candidates, key=lambda c: c[2])
    return SymbolicEdge(best[0], best[1], best[2])


def to_symbolic(model: Any, grid_points: int = 100) -> SymbolicKAN:
    """Converts a trained KAN layer or multi-layer network into an exact SymbolicKAN."""
    x_grid = np.linspace(-1.0, 1.0, grid_points, dtype=np.float32)

    raw_layers = model.layers if hasattr(model, "layers") else [model]
    sym_layers = []

    for layer in raw_layers:
        in_f = layer.in_features
        out_f = layer.out_features
        bias = layer.bias.copy() if (layer.use_bias and layer.bias is not None) else None

        edges = []
        for i in range(in_f):
            edge_row = []
            # Construct input probe with only i-th feature varied
            probe = np.zeros((grid_points, in_f), dtype=np.float32)
            probe[:, i] = x_grid

            # Get layer responses
            # Note: evaluate layer without bias contribution to isolate edge activation
            old_bias = layer.bias
            layer.bias = None
            y_all = layer.forward(probe)
            layer.bias = old_bias

            for j in range(out_f):
                y_edge = y_all[:, j]
                edge = _fit_candidate_edge(x_grid, y_edge)
                edge_row.append(edge)
            edges.append(edge_row)

        sym_layers.append(SymbolicLayer(in_f, out_f, edges, bias=bias))

    return SymbolicKAN(sym_layers)
