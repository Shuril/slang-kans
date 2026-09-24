# slang-KANs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Slang](https://img.shields.io/badge/Slang-Shader%20Language-orange.svg)](https://shader-slang.com/)
[![Platforms](https://img.shields.io/badge/Platforms-Metal%20%7C%20Vulkan%20%7C%20CUDA%20%7C%20CPU-green.svg)](https://github.com/shader-slang/slang)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**slang-KANs** is a high-performance, cross-platform library of **Kolmogorov-Arnold Networks (KAN)** written in [NVIDIA Slang](https://shader-slang.com/) and [slangpy](https://github.com/shader-slang/slangpy).

It compiles a single set of GPU compute shaders directly to **Metal (macOS)**, **Vulkan (Linux/Windows)**, **CUDA (NVIDIA)**, and **C++ (CPU)**, providing uniform numerical accuracy and hardware acceleration across different GPU vendors.

> *Русскоязычная версия документации доступна в [README_RU.md](README_RU.md).*

---

## Features

- **10 KAN Architectures**:
  - `FastKAN`: Gaussian RBF basis functions for smooth nonlinear activations.
  - `ReLUKAN`: Piecewise-linear tent basis with zero transcendental math operations.
  - `ChebyKAN`: Chebyshev polynomials of the 1st kind with Clenshaw recurrence.
  - `WavKAN`: Continuous wavelets (Mexican Hat, Morlet, DOG) for multiresolution time-frequency localization.
  - `FourierKAN`: Trigonometric Fourier series for periodic and harmonic targets.
  - `JacobiKAN`: Orthogonal Jacobi polynomials parameterized by $(\alpha, \beta)$.
  - `MultKAN`: KAN 2.0 architecture with explicit multiplication nodes ($u \cdot v$).
  - `LowRankKAN`: Bottleneck matrix factorization reducing parameters by up to 10x.
  - `RationalKAN`: Padé-Chebyshev rational functions ($P(x)/Q(x)$) for modeling singularities and steep gradients.
  - `KAN`: Cox-de Boor B-splines with cubic interpolation.

- **Cross-Platform GPU Kernels**:
  - Single shader source per architecture in `slang_kans/kernels/*.slang`.
  - JIT-compiled at runtime via `slangpy` to match the target host device.
  - Cooperative matrix abstraction (`linalg::CoopMat`) for hardware Tensor Cores (NVIDIA wmma, Vulkan cooperative matrix, Apple simdgroup).

- **Compression & Deployment**:
  - **INT8 / INT4 Quantization**: Channel-wise affine weight quantization.
  - **Structural Pruning**: Node-level importance scoring and physical layer compaction.
  - **Symbolic Extraction**: Distillation of trained edge curves into SymPy equations.
  - **C99 Header Export**: Zero-dependency standalone C functions (`.h`) for embedded deployment.
  - **Activation Checkpointing**: Recomputes basis activations during backward pass to reduce peak memory.
  - **Built-in Optimizers**: Pure NumPy/cross-platform suite including `AdamW`, `Muon` (5th-order Newton-Schulz orthogonalization), `Lion` (sign momentum), `RMSprop`, `Adam`, and `SGD`.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Shuril/slang-kans.git
cd slang-kans

# Install in editable mode
pip install -e .
```

### Requirements
- Python >= 3.9
- `slangpy >= 0.43.0`
- `numpy >= 1.22.0`
- `sympy` (optional, for symbolic equation distillation)

---

## Quick Start

```python
import numpy as np
import slang_kans

# 1. Instantiate a KAN model
model = slang_kans.FastKAN(layers_hidden=[2, 16, 1], num_centers=8)

# 2. Forward pass with NumPy arrays (executes on GPU via Slang)
x = np.random.uniform(-1.0, 1.0, (32, 2)).astype(np.float32)
y = model(x)
print("Output shape:", y.shape)

# 3. Post-training INT8 quantization
q_model = slang_kans.to_int8(model)
y_quant = q_model(x)

# 4. Extract closed-form symbolic equation
cheby_layer = slang_kans.ChebyKANLinear(2, 1, degree=3)
sym = slang_kans.to_symbolic(cheby_layer)
print("Symbolic formula:", sym.formula())

# 5. Export to pure ANSI C99 header
c_code = sym.to_c_code(func_name="evaluate_kan")

# 6. Built-in Optimizers (AdamW, Muon, Lion, RMSprop, SGD)
from slang_kans import AdamW, Muon, Lion
opt = Muon([(model.layers[0].weights, grad_weights)], lr=0.02, momentum=0.95)
opt.step()
```

---

## Supported Architectures

| Architecture | Basis Function | Key Property | Typical Use Case |
|---|---|---|---|
| `FastKAN` | Gaussian RBF $\exp(-d^2 / \sigma^2)$ | Fast evaluation, smooth gradients | General MLP replacement, vision |
| `ReLUKAN` | Piecewise-linear tent functions | Hardware-friendly, no transcendentals | Low-power devices, edge inference |
| `ChebyKAN` | Chebyshev polynomials $T_k(x)$ | Minimax polynomial optimality | Numerical physics, smooth functions |
| `WavKAN` | Wavelets (Mexican Hat, Morlet) | Time-frequency localization | Time-series, audio, non-stationary signals |
| `FourierKAN`| Trigonometric $(\cos, \sin)$ | Global harmonic analysis | Periodic signals, acoustic models |
| `JacobiKAN` | Jacobi polynomials $P_k^{(\alpha,\beta)}(x)$ | Generalized orthogonal basis | Boundary value problems, PDEs |
| `MultKAN` | B-splines + Multiplication nodes | Multiplicative feature interactions | Physical conservation laws, $u \cdot v$ |
| `LowRankKAN`| Low-rank linear projections | Significant parameter reduction | Wide/deep layers, high-dimensional inputs |
| `RationalKAN`| Padé rational functions $P(x)/Q(x)$ | Adaptive pole modeling | Stiff systems, asymptotic boundaries |
| `KAN` | Cubic Cox-de Boor B-splines | Traditional KAN formulation | Mathematical baseline |

---

## Performance Benchmarks

### 3-Way GPU Benchmark: MLX vs metal-KANs (v0.3.0) vs slang-KANs (v0.2.0)

Tested across all 10 architectures on **Apple Silicon GPU**, Layer `64 -> 64`:

| Architecture | Batch | MLX (ms) | metal-KANs (Pure Metal AMX) | slang-KANs (Shared-Memory GEMM) | Winner (Speedup) |
|---|---|---|---|---|---|
| **ChebyKAN** | 128 | 0.321 ms | 0.350 ms | **0.166 ms** | **slang-KANs (1.94x)** |
| | 1024 | 1.269 ms | 0.509 ms | **0.369 ms** | **slang-KANs (1.38x)** |
| | 4096 | 1.383 ms | **0.653 ms** | 0.892 ms | **metal-KANs (1.37x)** |
| **BSplineKAN** | 128 | 0.359 ms | 0.290 ms | **0.132 ms** | **slang-KANs (2.20x)** |
| | 1024 | 0.777 ms | **0.457 ms** | 0.545 ms | **metal-KANs (1.19x)** |
| | 4096 | 2.092 ms | **0.902 ms** | 1.402 ms | **metal-KANs (1.56x)** |
| **FastKAN** | 128 | **0.267 ms** | 0.297 ms | 0.696 ms | **MLX (1.11x)** |
| | 1024 | 0.446 ms | **0.399 ms** | 1.463 ms | **metal-KANs (1.12x)** |
| | 4096 | 1.445 ms | **1.177 ms** | 4.739 ms | **metal-KANs (1.23x)** |
| **WavKAN** | 128 | **0.335 ms** | 0.411 ms | 0.833 ms | **MLX (1.23x)** |
| | 1024 | 1.528 ms | **0.567 ms** | 1.835 ms | **metal-KANs (2.69x)** |
| | 4096 | 1.222 ms | **0.984 ms** | 3.762 ms | **metal-KANs (1.24x)** |
| **ReLUKAN** | 128 | **0.309 ms** | 0.378 ms | 0.861 ms | **MLX (1.22x)** |
| | 1024 | **0.513 ms** | 0.557 ms | 1.840 ms | **MLX (1.08x)** |
| | 4096 | **1.093 ms** | 1.135 ms | 3.250 ms | **MLX (1.04x)** |
| **FourierKAN** | 128 | **0.321 ms** | 0.381 ms | 0.896 ms | **MLX (1.19x)** |
| | 1024 | 0.872 ms | **0.641 ms** | 1.969 ms | **metal-KANs (1.36x)** |
| | 4096 | 2.219 ms | **0.928 ms** | 2.565 ms | **metal-KANs (2.39x)** |
| **JacobiKAN** | 128 | 0.292 ms | 0.298 ms | **0.144 ms** | **slang-KANs (2.03x)** |
| | 1024 | 0.578 ms | 0.465 ms | **0.359 ms** | **slang-KANs (1.30x)** |
| | 4096 | 1.521 ms | **0.574 ms** | 0.727 ms | **metal-KANs (1.27x)** |
| **RationalKAN**| 128 | 0.679 ms | 0.403 ms | **0.132 ms** | **slang-KANs (3.05x)** |
| | 1024 | 4.070 ms | 0.850 ms | **0.652 ms** | **slang-KANs (1.30x)** |
| | 4096 | 18.252 ms| 2.453 ms | **2.393 ms** | **slang-KANs (1.02x)** |
| **MultKAN** | 128 | **0.320 ms** | 0.509 ms | 0.803 ms | **MLX (1.59x)** |
| | 1024 | **0.428 ms** | 0.707 ms | 1.561 ms | **MLX (1.65x)** |
| | 4096 | **1.529 ms** | 1.853 ms | 6.638 ms | **MLX (1.21x)** |
| **LowRankKAN** | 128 | 0.344 ms | 0.454 ms | **0.301 ms** | **slang-KANs (1.14x)** |
| | 1024 | **0.608 ms** | 0.829 ms | 3.306 ms | **MLX (1.36x)** |
| | 4096 | **1.442 ms** | 2.257 ms | 11.785 ms| **MLX (1.57x)** |

### Optimization Architecture in v0.2.0:
- **16x16 Shared-Memory Tiled GEMM**: Introduced `gemm_tiled` in `gemm.slang` using threadgroup shared memory (`groupshared float sA[16][16], sB[16][16]`), cutting redundant memory traffic by up to 85%.
- **2-Pass Basis Architecture**: Decoupled basis computation from matrix contraction for batch sizes $B \ge 64$, ensuring sub-millisecond execution times.

---

## Architecture Overview

```
slang-kans/
├── slang_kans/
│   ├── kernels/           # Slang compute shaders (*.slang)
│   ├── layers/            # PyTorch/NumPy layer implementations
│   ├── device.py          # Device manager (Vulkan / Metal / CUDA / CPU)
│   ├── quantized.py       # INT8 / INT4 affine quantization
│   ├── pruning.py         # Node pruning and model compaction
│   ├── symbolic.py        # SymPy distillation and C99 code generation
│   ├── checkpoint.py      # Activation checkpointing
│   ├── optim.py           # Pure NumPy AdamW, Muon, Lion, RMSprop, SGD
│   └── utils.py           # Training step and parameter utilities
├── tests/
│   └── test_slang_kans.py # Full unit test suite
└── examples/
    └── quickstart.py      # Working usage example
```

---

## Running Tests

Run the test suite using Python's built-in `unittest`:

```bash
python -m unittest tests/test_slang_kans.py
```

Or using `pytest`:

```bash
pytest tests/
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
