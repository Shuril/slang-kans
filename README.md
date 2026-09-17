# slang-KANs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Slang](https://img.shields.io/badge/Slang-Shader%20Language-orange.svg)](https://shader-slang.com/)
[![Platforms](https://img.shields.io/badge/Platforms-Metal%20%7C%20Vulkan%20%7C%20CUDA%20%7C%20CPU-green.svg)](https://github.com/shader-slang/slang)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**slang-KANs** is a high-performance, cross-platform library of **Kolmogorov-Arnold Networks (KAN)** written in [NVIDIA Slang](https://shader-slang.com/) and [slangpy](https://github.com/shader-slang/slangpy).

It compiles a single set of GPU compute shaders directly to **Metal (macOS)**, **Vulkan (Linux/Windows)**, **CUDA (NVIDIA)**, and **C++ (CPU)**, providing uniform numerical accuracy and hardware acceleration across different GPU vendors.

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
