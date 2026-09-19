"""
slang-KANs: High-Performance Cross-Platform Kolmogorov-Arnold Networks Suite.
Hardware-accelerated with Slang shaders (Metal, Vulkan, CUDA, CPU), Tensor Cores (linalg::CoopMat),
INT8/INT4 quantization, structural pruning, activation checkpointing, and symbolic extraction.
Complete 1:1 API parity with mlx-KANs.
"""

from slang_kans.device import (
    is_slang_available,
    get_device,
    SlangDeviceManager
)

__version__ = "0.2.0"

from slang_kans.layers import (
    # Core B-Spline KAN
    KAN,
    KANLinear,
    # FastKAN (Gaussian RBF)
    FastKAN,
    FastKANLinear,
    # ReLUKAN (Piecewise Linear Tent)
    ReLUKAN,
    ReLUKANLinear,
    # ChebyKAN (Chebyshev Polynomials)
    ChebyKAN,
    ChebyKANLinear,
    # Wav-KAN (Continuous Wavelets)
    WavKAN,
    WavKANLinear,
    # FourierKAN (Harmonic Series)
    FourierKAN,
    FourierKANLinear,
    # JacobiKAN (Orthogonal Jacobi Polynomials)
    JacobiKAN,
    JacobiKANLinear,
    # MultKAN (KAN 2.0 with Multiplication Nodes)
    MultKAN,
    MultKANLinear,
    # LowRankKAN (Bottleneck / LoRA Factorization)
    LowRankKAN,
    LowRankKANLinear,
    # RationalKAN (Padé-Chebyshev Rational Functions)
    RationalKAN,
    RationalKANLinear,
)

from slang_kans.quantized import (
    QuantizedWeight,
    QuantizedKANLinear,
    QuantizedFastKANLinear,
    QuantizedReLUKANLinear,
    QuantizedChebyKANLinear,
    QuantizedWavKANLinear,
    QuantizedFourierKANLinear,
    QuantizedJacobiKANLinear,
    QuantizedLowRankKANLinear,
    QuantizedMultKANLinear,
    quantize,
    to_int8,
    to_int4,
    get_model_size,
)

from slang_kans.pruning import (
    compute_node_importance,
    prune,
    compact_kan,
)

from slang_kans.symbolic import (
    to_symbolic,
    SymbolicKAN,
    SymbolicLayer,
    SymbolicEdge,
)

from slang_kans.checkpoint import (
    checkpoint_kan,
    CheckpointedKAN,
)

from slang_kans.utils import (
    build_train_step,
    count_parameters,
    to_fp16,
    to_bf16,
)

__all__ = [
    # Device & Availability
    "is_slang_available",
    "get_device",
    "SlangDeviceManager",
    # Core Architectures (10 classes)
    "KAN",
    "KANLinear",
    "FastKAN",
    "FastKANLinear",
    "ReLUKAN",
    "ReLUKANLinear",
    "ChebyKAN",
    "ChebyKANLinear",
    "WavKAN",
    "WavKANLinear",
    "FourierKAN",
    "FourierKANLinear",
    "JacobiKAN",
    "JacobiKANLinear",
    "MultKAN",
    "MultKANLinear",
    "LowRankKAN",
    "LowRankKANLinear",
    "RationalKAN",
    "RationalKANLinear",
    # Quantization INT8 & INT4
    "QuantizedWeight",
    "QuantizedKANLinear",
    "QuantizedFastKANLinear",
    "QuantizedReLUKANLinear",
    "QuantizedChebyKANLinear",
    "QuantizedWavKANLinear",
    "QuantizedFourierKANLinear",
    "QuantizedJacobiKANLinear",
    "QuantizedLowRankKANLinear",
    "QuantizedMultKANLinear",
    "quantize",
    "to_int8",
    "to_int4",
    "get_model_size",
    # Sparsity & Pruning
    "compute_node_importance",
    "prune",
    "compact_kan",
    # Symbolic Extraction
    "to_symbolic",
    "SymbolicKAN",
    "SymbolicLayer",
    "SymbolicEdge",
    # Checkpointing & Memory
    "checkpoint_kan",
    "CheckpointedKAN",
    # Utilities & Training
    "build_train_step",
    "count_parameters",
    "to_fp16",
    "to_bf16",
]
