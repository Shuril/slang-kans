"""
Layers and Architectures for slang-KANs.
"""

from slang_kans.layers.base import SlangKANLayerBase
from slang_kans.layers.kan import KAN
from slang_kans.layers.fast_kan import FastKAN, FastKANLinear
from slang_kans.layers.cheby_kan import ChebyKAN
from slang_kans.layers.relu_kan import ReLUKAN, ReLUKANLinear
from slang_kans.layers.wav_kan import WavKAN, WavKANLinear
from slang_kans.layers.fourier_kan import FourierKAN, FourierKANLinear
from slang_kans.layers.jacobi_kan import JacobiKAN, JacobiKANLinear
from slang_kans.layers.mult_kan import MultKAN, MultKANLinear
from slang_kans.layers.low_rank_kan import LowRankKAN, LowRankKANLinear
from slang_kans.layers.rational_kan import RationalKAN, RationalKANLinear
from slang_kans.layers.gated_kan import GatedKAN
from slang_kans.layers.slang_kan import SlangKAN

# Aliases for parity with mlx_kans
KANLinear = KAN
ChebyKANLinear = ChebyKAN

__all__ = [
    "SlangKAN",
    "SlangKANLayerBase",
    # Core B-Spline KAN
    "KAN",
    "KANLinear",
    # FastKAN (Gaussian RBF)
    "FastKAN",
    "FastKANLinear",
    # ChebyKAN (Chebyshev Polynomials)
    "ChebyKAN",
    "ChebyKANLinear",
    # ReLUKAN (Piecewise Linear Tent)
    "ReLUKAN",
    "ReLUKANLinear",
    # WavKAN (Continuous Wavelets)
    "WavKAN",
    "WavKANLinear",
    # FourierKAN (Harmonic Fourier Series)
    "FourierKAN",
    "FourierKANLinear",
    # JacobiKAN (Orthogonal Jacobi Polynomials)
    "JacobiKAN",
    "JacobiKANLinear",
    # MultKAN (KAN 2.0 with Multiplication Nodes)
    "MultKAN",
    "MultKANLinear",
    # LowRankKAN (Bottleneck Factorization)
    "LowRankKAN",
    "LowRankKANLinear",
    # RationalKAN (Padé-Chebyshev Rational Functions)
    "RationalKAN",
    "RationalKANLinear",
]
