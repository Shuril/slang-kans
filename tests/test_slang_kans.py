"""
Comprehensive unit tests for slang-KANs.
Tests all 10 architectures, INT8/INT4 quantization, pruning, symbolic extraction,
and memory checkpointing across cross-platform Slang backends (Vulkan/Metal/CUDA/CPU).
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import slang_kans


class TestSlangKANs(unittest.TestCase):

    def test_01_device_and_availability(self):
        self.assertTrue(slang_kans.is_slang_available(), "slangpy should be available")
        dev = slang_kans.get_device()
        self.assertIsNotNone(dev.device, "Slang device instance should be initialized")
        self.assertTrue(any(t in dev.device_type.lower() for t in ["metal", "vulkan", "cuda", "cpu", "automatic"]))

    def test_02_all_10_architectures(self):
        B, D_in, D_out = 16, 4, 8
        x = np.random.uniform(-1.0, 1.0, (B, D_in)).astype(np.float32)

        architectures = [
            ("KAN (B-spline)", slang_kans.KAN(D_in, D_out, grid_size=5)),
            ("FastKAN (RBF)", slang_kans.FastKAN(D_in, D_out, num_centers=8)),
            ("ChebyKAN (Chebyshev)", slang_kans.ChebyKAN(D_in, D_out, degree=4)),
            ("ReLUKAN (Piecewise Tent)", slang_kans.ReLUKANLinear(D_in, D_out, num_grids=8)),
            ("WavKAN (Wavelets)", slang_kans.WavKANLinear(D_in, D_out, num_wavelets=8)),
            ("FourierKAN (Fourier Series)", slang_kans.FourierKANLinear(D_in, D_out, num_frequencies=4)),
            ("JacobiKAN (Jacobi)", slang_kans.JacobiKANLinear(D_in, D_out, degree=4)),
            ("MultKAN (KAN 2.0)", slang_kans.MultKANLinear(D_in, D_out, num_mult=4)),
            ("LowRankKAN (Bottleneck)", slang_kans.LowRankKANLinear(D_in, D_out, rank=4)),
            ("RationalKAN (Padé-Chebyshev)", slang_kans.RationalKANLinear(D_in, D_out, p_degree=4, q_degree=2)),
        ]

        for name, layer in architectures:
            out = layer.forward(x)
            self.assertEqual(out.shape, (B, D_out), f"{name} output shape mismatch")
            self.assertFalse(np.any(np.isnan(out)), f"{name} produced NaN")
            params = slang_kans.count_parameters(layer)
            self.assertGreater(params["total"], 0, f"{name} parameter count should be > 0")

    def test_03_multilayer_network(self):
        B = 8
        x = np.random.uniform(-1.0, 1.0, (B, 6)).astype(np.float32)
        net = slang_kans.ReLUKAN([6, 12, 3], num_grids=6)
        out = net(x)
        self.assertEqual(out.shape, (B, 3))
        self.assertFalse(np.any(np.isnan(out)))

    def test_04_quantization(self):
        layer = slang_kans.ChebyKANLinear(8, 16, degree=4)
        x = np.random.uniform(-1.0, 1.0, (10, 8)).astype(np.float32)
        orig_out = layer.forward(x)

        # INT8
        q8 = slang_kans.to_int8(layer)
        q8_out = q8.forward(x)
        err8 = np.max(np.abs(orig_out - q8_out))
        self.assertLess(err8, 0.1, f"INT8 max error too high: {err8}")

        # INT4
        q4 = slang_kans.to_int4(layer)
        q4_out = q4.forward(x)
        err4 = np.max(np.abs(orig_out - q4_out))
        self.assertLess(err4, 0.2, f"INT4 max error too high: {err4}")

        # Memory size check
        fp32_size = slang_kans.get_model_size(layer)["size_bytes"]
        int8_size = slang_kans.get_model_size(q8)["size_bytes"]
        self.assertLess(int8_size, fp32_size)

    def test_05_pruning_and_compaction(self):
        net = slang_kans.FastKAN([4, 8, 2], num_centers=6)
        p_info = slang_kans.prune(net, threshold=0.05)
        self.assertIn("sparsity", p_info)
        self.assertGreaterEqual(p_info["sparsity"], 0.0)

        importance = slang_kans.compute_node_importance(net)
        self.assertEqual(len(importance), 1)

        compacted = slang_kans.compact_kan(net, threshold=1e-5)
        self.assertIsNotNone(compacted)

    def test_06_symbolic_extraction(self):
        layer = slang_kans.ChebyKANLinear(2, 1, degree=2)
        sym = slang_kans.to_symbolic(layer)
        x = np.array([[0.5, -0.3]], dtype=np.float32)
        y_sym = sym.forward(x)
        self.assertEqual(y_sym.shape, (1, 1))

        c_code = sym.to_c_code()
        self.assertIn("void symbolic_kan_eval", c_code)

    def test_07_checkpointing(self):
        model = slang_kans.FastKAN([4, 8, 2], num_centers=4)
        cp_model = slang_kans.checkpoint_kan(model)
        x = np.random.uniform(-1.0, 1.0, (4, 4)).astype(np.float32)
        out = cp_model(x)
        self.assertEqual(out.shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
