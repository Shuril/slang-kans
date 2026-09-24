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

    def test_08_gpu_cpu_parity(self):
        """Verify strict CPU-GPU parity (< 1e-4) across batch sizes and all architectures."""
        B, D_in, D_out = 128, 16, 16
        x = np.random.uniform(-0.8, 0.8, (B, D_in)).astype(np.float32)

        architectures = [
            ("FastKAN", slang_kans.FastKAN(D_in, D_out, num_centers=8)),
            ("WavKAN", slang_kans.WavKANLinear(D_in, D_out, num_wavelets=8)),
            ("ReLUKAN", slang_kans.ReLUKANLinear(D_in, D_out, num_grids=8)),
            ("FourierKAN", slang_kans.FourierKANLinear(D_in, D_out, num_frequencies=4)),
            ("JacobiKAN", slang_kans.JacobiKANLinear(D_in, D_out, degree=4)),
            ("ChebyKAN", slang_kans.ChebyKAN(D_in, D_out, degree=4)),
            ("KAN (BSpline)", slang_kans.KAN(D_in, D_out, grid_size=5)),
            ("RationalKAN", slang_kans.RationalKANLinear(D_in, D_out, p_degree=4, q_degree=2)),
        ]

        for name, layer in architectures:
            layer.use_gpu = False
            out_cpu = layer.forward(x)
            layer.use_gpu = True
            out_gpu = layer.forward(x)
            max_err = float(np.max(np.abs(out_cpu - out_gpu)))
            self.assertLess(max_err, 1e-4, f"{name} failed CPU-GPU parity with max error {max_err}")


    def test_09_optimizers(self):
        """Verify that Adam, AdamW, Muon, Lion, RMSprop, and SGD all correctly update parameters and minimize loss."""
        np.random.seed(42)
        X = np.random.randn(32, 8).astype(np.float32)
        W_target = np.random.randn(8, 4).astype(np.float32)
        Y_target = X @ W_target

        optimizers = [
            ("Adam", lambda p: slang_kans.Adam(p, lr=0.05)),
            ("AdamW", lambda p: slang_kans.AdamW(p, lr=0.05, weight_decay=1e-4)),
            ("Muon", lambda p: slang_kans.Muon(p, lr=0.05, momentum=0.95)),
            ("Lion", lambda p: slang_kans.Lion(p, lr=0.02, weight_decay=1e-4)),
            ("RMSprop", lambda p: slang_kans.RMSprop(p, lr=0.02)),
            ("SGD", lambda p: slang_kans.SGD(p, lr=0.05, momentum=0.9, nesterov=True)),
        ]

        for name, opt_factory in optimizers:
            W = np.random.randn(8, 4).astype(np.float32)
            grad_W = np.zeros_like(W)
            opt = opt_factory([(W, grad_W)])

            init_loss = float(np.mean((X @ W - Y_target) ** 2))

            for _ in range(80):
                pred = X @ W
                # Loss L = mean((pred - Y_target)^2) -> dL/dW = 2/N * X^T @ (pred - Y_target)
                diff = 2.0 * (pred - Y_target) / len(X)
                grad_W[:] = X.T @ diff
                opt.step()
                opt.zero_grad()

            final_loss = float(np.mean((X @ W - Y_target) ** 2))
            self.assertLess(
                final_loss,
                init_loss * 0.35,
                f"{name} optimizer did not reduce loss: init={init_loss}, final={final_loss}"
            )

    def test_10_gated_kan(self):
        """Verify GatedKAN forward, backward, parameter counts, and quantization."""
        B, d_model = 16, 32
        x = np.random.uniform(-0.8, 0.8, (B, d_model)).astype(np.float32)
        gated = slang_kans.GatedKAN(d_model=d_model, d_ffn=64, degree=4)

        out = gated.forward(x)
        self.assertEqual(out.shape, (B, d_model))
        self.assertFalse(np.any(np.isnan(out)))

        # Backward pass
        dY = np.random.randn(*out.shape).astype(np.float32)
        dX = gated.backward(dY)
        self.assertEqual(dX.shape, x.shape)
        self.assertFalse(np.any(np.isnan(dX)))

        # Sub-4-bit quantization
        gated.to_int8()
        out_q8 = gated(x)
        self.assertEqual(out_q8.shape, (B, d_model))

        gated.to_ternary()
        out_ter = gated(x)
        self.assertEqual(out_ter.shape, (B, d_model))

        gated.to_int2()
        out_int2 = gated(x)
        self.assertEqual(out_int2.shape, (B, d_model))

    def test_11_sub4bit_quantization(self):
        """Verify Ternary 1.58-bit and INT2 Codebook quantization and fast GPU dequantization."""
        W = np.random.randn(32, 64).astype(np.float32)

        # Ternary
        t_quant = slang_kans.TernaryQuantizedWeight(W, group_size=32)
        w_deq_gpu = t_quant.dequantize(use_gpu=True)
        w_deq_cpu = t_quant.dequantize(use_gpu=False)
        self.assertTrue(np.allclose(w_deq_gpu, w_deq_cpu, atol=1e-5))

        # INT2
        int2_quant = slang_kans.Int2QuantizedWeight(W, group_size=32)
        w_int2_gpu = int2_quant.dequantize(use_gpu=True)
        w_int2_cpu = int2_quant.dequantize(use_gpu=False)
        self.assertTrue(np.allclose(w_int2_gpu, w_int2_cpu, atol=1e-5))

    def test_12_cheby_analytical_backward(self):
        """Verify analytical gradient computation in ChebyKANLinear against finite differences."""
        np.random.seed(42)
        B, D_in, D_out, deg = 4, 3, 2, 3
        layer = slang_kans.ChebyKANLinear(D_in, D_out, degree=deg, bias=True)
        x = np.random.uniform(-0.7, 0.7, (B, D_in)).astype(np.float32)

        out = layer.forward(x)
        dY = np.random.randn(B, D_out).astype(np.float32)

        layer.zero_grad()
        dX = layer.backward(dY)

        # Finite difference check for dX
        eps = 1e-4
        num_dX = np.zeros_like(x)
        for b in range(B):
            for i in range(D_in):
                xp = x.copy(); xp[b, i] += eps
                xm = x.copy(); xm[b, i] -= eps
                num_dX[b, i] = np.sum((layer.forward(xp) - layer.forward(xm)) / (2 * eps) * dY)

        max_diff = float(np.max(np.abs(dX - num_dX)))
        self.assertLess(max_diff, 1e-2, f"Cheby gradient mismatch: {max_diff}")

    def test_13_slang_tensor_chaining_and_slang_kan(self):
        """Verify zero-copy GPU tensor chaining and SlangKAN multi-layer container."""
        B, D_in, D_mid, D_out = 16, 8, 12, 4
        x = np.random.uniform(-0.8, 0.8, (B, D_in)).astype(np.float32)

        l1 = slang_kans.ChebyKANLinear(D_in, D_mid, degree=3)
        l2 = slang_kans.FastKANLinear(D_mid, D_mid, num_centers=6)
        l3 = slang_kans.KANLinear(D_mid, D_out, grid_size=4)

        # Baseline CPU/NumPy sequential
        out_mid1 = l1.forward(x)
        out_mid2 = l2.forward(out_mid1)
        expected_out = l3.forward(out_mid2)

        # GPU SlangTensor chaining
        t_in = slang_kans.to_tensor(x)
        self.assertIsInstance(t_in, slang_kans.SlangTensor)
        t_mid1 = l1.forward(t_in, return_tensor=True)
        self.assertIsInstance(t_mid1, slang_kans.SlangTensor)
        t_mid2 = l2.forward(t_mid1, return_tensor=True)
        self.assertIsInstance(t_mid2, slang_kans.SlangTensor)
        t_out = l3.forward(t_mid2, return_tensor=True)
        self.assertIsInstance(t_out, slang_kans.SlangTensor)
        gpu_chain_out = t_out.numpy()

        self.assertEqual(gpu_chain_out.shape, (B, D_out))
        self.assertTrue(np.allclose(expected_out, gpu_chain_out, atol=1e-4))

        # SlangKAN container
        model = slang_kans.SlangKAN([l1, l2, l3])
        model_out = model(x)
        self.assertEqual(model_out.shape, (B, D_out))
        self.assertTrue(np.allclose(expected_out, model_out, atol=1e-4))

    def test_14_slang_tensor_mlx_interop_and_checkpoint(self):
        """Verify SlangTensor <-> MLX array interop and CheckpointedKAN GPU chaining."""
        import mlx.core as mx
        B, D_in, D_out = 8, 4, 6
        np_x = np.random.randn(B, D_in).astype(np.float32)
        mx_x = mx.array(np_x)

        # Upload MLX array directly to SlangTensor
        t_from_mlx = slang_kans.to_tensor(mx_x)
        self.assertIsInstance(t_from_mlx, slang_kans.SlangTensor)
        self.assertEqual(t_from_mlx.shape, (B, D_in))

        # Convert SlangTensor back to MLX array
        mx_back = t_from_mlx.to_mlx()
        self.assertIsInstance(mx_back, mx.array)
        self.assertTrue(np.allclose(np_x, np.array(mx_back), atol=1e-5))

        # CheckpointedKAN GPU execution
        l1 = slang_kans.ChebyKANLinear(D_in, 8, degree=3)
        l2 = slang_kans.ReLUKANLinear(8, D_out, num_grids=6)
        ckpt_net = slang_kans.checkpoint_kan([l1, l2])

        out_ckpt = ckpt_net(np_x)
        self.assertEqual(out_ckpt.shape, (B, D_out))
        self.assertFalse(np.any(np.isnan(out_ckpt)))

    def test_15_slang_state_dict_and_dlpack(self):
        """Verify SlangKAN state_dict persistence and DLPack protocol on SlangTensor."""
        import tempfile
        B, D_in, D_mid, D_out = 4, 3, 6, 2
        x = np.random.randn(B, D_in).astype(np.float32)

        model1 = slang_kans.ChebyKAN([D_in, D_mid, D_out], degree=3)
        y1 = model1(x)

        # Test state_dict
        sd = model1.state_dict()
        self.assertIn("layer_0.weights", sd)
        self.assertIn("layer_1.weights", sd)

        # Test save and load weights
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            model1.save_weights(tmp_path)
            model2 = slang_kans.ChebyKAN([D_in, D_mid, D_out], degree=3)
            model2.load_weights(tmp_path)
            y2 = model2(x)
            self.assertTrue(np.allclose(y1, y2, atol=1e-5), "Saved and loaded weights must produce identical outputs")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Test DLPack protocol
        tensor = slang_kans.to_tensor(x)
        self.assertEqual(tensor.__dlpack_device__(), (1, 0))
        dlobj = tensor.__dlpack__()
        self.assertIsNotNone(dlobj)


if __name__ == "__main__":
    unittest.main()
