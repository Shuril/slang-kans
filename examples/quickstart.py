"""
Quickstart example for slang-KANs.
Demonstrates layer creation, GPU forward pass, quantization, and symbolic export.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import slang_kans


def main():
    dev = slang_kans.get_device()
    print(f"Active Slang device: {dev.device_type}")

    # 1. Instantiate a KAN model
    model = slang_kans.FastKAN(layers_hidden=[2, 16, 1], num_centers=8)
    params = slang_kans.count_parameters(model)
    print(f"FastKAN parameters: {params['total']}")

    # 2. Forward pass on GPU
    x = np.random.uniform(-1.0, 1.0, (8, 2)).astype(np.float32)
    y = model(x)
    print(f"Input shape: {x.shape} -> Output shape: {y.shape}")

    # 3. Quantize to INT8
    q_model = slang_kans.to_int8(model)
    fp32_sz = slang_kans.get_model_size(model)["size_bytes"]
    int8_sz = slang_kans.get_model_size(q_model)["size_bytes"]
    print(f"Size: FP32 = {fp32_sz} bytes -> INT8 = {int8_sz} bytes ({fp32_sz / int8_sz:.2f}x compression)")

    # 4. Symbolic mathematical extraction
    cheby = slang_kans.ChebyKANLinear(2, 1, degree=3)
    sym = slang_kans.to_symbolic(cheby)
    print(f"Symbolic formula: {sym.formula()}")


if __name__ == "__main__":
    main()
