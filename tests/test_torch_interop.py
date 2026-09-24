import pytest
import numpy as np
import torch
import torch.nn as nn

from slang_kans import TorchSlangKANLinear, TorchSlangKAN, is_slang_available


def test_torch_slang_kan_linear_cheby():
    layer = TorchSlangKANLinear(in_features=4, out_features=3, basis="cheby", degree=4, bias=True)
    x = torch.randn(8, 4, requires_grad=True)

    y = layer(x)
    assert y.shape == (8, 3)
    assert y.dtype == torch.float32

    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (8, 4)
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape
    if layer.bias_param is not None:
        assert layer.bias_param.grad is not None
        assert layer.bias_param.grad.shape == layer.bias_param.shape


def test_torch_slang_kan_linear_fastkan():
    layer = TorchSlangKANLinear(in_features=4, out_features=2, basis="fastkan", degree=5, bias=True)
    x = torch.randn(6, 4, requires_grad=True)

    y = layer(x)
    assert y.shape == (6, 2)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (6, 4)
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape


def test_torch_slang_kan_linear_relukan():
    layer = TorchSlangKANLinear(in_features=3, out_features=3, basis="relu", degree=4, bias=True)
    x = torch.randn(5, 3, requires_grad=True)

    y = layer(x)
    assert y.shape == (5, 3)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert layer.weight.grad is not None


def test_torch_slang_kan_optimization_step():
    """Verify PyTorch optimizer can update weights of TorchSlangKAN and decrease loss."""
    torch.manual_seed(42)
    model = TorchSlangKAN([4, 8, 2], basis="cheby", degree=3, bias=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    x = torch.randn(16, 4)
    target = torch.randn(16, 2)

    initial_loss = criterion(model(x), target).item()

    for _ in range(5):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

    final_loss = criterion(model(x), target).item()
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
