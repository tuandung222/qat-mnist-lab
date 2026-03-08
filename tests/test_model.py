import pytest
import torch
import torch.nn as nn
from model import (
    fake_quantize, 
    MinMaxObserver, 
    FakeQuantizeModule, 
    QATConv2d, 
    QATLinear, 
    BitLinear, 
    BitConv2d,
    weight_quant_bitnet,
    activation_quant_bitnet
)

def test_fake_quantize_math():
    """Test fake quantization math manually without autograd."""
    x = torch.tensor([-0.5, 0.0, 0.73, 1.2])
    scale = 0.00667
    zp = 75
    # For q_min=0, q_max=255
    # round(-0.5/0.00667) + 75 = round(-74.96) + 75 = -75 + 75 = 0
    # ...
    out = fake_quantize(x, scale=scale, zero_point=zp, q_min=0, q_max=255)
    
    assert out.shape == x.shape
    assert out.dtype == torch.float32

    # Verify clamping behavior
    x_high = torch.tensor([5.0]) # Should clamp
    out_high = fake_quantize(x_high, scale=scale, zero_point=zp, q_min=0, q_max=255)
    assert out_high.item() <= (255 - zp) * scale + 1e-4

def test_min_max_observer():
    """Test observer updates min/max and calculates correct scale/zp."""
    observer = MinMaxObserver(q_min=0, q_max=255)
    
    # Observe batch 1
    x1 = torch.tensor([-2.0, 0.0, 3.0])
    observer(x1)
    assert torch.allclose(observer.running_min, torch.tensor(-2.0))
    assert torch.allclose(observer.running_max, torch.tensor(3.0))
    
    # Observe batch 2 (new max, min stays same)
    x2 = torch.tensor([-1.0, 4.0])
    observer(x2)
    assert torch.allclose(observer.running_min, torch.tensor(-2.0))
    assert torch.allclose(observer.running_max, torch.tensor(4.0))

    scale, zp = observer.compute_qparams()
    # Range = 4 - (-2) = 6.0
    # scale = 6.0 / 255
    expected_scale = 6.0 / 255.0
    expected_zp = float(0 - round(-2.0 / expected_scale))
    
    assert torch.allclose(scale, torch.tensor(expected_scale, dtype=torch.float32))
    assert torch.allclose(zp, torch.tensor(expected_zp, dtype=torch.float32))

def test_fake_quant_module():
    """Test FakeQuantizeModule in training and eval mode."""
    module = FakeQuantizeModule(q_min=-128, q_max=127)
    
    module.train()
    x = torch.tensor([-10.0, 5.0, 10.0])
    out1 = module(x)
    
    # After first pass: min=-10, max=10
    scale1, zp1 = module.observer.compute_qparams()
    assert scale1.item() > 0

    module.eval()
    x_new = torch.tensor([-20.0, 20.0]) # Exceeds range
    out2 = module(x_new)
    
    # Range should NOT change in eval mode
    scale2, zp2 = module.observer.compute_qparams()
    assert scale1 == scale2
    assert zp1 == zp2

def test_qat_layers_forward():
    """Test forward pass of QAT Conv2d and Linear layers."""
    # Conv2d
    conv = QATConv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
    x = torch.randn(2, 1, 28, 28) # Batch size 2, 1 channel, 28x28
    out = conv(x)
    assert out.shape == (2, 4, 28, 28)
    
    # Linear
    linear = QATLinear(in_features=64, out_features=10)
    x_lin = torch.randn(2, 64)
    out_lin = linear(x_lin)
    assert out_lin.shape == (2, 10)

def test_ste_backward():
    """Test Straight-Through Estimator passes gradient through."""
    x = torch.tensor([0.5, 1.2], requires_grad=True)
    scale = 0.1
    zp = 0
    # range for valid grads is clamped to q_min, q_max
    out = fake_quantize(x, scale, zp, q_min=-128, q_max=127)
    loss = out.sum()
    loss.backward()
    
    # Grad should be 1.0 because values are within range
    assert torch.allclose(x.grad, torch.tensor([1.0, 1.0]))

def test_bitnet_quantization():
    """Test BitNet ternary weight quantization."""
    w = torch.tensor([-2.5, -0.5, 0.0, 1.0, 3.0])
    w_quant = weight_quant_bitnet(w)
    
    gamma = w.abs().mean().clamp(min=1e-8)
    
    # Unique values should only be: {-gamma, 0, gamma}
    unique_vals = torch.unique(w_quant)
    for val in unique_vals:
        assert torch.isclose(val, -gamma) or torch.isclose(val, torch.tensor(0.0)) or torch.isclose(val, gamma)

def test_bitnet_layers_forward():
    """Test forward passes of BitNet layers."""
    lin = BitLinear(in_features=16, out_features=8)
    x = torch.randn(4, 16)
    out = lin(x)
    assert out.shape == (4, 8)
    
    conv = BitConv2d(1, 4, kernel_size=3, padding=1)
    x_conv = torch.randn(2, 1, 14, 14)
    out_conv = conv(x_conv)
    assert out_conv.shape == (2, 4, 14, 14)
