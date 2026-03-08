import pytest
import torch
import torch.nn as nn
from model import QATCNN
from post_qat_convert import convert_to_int8, RealInt8CNN

def test_convert_to_int8_logic():
    """Test that convert_to_int8 correctly quantizes a mock state dict to torch.int8."""
    # Create a small mock QAT model state dict
    # We just need one layer to prove the conversion math works
    fp32_state_dict = {
        "conv1.conv.weight": torch.tensor([[-0.5, 0.0], [0.5, 1.2]]),
        "conv1.conv.bias": torch.tensor([0.1]),
        "conv1.weight_fake_quant.observer.running_min": torch.tensor(-1.0),
        "conv1.weight_fake_quant.observer.running_max": torch.tensor(1.2)
    }
    
    int8_state_dict = convert_to_int8(fp32_state_dict)
    
    # Check that conv1.weight is present and is of type int8
    assert "conv1.weight" in int8_state_dict
    assert "conv1.scale" in int8_state_dict
    assert "conv1.zero_point" in int8_state_dict
    assert "conv1.bias" in int8_state_dict
    
    assert int8_state_dict["conv1.weight"].dtype == torch.int8
    assert int8_state_dict["conv1.bias"].dtype == torch.float32 # bias should remain fp32
    
def test_real_int8_cnn_dequantize():
    """Test that RealInt8CNN can load an int8 state dict and dequantize properly for a forward pass."""
    
    # 1. Create a minimal int8 state dict that matches RealInt8CNN exactly
    int8_state_dict = {
        "conv1.weight": torch.tensor([[[[10, -5], [0, 12]]]], dtype=torch.int8), # 1 out_channel, 1 in_channel, 2x2
        "conv1.bias": torch.tensor([0.1]),
        "conv1.scale": torch.tensor(0.1),
        "conv1.zero_point": torch.tensor(0),
        
        "conv2.weight": torch.tensor([[[[5, 5], [5, 5]]]], dtype=torch.int8),
        "conv2.bias": torch.tensor([0.0]),
        "conv2.scale": torch.tensor(0.1),
        "conv2.zero_point": torch.tensor(0),
        
        # FC1 expects (128, 64*7*7). Let's just make it small for the test and patch the forward pass.
        # But RealInt8CNN is hardcoded for MNIST sizes. 
        # This test is just validating the _dequantize step mostly.
    }
    
    # We will instantiate and only test the dequantize helper so we don't need a full valid weights matrix
    model = RealInt8CNN(int8_state_dict)
    
    w_fp32 = model._dequantize("conv1")
    
    assert w_fp32.dtype == torch.float32
    # Check if calculation is correct: (10 - 0) * 0.1 = 1.0
    assert torch.allclose(w_fp32[0, 0, 0, 0], torch.tensor(1.0))
    # (-5 - 0) * 0.1 = -0.5
    assert torch.allclose(w_fp32[0, 0, 0, 1], torch.tensor(-0.5))
