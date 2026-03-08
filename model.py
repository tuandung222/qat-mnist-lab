# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
model.py — CNN architectures + from-scratch FakeQuantize for QAT learning.

This module implements:
  1. FakeQuantize       – A custom autograd function that simulates INT8
                          quantization in the forward pass while allowing
                          gradients to flow via Straight-Through Estimator.
  2. FakeQuantizeModule – nn.Module wrapper around FakeQuantize with learnable
                          or observed scale/zero_point.
  3. SimpleCNN          – Standard FP32 CNN for baseline training.
  4. QATConv2d / QATLinear – Drop-in replacements that apply fake quantization
                          to weights (and optionally activations).
  5. QATCNN             – The same CNN architecture using QAT layers.

NO torch.ao.quantization is used — everything is implemented from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Fake Quantize — the core building block
# ═══════════════════════════════════════════════════════════════════════════

class FakeQuantizeFunction(torch.autograd.Function):
    """
    Custom autograd function implementing fake quantization.

    ┌──────────────────────────────────────────────────────────────────┐
    │ PSEUDOCODE (MÃ GIẢ):                                           │
    │                                                                  │
    │  FORWARD(x, scale, zero_point, q_min, q_max):                   │
    │      x_int   = clamp(round(x / scale) + zero_point, q_min, q_max)│
    │      x_fake  = (x_int - zero_point) * scale                     │
    │      return x_fake   # FP32 tensor, nhưng chỉ chứa giá trị     │
    │                      # "có thể biểu diễn" bởi INT8              │
    │                                                                  │
    │  BACKWARD(grad_output):                                         │
    │      # Straight-Through Estimator:                              │
    │      # Chuyển gradient nguyên vẹn nếu x nằm trong vùng clamp   │
    │      mask = (x >= q_min_real) AND (x <= q_max_real)             │
    │      grad_input = grad_output * mask                            │
    │      return grad_input                                          │
    └──────────────────────────────────────────────────────────────────┘
    """

    @staticmethod
    def forward(ctx, x, scale, zero_point, q_min, q_max):
        # --- Quantize ---
        x_int = torch.clamp(torch.round(x / scale) + zero_point, q_min, q_max)

        # --- Dequantize ---
        x_fake_quantized = (x_int - zero_point) * scale

        # Save for backward: we need to know which values were clipped
        ctx.q_min_real = (q_min - zero_point) * scale
        ctx.q_max_real = (q_max - zero_point) * scale
        ctx.save_for_backward(x)

        return x_fake_quantized

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        q_min_real = ctx.q_min_real
        q_max_real = ctx.q_max_real

        # Straight-Through Estimator with gradient clipping:
        # Pass gradient through ONLY where x was within quantizable range.
        # Where x was clipped (saturated), gradient is zeroed out.
        mask = (x >= q_min_real) & (x <= q_max_real)
        grad_input = grad_output * mask.float()

        # No gradient for scale, zero_point, q_min, q_max
        return grad_input, None, None, None, None


def fake_quantize(x, scale, zero_point, q_min=0, q_max=255):
    """Functional interface for fake quantization."""
    return FakeQuantizeFunction.apply(x, scale, zero_point, q_min, q_max)


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: Observer — tracks min/max to compute scale & zero_point
# ═══════════════════════════════════════════════════════════════════════════

class MinMaxObserver(nn.Module):
    """
    Observes the running min/max of input tensors and computes
    quantization parameters (scale, zero_point).

    ┌──────────────────────────────────────────────────────────────────┐
    │ PSEUDOCODE (MÃ GIẢ):                                           │
    │                                                                  │
    │  OBSERVE(x):                                                    │
    │      running_min = min(running_min, x.min())                    │
    │      running_max = max(running_max, x.max())                    │
    │                                                                  │
    │  COMPUTE_PARAMS():                                              │
    │      scale      = (running_max - running_min) / (q_max - q_min) │
    │      zero_point = q_min - round(running_min / scale)            │
    │      zero_point = clamp(zero_point, q_min, q_max)               │
    │      return scale, zero_point                                   │
    └──────────────────────────────────────────────────────────────────┘

    Đây là observer đơn giản nhất (MinMax). Trong thực tế có thêm:
      - MovingAverageMinMaxObserver: dùng EMA, ít ảnh hưởng outlier
      - HistogramObserver: dùng KL-divergence để tìm clipping tối ưu
    """

    def __init__(self, q_min=0, q_max=255):
        super().__init__()
        self.q_min = q_min
        self.q_max = q_max
        self.register_buffer("running_min", torch.tensor(float("inf")))
        self.register_buffer("running_max", torch.tensor(float("-inf")))

    def forward(self, x: torch.Tensor):
        """Observe (cập nhật min/max) rồi trả lại x nguyên vẹn."""
        with torch.no_grad():
            self.running_min = torch.min(self.running_min, x.min())
            self.running_max = torch.max(self.running_max, x.max())
        return x

    def compute_qparams(self):
        """Tính scale và zero_point từ running min/max."""
        scale = (self.running_max - self.running_min) / (self.q_max - self.q_min)
        scale = torch.clamp(scale, min=1e-8)  # tránh chia cho 0
        zero_point = self.q_min - torch.round(self.running_min / scale)
        zero_point = torch.clamp(zero_point, self.q_min, self.q_max)
        return scale, zero_point


# ═══════════════════════════════════════════════════════════════════════════
# PART 3: FakeQuantize Module — kết hợp Observer + Fake Quantization
# ═══════════════════════════════════════════════════════════════════════════

class FakeQuantizeModule(nn.Module):
    """
    Module bọc quanh FakeQuantizeFunction + MinMaxObserver.

    Hoạt động:
      - Training mode:  observe min/max → tính scale/zp → fake quantize
      - Eval mode:      dùng scale/zp đã tính → fake quantize (không cập nhật)

    ┌──────────────────────────────────────────────────────────────────┐
    │ PSEUDOCODE (MÃ GIẢ):                                           │
    │                                                                  │
    │  FORWARD(x):                                                    │
    │    if training:                                                  │
    │        observer.observe(x)         # cập nhật running min/max   │
    │    scale, zero_point = observer.compute_qparams()               │
    │    return fake_quantize(x, scale, zero_point)                   │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, q_min=0, q_max=255):
        super().__init__()
        self.q_min = q_min
        self.q_max = q_max
        self.observer = MinMaxObserver(q_min, q_max)

    def forward(self, x):
        if self.training:
            self.observer(x)
        scale, zero_point = self.observer.compute_qparams()
        return fake_quantize(x, scale, zero_point, self.q_min, self.q_max)

    def extra_repr(self):
        return f"q_range=[{self.q_min}, {self.q_max}]"


# ═══════════════════════════════════════════════════════════════════════════
# PART 4: SimpleCNN — baseline FP32 (không có quantization)
# ═══════════════════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    """
    A minimal CNN for MNIST (28×28 grayscale, 10 classes).

    Architecture:
        conv1 (1→32, 3×3) → ReLU → MaxPool(2)
        conv2 (32→64, 3×3) → ReLU → MaxPool(2)
        flatten → fc1 (64*7*7 → 128) → ReLU → fc2 (128 → 10)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# PART 5: QAT Layers — Conv2d & Linear with built-in fake quantization
# ═══════════════════════════════════════════════════════════════════════════

class QATConv2d(nn.Module):
    """
    Conv2d layer with fake quantization on BOTH weights and activations.

    ┌──────────────────────────────────────────────────────────────────┐
    │ PSEUDOCODE (forward):                                           │
    │                                                                  │
    │  1. activation_fq = fake_quantize(input_activation)             │
    │     → Mô phỏng việc activation bị quantize khi đi qua layer    │
    │                                                                  │
    │  2. weight_fq = fake_quantize(self.weight)                      │
    │     → Mô phỏng việc weight bị quantize khi lưu trữ INT8        │
    │                                                                  │
    │  3. output = conv2d(activation_fq, weight_fq, bias)             │
    │     → Phép tích chập dùng giá trị đã fake-quantize             │
    │                                                                  │
    │  Trong backward:                                                │
    │     gradient chảy xuyên qua fake_quantize nhờ STE              │
    │     → weight được cập nhật bình thường bằng optimizer           │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

        # Separate fake-quant modules for weight and activation
        self.weight_fake_quant = FakeQuantizeModule(q_min=-128, q_max=127)  # signed INT8
        self.activation_fake_quant = FakeQuantizeModule(q_min=0, q_max=255)  # unsigned INT8

    def forward(self, x):
        # 1. Fake-quantize the INPUT ACTIVATION
        x_fq = self.activation_fake_quant(x)

        # 2. Fake-quantize the WEIGHT
        w_fq = self.weight_fake_quant(self.conv.weight)

        # 3. Convolution with fake-quantized values
        return F.conv2d(x_fq, w_fq, self.conv.bias,
                        self.conv.stride, self.conv.padding)


class QATLinear(nn.Module):
    """
    Linear layer with fake quantization on weights and activations.
    Same principle as QATConv2d.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        self.weight_fake_quant = FakeQuantizeModule(q_min=-128, q_max=127)
        self.activation_fake_quant = FakeQuantizeModule(q_min=0, q_max=255)

    def forward(self, x):
        x_fq = self.activation_fake_quant(x)
        w_fq = self.weight_fake_quant(self.linear.weight)
        return F.linear(x_fq, w_fq, self.linear.bias)


# ═══════════════════════════════════════════════════════════════════════════
# PART 6: QATCNN — CNN with from-scratch QAT
# ═══════════════════════════════════════════════════════════════════════════

class QATCNN(nn.Module):
    """
    Same architecture as SimpleCNN, but every Conv2d/Linear is replaced
    with its QAT variant that performs fake quantization from scratch.

    ┌──────────────────────────────────────────────────────────────────┐
    │  Luồng forward:                                                 │
    │                                                                  │
    │  input (FP32)                                                   │
    │    │                                                            │
    │    ▼                                                            │
    │  QATConv2d(1→32)  ← fake_quant(activation) + fake_quant(weight)│
    │    │                                                            │
    │    ▼                                                            │
    │  ReLU → MaxPool                                                 │
    │    │                                                            │
    │    ▼                                                            │
    │  QATConv2d(32→64) ← fake_quant(activation) + fake_quant(weight)│
    │    │                                                            │
    │    ▼                                                            │
    │  ReLU → MaxPool → Flatten                                       │
    │    │                                                            │
    │    ▼                                                            │
    │  QATLinear(3136→128) ← fake_quant(act) + fake_quant(weight)    │
    │    │                                                            │
    │    ▼                                                            │
    │  ReLU                                                           │
    │    │                                                            │
    │    ▼                                                            │
    │  QATLinear(128→10)   ← fake_quant(act) + fake_quant(weight)    │
    │    │                                                            │
    │    ▼                                                            │
    │  output (FP32, nhưng mang "dấu ấn" quantization noise)         │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()
        self.conv1 = QATConv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = QATConv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = QATLinear(64 * 7 * 7, 128)
        self.fc2 = QATLinear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def load_pretrained(self, state_dict):
        """
        Load weights from a SimpleCNN state_dict into the QAT model.
        Maps: conv1.weight → conv1.conv.weight, fc1.weight → fc1.linear.weight, etc.
        """
        mapping = {
            "conv1.weight": "conv1.conv.weight",
            "conv1.bias": "conv1.conv.bias",
            "conv2.weight": "conv2.conv.weight",
            "conv2.bias": "conv2.conv.bias",
            "fc1.weight": "fc1.linear.weight",
            "fc1.bias": "fc1.linear.bias",
            "fc2.weight": "fc2.linear.weight",
            "fc2.bias": "fc2.linear.bias",
        }
        new_state_dict = {}
        for old_key, new_key in mapping.items():
            if old_key in state_dict:
                new_state_dict[new_key] = state_dict[old_key]

        self.load_state_dict(new_state_dict, strict=False)

    def get_quantization_stats(self):
        """Print the learned scale/zero_point for each layer (for educational purposes)."""
        stats = []
        for name, module in self.named_modules():
            if isinstance(module, FakeQuantizeModule):
                scale, zp = module.observer.compute_qparams()
                stats.append({
                    "name": name,
                    "scale": scale.item(),
                    "zero_point": zp.item(),
                    "observed_min": module.observer.running_min.item(),
                    "observed_max": module.observer.running_max.item(),
                })
        return stats


# ═══════════════════════════════════════════════════════════════════════════
# PART 7: BitNet 1.58b — Alternative Schema (Ternary Weights {-1, 0, 1})
# ═══════════════════════════════════════════════════════════════════════════

def weight_quant_bitnet(w):
    """
    BitNet 1.58b Weight Quantization:
    W_q = clamp(round(W / gamma), -1, 1)
    where gamma = mean(|W|)

    Returns the fake-quantized weights (still FP32 but only contains {-gamma, 0, gamma})
    """
    # Tính scale bằng absolute mean
    gamma = w.abs().mean().clamp(min=1e-8)
    
    # Quantize về {-1, 0, 1}
    w_scaled = w / gamma
    w_int = torch.clamp(torch.round(w_scaled), -1, 1)
    
    # Dequantize về FP32 để gradient có thể scale đúng
    w_fake_quant = w_int * gamma
    
    # Dùng STE để gradient chảy thẳng qua đoạn làm tròn
    # (Ở BitNet gốc người ta hay viết: output = forward_val + input - input.detach())
    # Kỹ thuật này tương đương với STE autograd:
    w_out = w + (w_fake_quant - w).detach()
    return w_out


def activation_quant_bitnet(x, num_bits=8):
    """
    BitNet absmax quantization cho activation (thường dùng 8-bit).
    Trái với Weight (ternary), Activation thường giữ 8-bit để tránh mất quá nhiều thông tin.
    """
    q_max = (2 ** (num_bits - 1)) - 1  # 127
    gamma = x.abs().max().clamp(min=1e-8)
    scale = q_max / gamma
    
    x_int = torch.clamp(torch.round(x * scale), -q_max, q_max)
    x_fake_quant = x_int / scale
    
    # STE
    x_out = x + (x_fake_quant - x).detach()
    return x_out


class BitLinear(nn.Module):
    """
    Linear layer sử dụng schema của BitNet 1.58b:
    - Weights: Ternary {-1, 0, 1}
    - Activations: 8-bit
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Bỏ qua LayerNorm để code gọn, trong paper thực tế có RMSNorm trước khi lượng tử hoá

    def forward(self, x):
        x_quant = activation_quant_bitnet(x)
        w_quant = weight_quant_bitnet(self.linear.weight)
        return F.linear(x_quant, w_quant, self.linear.bias)


class BitConv2d(nn.Module):
    """
    Conv2d với BitNet schema (dùng cho CNN demo).
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        x_quant = activation_quant_bitnet(x)
        w_quant = weight_quant_bitnet(self.conv.weight)
        return F.conv2d(x_quant, w_quant, self.conv.bias, 
                        self.conv.stride, self.conv.padding)


class BitNetCNN(nn.Module):
    """
    CNN architecture built entirely with 1.58-bit models (BitConv2d, BitLinear).
    """
    def __init__(self):
        super().__init__()
        self.conv1 = BitConv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = BitConv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = BitLinear(64 * 7 * 7, 128)
        self.fc2 = BitLinear(128, 10)

    def forward(self, x):
        # Thông thường layer đầu tiên ít khi lượng tử trọng số khắc nghiệt, 
        # nhưng trong demo ta lượng tử mọi thứ.
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
