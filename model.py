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
model.py — CNN architectures for MNIST classification.

This module defines two variants of a simple CNN:
  1. SimpleCNN      – A standard FP32 model for baseline training.
  2. QuantizedCNN   – The same architecture wrapped with QuantStub/DeQuantStub
                      and fuse_modules() support for Quantization-Aware Training.
"""

import torch
import torch.nn as nn
import torch.ao.quantization as quant


# ---------------------------------------------------------------------------
# 1. Baseline CNN  (FP32)
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """
    A minimal CNN for MNIST (28×28 grayscale images, 10 classes).

    Architecture:
        conv1 (1→32, 3×3) → ReLU → MaxPool(2)
        conv2 (32→64, 3×3) → ReLU → MaxPool(2)
        flatten → fc1 (64*5*5 → 128) → ReLU → fc2 (128 → 10)
    """

    def __init__(self):
        super().__init__()
        # --- Feature extractor ---
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # --- Classifier ---
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# 2. Quantization-Aware CNN  (for QAT)
# ---------------------------------------------------------------------------
class QuantizedCNN(nn.Module):
    """
    Same architecture as SimpleCNN, but augmented for QAT:

    • QuantStub   at the input  – converts FP32 tensors → quantized tensors
    • DeQuantStub at the output – converts quantized tensors → FP32
    • fuse_modules() merges (Conv2d + ReLU) and (Linear + ReLU) pairs so that
      quantization observers can track their joint distributions.

    Typical workflow:
        1. Instantiate QuantizedCNN and load pre-trained FP32 weights.
        2. Call model.fuse_model() to fuse eligible layers.
        3. Set model.qconfig and call torch.ao.quantization.prepare_qat(model).
        4. Fine-tune for a few epochs (fake-quantized forward pass).
        5. Call torch.ao.quantization.convert(model) to get a real INT8 model.
    """

    def __init__(self):
        super().__init__()
        # Quant / DeQuant stubs
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

        # --- Feature extractor (same as SimpleCNN) ---
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # --- Classifier ---
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)                              # FP32 → fake-quant
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)                             # fake-quant → FP32
        return x

    def fuse_model(self):
        """Fuse Conv+ReLU and Linear+ReLU pairs in-place for quantization."""
        torch.ao.quantization.fuse_modules(
            self,
            [["conv1", "relu1"], ["conv2", "relu2"], ["fc1", "relu3"]],
            inplace=True,
        )
