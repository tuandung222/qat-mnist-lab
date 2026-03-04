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
train_qat.py — Quantization-Aware Training FROM SCRATCH (no torch.ao.quantization).

Usage:
    python train_qat.py          (after running train_baseline.py first)

Pipeline:
    1. Load pre-trained SimpleCNN weights into QATCNN.
    2. Fine-tune for a few epochs — the FakeQuantize modules inside each
       QATConv2d/QATLinear simulate INT8 rounding during forward,
       while gradients flow through via STE during backward.
    3. After training, export the quantization parameters (scale, zero_point)
       and save the model.

Everything is implemented from scratch in model.py — no torch.ao is used.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN, QATCNN

# ── Hyper-parameters ─────────────────────────────────────────────────────────
BATCH_SIZE = 128
QAT_EPOCHS = 3          # QAT fine-tuning needs fewer epochs
LEARNING_RATE = 1e-4     # Lower LR for fine-tuning
BASELINE_PATH = "baseline_model.pth"
SAVE_PATH = "qat_model.pth"


# ── Data (same transforms as baseline) ───────────────────────────────────────
def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ── Training loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    """
    ┌──────────────────────────────────────────────────────────────────┐
    │ PSEUDOCODE — QAT Training Step:                                 │
    │                                                                  │
    │  for each batch (images, labels):                               │
    │      # FORWARD PASS:                                            │
    │      #   Mỗi QATConv2d/QATLinear tự động:                       │
    │      #   1. Observer cập nhật running_min/max của activation     │
    │      #   2. Fake-quantize activation (round → clamp → scale)    │
    │      #   3. Observer cập nhật running_min/max của weight         │
    │      #   4. Fake-quantize weight                                │
    │      #   5. Tính conv2d/linear với giá trị đã fake-quantize     │
    │      #                                                          │
    │      outputs = model(images)                                    │
    │      loss = cross_entropy(outputs, labels)                      │
    │                                                                  │
    │      # BACKWARD PASS:                                           │
    │      #   Gradient chảy ngược qua mỗi layer:                    │
    │      #   1. Qua fake_quantize → STE: gradient × mask            │
    │      #      (mask=1 nếu giá trị KHÔNG bị clamp,                │
    │      #       mask=0 nếu giá trị BỊ clamp/saturate)             │
    │      #   2. Qua conv2d/linear → gradient bình thường            │
    │      #   3. Weight được cập nhật: w = w - lr * grad            │
    │      #   (Weight cập nhật ở FP32, nhưng forward pass tiếp theo  │
    │      #    sẽ lại fake-quantize weight → mô hình học cách        │
    │      #    tối ưu weight sao cho KHI BỊ quantize vẫn tốt)       │
    │      #                                                          │
    │      loss.backward()                                            │
    │      optimizer.step()                                           │
    └──────────────────────────────────────────────────────────────────┘
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        optimizer.zero_grad()

        # Forward: fake quantization happens inside each QAT layer
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward: STE allows gradients to flow through fake_quantize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    print(f"  QAT Epoch {epoch}: loss={avg_loss:.4f}  train_acc={accuracy:.2f}%")


# ── Evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        outputs = model(images)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Quantization-Aware Training — FROM SCRATCH (no torch.ao)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load pre-trained FP32 weights into QAT model
    # ------------------------------------------------------------------
    if not os.path.exists(BASELINE_PATH):
        print(f"❌  Baseline model not found at '{BASELINE_PATH}'.")
        print("   Run  python train_baseline.py  first.")
        return

    qat_model = QATCNN()

    # Load SimpleCNN weights → map to QATCNN naming convention
    baseline_state = torch.load(BASELINE_PATH, map_location="cpu", weights_only=True)
    qat_model.load_pretrained(baseline_state)
    print("✅  Loaded pre-trained baseline weights into QATCNN.")
    print()

    # ------------------------------------------------------------------
    # Step 2: Print model structure (see the FakeQuantize modules)
    # ------------------------------------------------------------------
    print("📐  Model structure (note the FakeQuantize inside each layer):")
    print("-" * 60)
    for name, module in qat_model.named_modules():
        if name:  # skip root module
            indent = "  " * name.count(".")
            print(f"  {indent}{name}: {module.__class__.__name__}")
    print("-" * 60)
    print()

    # ------------------------------------------------------------------
    # Step 3: Fine-tune with QAT
    # ------------------------------------------------------------------
    train_loader, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(qat_model.parameters(), lr=LEARNING_RATE)

    # Evaluate BEFORE QAT fine-tuning (with random observer states)
    baseline_acc = evaluate(qat_model, test_loader)
    print(f"📊  Accuracy before QAT fine-tuning: {baseline_acc:.2f}%")
    print(f"    (Lower than baseline because observers haven't calibrated yet)")
    print()

    print(f"🏋️  Fine-tuning for {QAT_EPOCHS} QAT epochs …")
    for epoch in range(1, QAT_EPOCHS + 1):
        train_one_epoch(qat_model, train_loader, criterion, optimizer, epoch)

    # ------------------------------------------------------------------
    # Step 4: Evaluate after QAT
    # ------------------------------------------------------------------
    qat_acc = evaluate(qat_model, test_loader)
    print(f"\n📊  QAT test accuracy: {qat_acc:.2f}%")

    # ------------------------------------------------------------------
    # Step 5: Show learned quantization parameters
    # ------------------------------------------------------------------
    print("\n📐  Learned quantization parameters per layer:")
    print("-" * 70)
    print(f"  {'Layer':<40} {'Scale':>10} {'ZP':>8} {'Min':>10} {'Max':>10}")
    print("-" * 70)
    for stat in qat_model.get_quantization_stats():
        print(
            f"  {stat['name']:<40}"
            f" {stat['scale']:>10.6f}"
            f" {stat['zero_point']:>8.1f}"
            f" {stat['observed_min']:>10.4f}"
            f" {stat['observed_max']:>10.4f}"
        )
    print("-" * 70)

    # ------------------------------------------------------------------
    # Step 6: Save model
    # ------------------------------------------------------------------
    torch.save(qat_model.state_dict(), SAVE_PATH)
    print(f"\n💾  QAT model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
