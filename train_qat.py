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
train_qat.py — Quantization-Aware Training (QAT) on MNIST.

Usage:
    python train_qat.py          (after running train_baseline.py first)

What it does:
    1. Creates a QuantizedCNN and loads the pre-trained FP32 weights.
    2. Fuses Conv+ReLU / Linear+ReLU layers.
    3. Attaches a QAT-specific qconfig (fake-quantized weights & activations).
    4. Calls prepare_qat() to insert fake-quant observers.
    5. Fine-tunes for 3 epochs so the model learns to compensate for
       quantization noise.
    6. Calls convert() to produce a real INT8 model.
    7. Evaluates and saves the quantized model to  qat_model.pth .

Key concepts illustrated:
    • QuantStub / DeQuantStub
    • fuse_modules()
    • prepare_qat()  vs  prepare() (PTQ)
    • convert()
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.ao.quantization as quant

from model import SimpleCNN, QuantizedCNN

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


# ── Training loop (same as baseline) ─────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # QAT runs on CPU only (PyTorch eager-mode quantization limitation)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
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
    print("=" * 60)
    print("  Quantization-Aware Training (QAT) — QuantizedCNN on MNIST")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load the pre-trained FP32 weights into QuantizedCNN
    # ------------------------------------------------------------------
    if not os.path.exists(BASELINE_PATH):
        print(f"❌  Baseline model not found at '{BASELINE_PATH}'.")
        print("   Run  python train_baseline.py  first.")
        return

    model = QuantizedCNN()

    # Load state_dict from SimpleCNN — keys match (same layer names).
    state_dict = torch.load(BASELINE_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)  # strict=False skips quant/dequant
    print("✅  Loaded pre-trained baseline weights.")

    # ------------------------------------------------------------------
    # Step 2: Fuse layers
    #   Conv2d + ReLU  →  ConvReLU2d   (single fused module)
    #   Linear + ReLU  →  LinearReLU   (single fused module)
    # This is REQUIRED before prepare_qat so that observers see the
    # fused operation instead of separate ones.
    # ------------------------------------------------------------------
    model.fuse_model()
    print("🔗  Fused Conv+ReLU and Linear+ReLU layers.")

    # ------------------------------------------------------------------
    # Step 3: Attach QAT qconfig
    #   qconfig specifies HOW weights and activations are fake-quantized
    #   during training.  get_default_qat_qconfig("x86") is a good
    #   default for desktop CPUs.
    # ------------------------------------------------------------------
    model.qconfig = quant.get_default_qat_qconfig("x86")
    print(f"📐  QAT qconfig: {model.qconfig}")

    # ------------------------------------------------------------------
    # Step 4: prepare_qat()  — inserts fake-quant modules
    #   After this call the model's forward pass will simulate INT8
    #   arithmetic while still using FP32 under the hood so that
    #   gradients can flow and the model can adapt.
    # ------------------------------------------------------------------
    quant.prepare_qat(model, inplace=True)
    print("🔧  Model prepared for QAT (fake-quant observers inserted).\n")

    # ------------------------------------------------------------------
    # Step 5: Fine-tune with fake quantization
    # ------------------------------------------------------------------
    train_loader, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training for {QAT_EPOCHS} QAT epochs …")
    for epoch in range(1, QAT_EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch)

    # Evaluate while still in fake-quant mode
    fake_quant_acc = evaluate(model, test_loader)
    print(f"\n📊  Fake-quant test accuracy: {fake_quant_acc:.2f}%")

    # ------------------------------------------------------------------
    # Step 6: convert()  — replace fake-quant ops with real INT8 ops
    #   After conversion the model is a true quantized model that runs
    #   with integer arithmetic on supported backends.
    # ------------------------------------------------------------------
    model.eval()  # must be in eval mode before convert
    quantized_model = quant.convert(model)
    print("⚡  Model converted to INT8 (real quantized).")

    # Evaluate the converted INT8 model
    int8_acc = evaluate(quantized_model, test_loader)
    print(f"📊  INT8 test accuracy: {int8_acc:.2f}%")

    # ------------------------------------------------------------------
    # Step 7: Save the quantized model
    # ------------------------------------------------------------------
    torch.save(quantized_model.state_dict(), SAVE_PATH)
    print(f"💾  Quantized model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
