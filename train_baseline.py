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
train_baseline.py — Train a standard FP32 CNN on MNIST.

Usage:
    python train_baseline.py

What it does:
    1. Downloads MNIST (auto-cached in ./data/).
    2. Trains SimpleCNN for 5 epochs with Adam.
    3. Evaluates on the test set and prints accuracy.
    4. Saves the trained weights to  baseline_model.pth .
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN

# ── Hyper-parameters ─────────────────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "baseline_model.pth"


# ── Data ─────────────────────────────────────────────────────────────────────
def get_dataloaders():
    """Return MNIST train & test DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean & std
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ── Training loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

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
    print(f"  Epoch {epoch}: loss={avg_loss:.4f}  train_acc={accuracy:.2f}%")


# ── Evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Baseline FP32 Training — SimpleCNN on MNIST")
    print("=" * 60)
    print(f"Device : {DEVICE}")
    print(f"Epochs : {EPOCHS}")
    print(f"Batch  : {BATCH_SIZE}")
    print()

    train_loader, test_loader = get_dataloaders()

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch)

    test_acc = evaluate(model, test_loader)
    print(f"\n✅  Test accuracy: {test_acc:.2f}%")

    # Save weights (state_dict only — lighter & more portable)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"💾  Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
