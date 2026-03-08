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
train_bitnet.py — Training mô hình BitNet 1.58b (Ternary weights)

Script mô phỏng cách train một mô hình lượng tử cực độ (extreme quantization):
- Weights chỉ mang 3 giá trị: {-1, 0, 1}
- Activations bị lượng tử hoá xuống 8-bit (absmax)
- Không bắt đầu từ FP32 pre-trained, train TỪ ĐẦU (from scratch).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import BitNetCNN

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3
SAVE_PATH = "bitnet_model.pth"

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

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
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
    print(f"  BitNet Epoch {epoch}: loss={avg_loss:.4f}  train_acc={accuracy:.2f}%")

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

def main():
    print("=" * 70)
    print("  Alternative Schema: Training BitNet 1.58b (Ternary Weights)")
    print("=" * 70)

    train_loader, test_loader = get_dataloaders()
    model = BitNetCNN()
    
    # Đối với BitNet / QAT từ đầu, Weight Decay thường vô tác dụng hoặc gây hại 
    # với weights fp32 ẩn.
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch)

    acc = evaluate(model, test_loader)
    print(f"\n📊 BitNet 1.58b Test accuracy: {acc:.2f}%")
    
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"💾 Đã lưu BitNet model vào {SAVE_PATH}")
    
    # In thủ công thử weight phân phối của Conv1 để thấy nó thực sự là {-1, 0, 1} * gamma
    from model import weight_quant_bitnet
    w_fp32 = model.conv1.conv.weight
    w_ternary = weight_quant_bitnet(w_fp32)
    gamma = w_fp32.abs().mean().clamp(min=1e-8)
    w_int = torch.clamp(torch.round(w_fp32 / gamma), -1, 1)
    
    print("\n🧐 Cấu trúc weight thực sự ở layer conv1 (đã biến thành -1, 0, 1):")
    unique_vals = torch.unique(w_int)
    print(f"   Các giá trị discrete: {unique_vals.tolist()}")

if __name__ == "__main__":
    main()
