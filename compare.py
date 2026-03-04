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
compare.py — Compare the baseline FP32 model with the QAT INT8 model.

Usage:
    python compare.py      (after running both training scripts)

Prints a side-by-side table showing:
    • Saved file size (KB)
    • Test accuracy (%)
    • Average inference time per batch (ms)
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.ao.quantization as quant

from model import SimpleCNN, QuantizedCNN

# ── Constants ────────────────────────────────────────────────────────────────
BATCH_SIZE = 128
BASELINE_PATH = "baseline_model.pth"
QAT_PATH = "qat_model.pth"
NUM_WARMUP_BATCHES = 5
NUM_TIMED_BATCHES = 50


# ── Data ─────────────────────────────────────────────────────────────────────
def get_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    return DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# ── Helpers ──────────────────────────────────────────────────────────────────
def file_size_kb(path: str) -> float:
    return os.path.getsize(path) / 1024


@torch.no_grad()
def evaluate(model, loader) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def measure_latency(model, loader) -> float:
    """Return average inference time per batch in milliseconds."""
    model.eval()
    batches = iter(loader)

    # Warm-up
    for _ in range(NUM_WARMUP_BATCHES):
        images, _ = next(batches)
        _ = model(images)

    # Timed runs
    total_ms = 0.0
    for _ in range(NUM_TIMED_BATCHES):
        images, _ = next(batches)
        start = time.perf_counter()
        _ = model(images)
        total_ms += (time.perf_counter() - start) * 1000

    return total_ms / NUM_TIMED_BATCHES


# ── Load models ──────────────────────────────────────────────────────────────
def load_baseline() -> SimpleCNN:
    model = SimpleCNN()
    model.load_state_dict(torch.load(BASELINE_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_quantized() -> QuantizedCNN:
    model = QuantizedCNN()
    model.fuse_model()
    model.qconfig = quant.get_default_qat_qconfig("x86")
    quant.prepare_qat(model, inplace=True)
    model.eval()
    quantized_model = quant.convert(model)
    quantized_model.load_state_dict(
        torch.load(QAT_PATH, map_location="cpu", weights_only=True)
    )
    quantized_model.eval()
    return quantized_model


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    for path, label in [(BASELINE_PATH, "Baseline"), (QAT_PATH, "QAT")]:
        if not os.path.exists(path):
            print(f"❌  {label} model not found at '{path}'. Train it first.")
            return

    test_loader = get_test_loader()

    print("Loading models …")
    fp32_model = load_baseline()
    int8_model = load_quantized()

    print("Evaluating accuracy …")
    fp32_acc = evaluate(fp32_model, test_loader)
    int8_acc = evaluate(int8_model, test_loader)

    print("Measuring inference latency …\n")
    fp32_latency = measure_latency(fp32_model, test_loader)
    int8_latency = measure_latency(int8_model, test_loader)

    fp32_size = file_size_kb(BASELINE_PATH)
    int8_size = file_size_kb(QAT_PATH)

    # ── Pretty-print comparison table ────────────────────────────────────
    header = f"{'Metric':<30} {'FP32 Baseline':>15} {'QAT INT8':>15} {'Δ':>10}"
    sep = "─" * len(header)

    print("=" * len(header))
    print("  📊  Model Comparison: FP32 vs QAT INT8")
    print("=" * len(header))
    print(header)
    print(sep)
    print(
        f"{'Model file size (KB)':<30}"
        f" {fp32_size:>14.1f}"
        f" {int8_size:>14.1f}"
        f" {(int8_size / fp32_size) * 100:>9.1f}%"
    )
    print(
        f"{'Test accuracy (%)':<30}"
        f" {fp32_acc:>14.2f}"
        f" {int8_acc:>14.2f}"
        f" {int8_acc - fp32_acc:>+9.2f}"
    )
    print(
        f"{'Avg batch latency (ms)':<30}"
        f" {fp32_latency:>14.2f}"
        f" {int8_latency:>14.2f}"
        f" {((int8_latency - fp32_latency) / fp32_latency) * 100:>+8.1f}%"
    )
    print(sep)
    print(
        f"\n🗜️  Size reduction: {fp32_size / int8_size:.1f}×  smaller  "
        f"({fp32_size:.0f} KB → {int8_size:.0f} KB)"
    )
    print(f"🎯  Accuracy delta: {int8_acc - fp32_acc:+.2f}%")
    print()


if __name__ == "__main__":
    main()
