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
compare.py — Compare the baseline FP32 model with the QAT model.

Usage:
    python compare.py      (after running both training scripts)

Prints a side-by-side table showing:
    • Saved file size (KB)
    • Test accuracy (%)
    • Average inference time per batch (ms)
    • Quantization stats from the QAT model
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN, QATCNN

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


def load_qat() -> QATCNN:
    model = QATCNN()
    model.load_state_dict(
        torch.load(QAT_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    for path, label in [(BASELINE_PATH, "Baseline"), (QAT_PATH, "QAT")]:
        if not os.path.exists(path):
            print(f"❌  {label} model not found at '{path}'. Train it first.")
            return

    test_loader = get_test_loader()

    print("Loading models …")
    fp32_model = load_baseline()
    qat_model = load_qat()

    print("Evaluating accuracy …")
    fp32_acc = evaluate(fp32_model, test_loader)
    qat_acc = evaluate(qat_model, test_loader)

    print("Measuring inference latency …\n")
    fp32_latency = measure_latency(fp32_model, test_loader)
    qat_latency = measure_latency(qat_model, test_loader)

    fp32_size = file_size_kb(BASELINE_PATH)
    qat_size = file_size_kb(QAT_PATH)

    # ── Pretty-print comparison table ────────────────────────────────────
    header = f"{'Metric':<30} {'FP32 Baseline':>15} {'QAT Model':>15} {'Δ':>10}"
    sep = "─" * len(header)

    print("=" * len(header))
    print("  📊  Model Comparison: FP32 Baseline vs QAT (from-scratch)")
    print("=" * len(header))
    print(header)
    print(sep)
    print(
        f"{'Model file size (KB)':<30}"
        f" {fp32_size:>14.1f}"
        f" {qat_size:>14.1f}"
        f" {(qat_size / fp32_size) * 100:>9.1f}%"
    )
    print(
        f"{'Test accuracy (%)':<30}"
        f" {fp32_acc:>14.2f}"
        f" {qat_acc:>14.2f}"
        f" {qat_acc - fp32_acc:>+9.2f}"
    )
    print(
        f"{'Avg batch latency (ms)':<30}"
        f" {fp32_latency:>14.2f}"
        f" {qat_latency:>14.2f}"
        f" {((qat_latency - fp32_latency) / fp32_latency) * 100:>+8.1f}%"
    )
    print(sep)

    # ── Quantization stats from QAT model ────────────────────────────────
    print("\n📐  Quantization parameters learned during QAT:")
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

    print(
        f"\n📝  Note: QAT model file is larger because it stores FP32 weights"
        f"\n    + observer state. In production, weights would be converted"
        f"\n    to INT8 using the learned scale/zp → ~4× smaller."
    )
    print()


if __name__ == "__main__":
    main()
