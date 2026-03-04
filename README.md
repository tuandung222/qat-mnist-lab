# 🧪 QAT MNIST Lab — Quantization-Aware Training From Scratch

A **self-contained toy project** for learning **Quantization-Aware Training (QAT)** — implemented **from scratch** using pure PyTorch autograd. **No `torch.ao.quantization` is used.**

You'll train a simple CNN on MNIST, then apply QAT with hand-written fake quantization to understand exactly how it works under the hood.

---

## 🎯 What you'll learn

- What **fake quantization** is and why it's needed
- How **Straight-Through Estimator (STE)** allows gradients to flow through rounding
- How **activations** and **weights** are treated differently during QAT
- The relationship between **PTQ** (Post-Training Quantization) and **QAT**
- How **observers** track tensor statistics to compute scale/zero_point

---

## 📂 Project Structure

```
qat-mnist-lab/
├── README.md                  ← You are here
├── requirements.txt           ← PyTorch dependencies
├── model.py                   ← From-scratch: FakeQuantize, Observer, QAT layers, CNN
├── train_baseline.py          ← Step 1: Train a standard FP32 model
├── train_qat.py               ← Step 2: Fine-tune with from-scratch QAT
├── compare.py                 ← Step 3: Compare models side-by-side
└── docs/
    └── qat_explained.md       ← 📘 Detailed educational guide (Vietnamese)
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Step 1: Train FP32 baseline (~99% accuracy)
python train_baseline.py

# Step 2: QAT fine-tuning with from-scratch fake quantization
python train_qat.py

# Step 3: Compare baseline vs QAT model
python compare.py
```

---

## 🧱 Key Components (all in `model.py`)

| Component | What it does |
|-----------|-------------|
| `FakeQuantizeFunction` | Custom `torch.autograd.Function`: quantize→dequantize in forward, STE in backward |
| `MinMaxObserver` | Tracks running min/max of tensors to compute scale & zero_point |
| `FakeQuantizeModule` | `nn.Module` wrapping Observer + FakeQuantize for easy use |
| `QATConv2d` / `QATLinear` | Drop-in layer replacements with fake-quant on both **activation** and **weight** |
| `QATCNN` | Same CNN architecture as `SimpleCNN`, using QAT layers |

---

## 📘 Educational Document

For a **deep-dive explanation** (with pseudocode, diagrams, and math), see:

👉 **[`docs/qat_explained.md`](docs/qat_explained.md)**

Covers:
- Fake quantize mechanism (pseudocode + visualization)
- PTQ ↔ QAT relationship
- How activations & weights flow through forward/backward
- Straight-Through Estimator (STE)
- Observer internals
- FAQ

---

## 📚 References

- [Quantization and Training of Neural Networks (Jacob et al., 2018)](https://arxiv.org/abs/1712.05877)
- [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html)
- [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
