# 🧪 QAT MNIST Lab — Quantization-Aware Training with a Simple CNN

A **self-contained toy project** for learning **Quantization-Aware Training (QAT)** using PyTorch. You'll train a simple CNN on MNIST, then apply QAT to produce a smaller INT8 model with comparable accuracy.

---

## 📖 What is Quantization?

**Quantization** reduces a neural network's numerical precision (e.g., from 32-bit floats to 8-bit integers). This yields:

| Benefit | Why it matters |
|---------|---------------|
| **Smaller model** | 2–4× reduction in file size |
| **Faster inference** | INT8 math is cheaper on CPUs/edge devices |
| **Lower power** | Ideal for mobile & embedded deployment |

### PTQ vs QAT

| | Post-Training Quantization (PTQ) | Quantization-Aware Training (QAT) |
|---|---|---|
| **When** | After training is complete | During training (fine-tuning) |
| **How** | Calibrate with a small dataset | Simulate quantization in the forward pass |
| **Quality** | Can degrade accuracy on sensitive models | Model adapts to quantization noise → better accuracy |
| **Cost** | Cheap (no training) | Moderate (a few extra epochs) |

> **This project focuses on QAT**, which is the recommended approach when PTQ causes unacceptable accuracy loss.

---

## 🏗️ Project Structure

```
qat-mnist-lab/
├── README.md            ← You are here
├── requirements.txt     ← PyTorch dependencies
├── model.py             ← SimpleCNN (FP32) + QuantizedCNN (QAT-ready)
├── train_baseline.py    ← Step 1: Train a standard FP32 model
├── train_qat.py         ← Step 2: Fine-tune with QAT, convert to INT8
└── compare.py           ← Step 3: Compare size, accuracy, speed
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the FP32 baseline

```bash
python train_baseline.py
```

This trains `SimpleCNN` for 5 epochs and saves `baseline_model.pth`.
Expected output: **~98–99% test accuracy**.

### 3. Run QAT fine-tuning

```bash
python train_qat.py
```

This:
1. Loads the baseline weights into `QuantizedCNN`
2. Fuses `Conv+ReLU` and `Linear+ReLU` layers
3. Inserts **fake-quantization** observers
4. Fine-tunes for 3 epochs
5. Converts to a **real INT8** model
6. Saves `qat_model.pth`

### 4. Compare the models

```bash
python compare.py
```

Prints a comparison table: file size, accuracy, and inference latency.

---

## 🔑 Key QAT Concepts (Code Walkthrough)

### 1. `QuantStub` / `DeQuantStub` (in `model.py`)

```python
self.quant = quant.QuantStub()      # marks input for quantization
self.dequant = quant.DeQuantStub()  # converts back to FP32 at output
```

These tell PyTorch **where the quantized region begins and ends** in the forward pass.

### 2. `fuse_modules()` (in `model.py`)

```python
torch.ao.quantization.fuse_modules(
    self,
    [["conv1", "relu1"], ["conv2", "relu2"], ["fc1", "relu3"]],
    inplace=True,
)
```

Fusing merges adjacent operations (e.g., `Conv2d + ReLU`) into a single module. This is **required** before QAT so that the quantization observers see the fused operation's output range, not two separate ones.

### 3. `prepare_qat()` (in `train_qat.py`)

```python
model.qconfig = quant.get_default_qat_qconfig("x86")
quant.prepare_qat(model, inplace=True)
```

- `qconfig` specifies *how* to fake-quantize (activation observer, weight observer, bit-width, etc.)
- `prepare_qat()` inserts **FakeQuantize** modules that simulate INT8 rounding during training while keeping FP32 gradients flowing for backpropagation.

### 4. `convert()` (in `train_qat.py`)

```python
model.eval()
quantized_model = quant.convert(model)
```

After fine-tuning, `convert()` replaces the fake-quant modules with **real quantized operators** (e.g., `torch.nn.quantized.Conv2d`). The resulting model performs actual INT8 arithmetic.

---

## 📊 Expected Results

| Metric | FP32 Baseline | QAT INT8 |
|--------|:---:|:---:|
| Test accuracy | ~99% | ~99% |
| Model file size | ~800 KB | ~200–300 KB |
| Speedup (CPU) | 1× | ~1.5–2× |

> Exact numbers vary by hardware. The key takeaway is that QAT maintains accuracy while significantly reducing model size.

---

## 📚 Further Reading

- [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html)
- [PyTorch QAT Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [Introduction to Quantization on PyTorch (Blog)](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (Paper)](https://arxiv.org/abs/1712.05877)

---

## ⚠️ Notes & Limitations

- **CPU only**: PyTorch eager-mode quantization (`torch.ao.quantization`) only supports CPU backends. GPU quantization requires `torch.compile` + `torchao`.
- **Toy model**: This CNN is intentionally simple. For production models, consider using `torch.ao.quantization` with architectures like MobileNet or ResNet.
- **Backend**: The `x86` qconfig is used. On ARM devices, use `qnnpack` instead.
