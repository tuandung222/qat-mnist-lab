# 📘 Quantization-Aware Training (QAT) — Hướng Dẫn Chi Tiết

> Tài liệu giáo dục này giải thích từng bước về **Quantization-Aware Training** trong deep learning, sử dụng PyTorch và mạng CNN đơn giản trên tập MNIST làm ví dụ minh hoạ.

---

## Mục lục

1. [Tổng quan về Quantization](#1-tổng-quan-về-quantization)
2. [Tại sao cần Quantization?](#2-tại-sao-cần-quantization)
3. [Các phương pháp Quantization](#3-các-phương-pháp-quantization)
4. [Deep dive: Quantization-Aware Training](#4-deep-dive-quantization-aware-training)
5. [Kiến trúc mô hình trong project này](#5-kiến-trúc-mô-hình-trong-project-này)
6. [Giải thích từng bước trong code](#6-giải-thích-từng-bước-trong-code)
7. [Kết quả mẫu & phân tích](#7-kết-quả-mẫu--phân-tích)
8. [Các câu hỏi thường gặp (FAQ)](#8-các-câu-hỏi-thường-gặp-faq)
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)

---

## 1. Tổng quan về Quantization

### 1.1 Quantization là gì?

**Quantization** (lượng tử hoá) là kỹ thuật giảm **độ chính xác số học** (numerical precision) của các tham số và phép tính trong mạng neural. Cụ thể:

```
FP32 (32-bit floating point)  →  INT8 (8-bit integer)
```

Mỗi weight và activation trong mô hình, thay vì được lưu dưới dạng số thực 32-bit, sẽ được **ánh xạ** (map) sang một số nguyên 8-bit thông qua công thức:

```
x_quantized = round(x_real / scale) + zero_point
```

Trong đó:
- **`scale`** (hệ số tỷ lệ): xác định khoảng giá trị thực mà mỗi "bước" INT8 đại diện
- **`zero_point`** (điểm gốc): giá trị INT8 tương ứng với 0.0 trong không gian thực
- **`round()`**: làm tròn đến số nguyên gần nhất — đây chính là nguồn gốc của **quantization error**

### 1.2 Ví dụ trực quan

Giả sử một weight có giá trị `0.73` và ta có `scale=0.01`, `zero_point=128`:

```
x_q = round(0.73 / 0.01) + 128 = round(73) + 128 = 201

Khi de-quantize (giải lượng tử):
x_dq = (201 - 128) × 0.01 = 0.73   ✅ Chính xác!
```

Nhưng nếu giá trị là `0.735`:

```
x_q = round(0.735 / 0.01) + 128 = round(73.5) + 128 = 74 + 128 = 202
x_dq = (202 - 128) × 0.01 = 0.74   ⚠️ Sai lệch 0.005 (quantization error)
```

> Mặc dù sai lệch này rất nhỏ cho một tham số, nhưng khi nhân lên hàng triệu tham số, nó có thể ảnh hưởng đáng kể đến accuracy.

---

## 2. Tại sao cần Quantization?

### 2.1 Lợi ích

| Lợi ích | Giải thích |
|---------|-----------|
| 📦 **Giảm kích thước mô hình** | FP32: 4 bytes/param → INT8: 1 byte/param → **giảm ~4×** |
| ⚡ **Tăng tốc inference** | Phép tính INT8 nhanh hơn FP32 trên hầu hết CPU/NPU |
| 🔋 **Tiết kiệm năng lượng** | Ít bộ nhớ truy cập, ít phép tính → tiêu thụ ít điện hơn |
| 📱 **Deploy trên edge** | Mô hình nhỏ gọn, chạy được trên điện thoại, IoT, vi điều khiển |

### 2.2 Khi nào nên dùng?

```
Training (GPU, cloud)           →  FP32 hoặc FP16/BF16
                                          ↓
                  Quantization (FP32 → INT8)
                                          ↓
Deployment (CPU, mobile, edge)  →  INT8 (nhỏ, nhanh)
```

Quantization đặc biệt quan trọng khi:
- Deploy mô hình lên **thiết bị di động** (Android, iOS)
- Chạy trên **vi điều khiển** (MCU) với bộ nhớ giới hạn
- Giảm **chi phí server** cho inference trên cloud
- Cần **latency thấp** cho ứng dụng real-time

---

## 3. Các phương pháp Quantization

### 3.1 Post-Training Quantization (PTQ)

**Ý tưởng**: Sau khi train xong mô hình FP32, ta quantize trực tiếp sang INT8 **mà không cần train lại**.

```
┌─────────────────┐       ┌──────────────────┐       ┌──────────────┐
│ Trained FP32    │ ────► │  Calibrate with   │ ────► │ Quantized    │
│ Model           │       │  sample data      │       │ INT8 Model   │
└─────────────────┘       └──────────────────┘       └──────────────┘
```

**Ưu điểm**: Nhanh, đơn giản, không cần training data đầy đủ.
**Nhược điểm**: Có thể giảm accuracy đáng kể, đặc biệt với mô hình nhỏ.

### 3.2 Quantization-Aware Training (QAT)  ← **Focus của project này**

**Ý tưởng**: Trong quá trình training (hoặc fine-tuning), ta **mô phỏng** (simulate) hiệu ứng quantization để mô hình **học cách bù đắp** (compensate) cho quantization error.

```
┌─────────────────┐       ┌──────────────────┐       ┌──────────────┐       ┌──────────────┐
│ Pre-trained     │ ────► │  Insert fake      │ ────► │ Fine-tune    │ ────► │ Convert to   │
│ FP32 Model      │       │  quantization     │       │ (QAT)        │       │ real INT8    │
└─────────────────┘       └──────────────────┘       └──────────────┘       └──────────────┘
```

**Ưu điểm**: Accuracy cao hơn PTQ vì mô hình đã "quen" với quantization noise.
**Nhược điểm**: Cần thêm vài epoch fine-tuning.

### 3.3 So sánh PTQ vs QAT

```
Accuracy
  ▲
  │  ████ FP32 (baseline)
  │  ████████████████████████████████████████  99.1%
  │
  │  ████ QAT INT8
  │  ███████████████████████████████████████   98.9%  (giảm ~0.2%)
  │
  │  ████ PTQ INT8
  │  ████████████████████████████████████      97.5%  (giảm ~1.6%)
  │
  └──────────────────────────────────────────►
```

> **Kết luận**: QAT cho accuracy gần FP32 hơn nhiều so với PTQ, đặc biệt quan trọng với mô hình nhỏ hoặc task nhạy cảm.

---

## 4. Deep dive: Quantization-Aware Training

### 4.1 Fake Quantization (Lượng tử hoá giả)

Đây là **cơ chế cốt lõi** của QAT. Trong forward pass:

```
                    Forward Pass (Training)
                    ═══════════════════════

Input (FP32)
    │
    ▼
┌─────────────────────────────────────────┐
│         Fake Quantize (simulate)        │
│                                         │
│   x_fq = dequant(quant(x))             │
│        = (round(x/s) + z - z) × s      │
│        = round(x/s) × s                │
│                                         │
│   → Vẫn là FP32 nhưng đã bị "làm tròn" │
│     giống như INT8                       │
└─────────────────────────────────────────┘
    │
    ▼
Conv2d / Linear (FP32 arithmetic)
    │
    ▼
Output (FP32, nhưng mang "dấu ấn" quantization)
```

**Tại sao gọi là "fake"?**
- Dữ liệu **vẫn là FP32** (cần cho backpropagation)
- Nhưng đã bị **làm tròn giống INT8** → mô hình "cảm nhận" được quantization error
- Gradient vẫn chảy bình thường qua **Straight-Through Estimator (STE)**

### 4.2 Straight-Through Estimator (STE)

Phép `round()` có đạo hàm = 0 gần như mọi nơi, nên không thể backpropagate qua được. STE giải quyết bằng cách:

```
Forward:   x_out = round(x)        ← dùng round() thực sự
Backward:  ∂L/∂x = ∂L/∂x_out × 1  ← giả vờ round() là hàm đồng nhất (identity)
```

Nhờ STE, gradient "chảy xuyên qua" (straight through) phép round, cho phép mô hình học cách điều chỉnh weights để **tối thiểu hoá loss ngay cả khi bị quantize**.

### 4.3 Observer & QConfig

**Observer** theo dõi (observe) phân phối giá trị của activations/weights trong quá trình training để xác định `scale` và `zero_point` tối ưu.

```python
# QConfig = cấu hình cho quantization
qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")

# Bên trong, qconfig chỉ định:
# - activation observer: theo dõi phạm vi giá trị activation
# - weight observer:     theo dõi phạm vi giá trị weight
# - quant scheme:        per-tensor hay per-channel
# - dtype:               quint8 (unsigned) hoặc qint8 (signed)
```

**Các loại observer thường gặp:**

| Observer | Cách hoạt động |
|----------|----------------|
| `MinMaxObserver` | Theo dõi min/max tuyệt đối → tính scale |
| `MovingAverageMinMaxObserver` | Giống trên nhưng dùng trung bình động (ít ảnh hưởng bởi outlier) |
| `HistogramObserver` | Xây histogram → tìm ngưỡng tối ưu bằng KL-divergence |
| `PerChannelMinMaxObserver` | Min/max riêng cho từng channel (cho weights) |

### 4.4 Fuse Modules (Hợp nhất các lớp)

Trước khi QAT, ta cần **fuse** (hợp nhất) các cặp layer liền kề:

```
Trước fuse:                    Sau fuse:
┌─────────┐                    ┌───────────────┐
│ Conv2d  │                    │ ConvReLU2d    │  ← một module duy nhất
├─────────┤        ────►       │ (fused)       │
│ ReLU    │                    └───────────────┘
└─────────┘
```

**Tại sao cần fuse?**

1. **Chính xác hơn**: Observer đo phạm vi **sau ReLU** (luôn ≥ 0), thay vì đo riêng Conv (có thể âm) và ReLU
2. **Nhanh hơn**: Giảm một operation trong inference
3. **Các cặp có thể fuse**: `Conv+ReLU`, `Conv+BN+ReLU`, `Linear+ReLU`, `BN+ReLU`

```python
# Trong model.py:
torch.ao.quantization.fuse_modules(
    self,
    [
        ["conv1", "relu1"],    # Conv2d + ReLU → ConvReLU2d
        ["conv2", "relu2"],    # Conv2d + ReLU → ConvReLU2d
        ["fc1",   "relu3"],    # Linear + ReLU → LinearReLU
    ],
    inplace=True,
)
```

---

## 5. Kiến trúc mô hình trong project này

### 5.1 SimpleCNN (Baseline FP32)

```
┌───────────────────────────────────────────────────────────────┐
│                       SimpleCNN                               │
│                                                               │
│  Input: 1×28×28 (MNIST grayscale)                            │
│                                                               │
│  ┌──────────────┐   ┌──────┐   ┌───────────┐                │
│  │ Conv2d       │──►│ ReLU │──►│ MaxPool2d │  → 32×14×14    │
│  │ 1→32, 3×3   │   └──────┘   │ kernel=2  │                │
│  └──────────────┘              └───────────┘                │
│                                                               │
│  ┌──────────────┐   ┌──────┐   ┌───────────┐                │
│  │ Conv2d       │──►│ ReLU │──►│ MaxPool2d │  → 64×7×7     │
│  │ 32→64, 3×3  │   └──────┘   │ kernel=2  │                │
│  └──────────────┘              └───────────┘                │
│                                                               │
│  Flatten → 64×7×7 = 3136                                     │
│                                                               │
│  ┌──────────────┐   ┌──────┐                                 │
│  │ Linear       │──►│ ReLU │  → 128                          │
│  │ 3136→128     │   └──────┘                                 │
│  └──────────────┘                                            │
│                                                               │
│  ┌──────────────┐                                            │
│  │ Linear       │  → 10 (classes 0-9)                        │
│  │ 128→10       │                                            │
│  └──────────────┘                                            │
└───────────────────────────────────────────────────────────────┘
```

### 5.2 QuantizedCNN (QAT-ready)

Kiến trúc **giống hệt** SimpleCNN, chỉ thêm hai thành phần:

```
┌───────────────────────────────────────────────────────────────┐
│                      QuantizedCNN                             │
│                                                               │
│  ┌───────────┐  ← THÊM: Chuyển FP32 → fake-quantized       │
│  │ QuantStub │                                                │
│  └─────┬─────┘                                                │
│        │                                                      │
│   (Cùng kiến trúc CNN như SimpleCNN)                         │
│        │                                                      │
│  ┌─────┴───────┐  ← THÊM: Chuyển fake-quantized → FP32     │
│  │ DeQuantStub │                                              │
│  └─────────────┘                                              │
│                                                               │
│  + fuse_model() method để fuse Conv+ReLU, Linear+ReLU        │
└───────────────────────────────────────────────────────────────┘
```

**Tại sao cần hai model riêng?**
- `SimpleCNN`: dùng cho training FP32 bình thường, không có overhead từ quant stubs
- `QuantizedCNN`: chứa QuantStub/DeQuantStub và fuse_model() cần thiết cho QAT
- Cả hai **chia chung tên layer** nên có thể load state_dict từ SimpleCNN sang QuantizedCNN

---

## 6. Giải thích từng bước trong code

### 6.1 Bước 1: Train baseline (`train_baseline.py`)

```
Mục đích: Có một mô hình FP32 đã train tốt → làm "điểm xuất phát" cho QAT
```

**Luồng thực thi:**

```
1. Tải MNIST dataset
       ↓
2. Khởi tạo SimpleCNN
       ↓
3. Train 5 epochs với Adam optimizer (lr=1e-3)
       ↓
4. Đánh giá trên test set → ~99% accuracy
       ↓
5. Lưu state_dict → baseline_model.pth
```

**Code quan trọng:**

```python
# Normalize MNIST theo mean/std chuẩn
transforms.Normalize((0.1307,), (0.3081,))

# Lưu state_dict (không lưu toàn bộ model)
# → Nhẹ hơn, portable hơn, tránh vấn đề serialization
torch.save(model.state_dict(), "baseline_model.pth")
```

### 6.2 Bước 2: QAT fine-tuning (`train_qat.py`)

Đây là **phần quan trọng nhất**. Luồng thực thi chi tiết:

```
╔═══════════════════════════════════════════════════════════════════╗
║                    QAT Pipeline — Từng bước                      ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Step 1: Load pre-trained weights                                ║
║  ─────────────────────────────────                               ║
║  model = QuantizedCNN()                                          ║
║  model.load_state_dict(baseline_weights, strict=False)           ║
║                                                                   ║
║  → strict=False: bỏ qua quant/dequant keys không có              ║
║    trong SimpleCNN (vì SimpleCNN không có QuantStub)             ║
║                                                                   ║
║──────────────────────────────────────────────────────────────────║
║                                                                   ║
║  Step 2: Fuse modules                                            ║
║  ────────────────────                                            ║
║  model.fuse_model()                                              ║
║                                                                   ║
║  Trước:  conv1 → relu1 → pool1 → conv2 → relu2 → pool2         ║
║  Sau:    conv1_relu1 → pool1 → conv2_relu2 → pool2              ║
║          (ConvReLU2d)         (ConvReLU2d)                       ║
║                                                                   ║
║──────────────────────────────────────────────────────────────────║
║                                                                   ║
║  Step 3: Set QConfig                                             ║
║  ───────────────────                                             ║
║  model.qconfig = get_default_qat_qconfig("x86")                 ║
║                                                                   ║
║  → Chỉ định observer loại nào cho activation & weight            ║
║  → "x86": tối ưu cho CPU Intel/AMD                              ║
║  → Alternatives: "qnnpack" (ARM), "fbgemm" (Facebook)           ║
║                                                                   ║
║──────────────────────────────────────────────────────────────────║
║                                                                   ║
║  Step 4: prepare_qat()                                           ║
║  ─────────────────────                                           ║
║  quant.prepare_qat(model, inplace=True)                          ║
║                                                                   ║
║  → Chèn FakeQuantize modules vào mỗi layer                      ║
║  → Mỗi forward pass sẽ: quantize → dequantize (simulate INT8)   ║
║  → Observer bắt đầu thu thập statistics                          ║
║                                                                   ║
║──────────────────────────────────────────────────────────────────║
║                                                                   ║
║  Step 5: Fine-tune (3 epochs)                                    ║
║  ────────────────────────────                                    ║
║  for epoch in range(3):                                          ║
║      train_one_epoch(model, ...)                                 ║
║                                                                   ║
║  → Learning rate thấp hơn (1e-4 vs 1e-3) vì chỉ fine-tune      ║
║  → Mô hình học cách điều chỉnh weights để bù quantization noise ║
║  → Observer cập nhật scale/zero_point liên tục                   ║
║                                                                   ║
║──────────────────────────────────────────────────────────────────║
║                                                                   ║
║  Step 6: convert()                                               ║
║  ─────────────────                                               ║
║  model.eval()                       ← BẮT BUỘC trước convert    ║
║  quantized = quant.convert(model)                                ║
║                                                                   ║
║  → Thay FakeQuantize bằng quantized operators thực               ║
║  → nn.Conv2d → nn.quantized.Conv2d  (INT8 arithmetic)           ║
║  → nn.Linear → nn.quantized.Linear  (INT8 arithmetic)           ║
║  → Mô hình bây giờ thực sự chạy INT8!                           ║
║                                                                   ║
║──────────────────────────────────────────────────────────────────║
║                                                                   ║
║  Step 7: Save                                                    ║
║  ────────                                                        ║
║  torch.save(quantized.state_dict(), "qat_model.pth")            ║
║                                                                   ║
║  → File nhỏ hơn ~3-4× so với baseline_model.pth                 ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

### 6.3 Bước 3: So sánh (`compare.py`)

```
1. Load SimpleCNN ← baseline_model.pth
2. Load QuantizedCNN ← qat_model.pth
   (phải reproduce cùng pipeline: fuse → prepare_qat → convert → load)
3. Đo: file size, test accuracy, inference latency
4. In bảng so sánh
```

**Lưu ý quan trọng**: Khi load mô hình quantized, bạn phải **reproduce toàn bộ pipeline** (fuse → prepare_qat → convert) trước khi load state_dict. Đây là vì cấu trúc module thay đổi sau convert().

---

## 7. Kết quả mẫu & phân tích

### 7.1 Bảng kết quả dự kiến

```
════════════════════════════════════════════════════════════════════
  📊  Model Comparison: FP32 vs QAT INT8
════════════════════════════════════════════════════════════════════
Metric                         FP32 Baseline        QAT INT8           Δ
──────────────────────────────────────────────────────────────────────
Model file size (KB)                  ~800            ~200-300      ~25-35%
Test accuracy (%)                    ~99.1             ~98.9        -0.2
Avg batch latency (ms)                ~5.0              ~3.0        -40%
──────────────────────────────────────────────────────────────────────
```

### 7.2 Phân tích

**Kích thước mô hình:**
- FP32: mỗi parameter chiếm 4 bytes (float32)
- INT8: mỗi parameter chiếm 1 byte → giảm ~4×
- Thực tế giảm ~3× vì có thêm metadata (scale, zero_point)

**Accuracy:**
- Giảm rất ít (~0.1-0.3%) nhờ QAT cho phép mô hình thích nghi
- Với PTQ thuần tuý, accuracy có thể giảm ~1-2%

**Latency:**
- Phép tính INT8 nhanh hơn FP32 trên CPU
- Tốc độ tăng phụ thuộc vào phần cứng (CPU có hỗ trợ AVX-512 VNNI sẽ nhanh hơn)

---

## 8. Các câu hỏi thường gặp (FAQ)

### Q1: QAT chỉ chạy trên CPU, không chạy trên GPU?

**Đúng** (với PyTorch eager-mode quantization). Đây là hạn chế của `torch.ao.quantization`:
- **Training QAT**: chạy trên CPU (chậm hơn nhưng cần ít epoch)
- **Inference INT8**: chạy trên CPU

Để quantize cho GPU, dùng:
- `torch.compile` + `torchao` (PyTorch 2.x+)
- TensorRT (NVIDIA)
- ONNX Runtime

### Q2: Tại sao cần `model.eval()` trước `convert()`?

- `convert()` cần model ở eval mode để **đóng băng** BatchNorm statistics
- Nếu không, BN sẽ tiếp tục cập nhật running mean/variance → kết quả không ổn định

### Q3: Khi nào nên dùng PTQ thay vì QAT?

| Tình huống | Nên dùng |
|-----------|---------|
| Mô hình lớn (ResNet-50+) | PTQ thường đủ tốt |
| Accuracy giảm < 1% với PTQ | PTQ |
| Accuracy giảm > 1% với PTQ | **QAT** |
| Mô hình nhỏ (MobileNet, CNN nhỏ) | **QAT** (nhạy hơn với quantization) |
| Không có training pipeline | PTQ |

### Q4: `strict=False` khi load state_dict nghĩa là gì?

```python
model.load_state_dict(state_dict, strict=False)
```

- `strict=True` (mặc định): yêu cầu **tất cả keys phải khớp** chính xác
- `strict=False`: **bỏ qua keys thừa/thiếu**

Trong project này, `QuantizedCNN` có thêm `quant` và `dequant` modules mà `SimpleCNN` không có → cần `strict=False`.

### Q5: Per-tensor vs Per-channel quantization là gì?

- **Per-tensor**: một cặp (scale, zero_point) cho **toàn bộ tensor**
- **Per-channel**: mỗi **output channel** có cặp (scale, zero_point) riêng

```
Per-tensor:    toàn bộ weight tensor chia chung 1 scale
Per-channel:   channel 0 có scale_0, channel 1 có scale_1, ...
```

Per-channel cho accuracy tốt hơn (đặc biệt với weights), nhưng phức tạp hơn.
PyTorch mặc định dùng per-channel cho weights và per-tensor cho activations.

---

## 9. Tài liệu tham khảo

### Papers
1. **Jacob et al. (2018)** — [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
   - Paper gốc giới thiệu QAT, được Google áp dụng cho MobileNet
2. **Krishnamoorthi (2018)** — [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/abs/1806.08342)
   - Tổng quan toàn diện về các phương pháp quantization

### PyTorch Documentation
3. [PyTorch Quantization Overview](https://pytorch.org/docs/stable/quantization.html)
4. [Static Quantization Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
5. [QAT Tutorial with MobileNetV2](https://pytorch.org/tutorials/prototype/quantization_in_pytorch_2_0_export_tutorial.html)

### Blog Posts
6. [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
7. [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)

---

> 📝 **Ghi chú**: Tài liệu này được thiết kế cho mục đích giáo dục. Trong production, hãy tham khảo thêm các kỹ thuật nâng cao như mixed-precision quantization, quantization-aware distillation, và hardware-specific optimization.
