# 📘 Quantization-Aware Training (QAT) — Hướng Dẫn Chi Tiết

> **Tài liệu giáo dục** giải thích **từ gốc rễ** về Quantization-Aware Training.
> Mọi thứ được implement from scratch — **KHÔNG dùng `torch.ao.quantization`**.
> Có mã giả (pseudocode) cho mọi thành phần.

---

## Mục lục

1. [Quantization là gì?](#1-quantization-là-gì)
2. [Tại sao cần Quantization?](#2-tại-sao-cần-quantization)
3. [Fake Quantize — trái tim của QAT](#3-fake-quantize--trái-tim-của-qat)
4. [Mối liên hệ PTQ ↔ QAT](#4-mối-liên-hệ-ptq--qat)
5. [Activation và Weight trong QAT](#5-activation-và-weight-trong-qat)
6. [Straight-Through Estimator (STE)](#6-straight-through-estimator-ste)
7. [Observer — thu thập thống kê](#7-observer--thu-thập-thống-kê)
8. [Kiến trúc from-scratch trong project](#8-kiến-trúc-from-scratch-trong-project)
9. [Giải thích code từng bước](#9-giải-thích-code-từng-bước)
10. [FAQ](#10-faq)
11. [Tài liệu tham khảo](#11-tài-liệu-tham-khảo)

---

## 1. Quantization là gì?

### 1.1 Định nghĩa

**Quantization** (lượng tử hoá) là quá trình ánh xạ giá trị từ **không gian liên tục** (floating point) sang **không gian rời rạc** (integer).

```
Thế giới thực (FP32):    0.0000  0.0001  0.0002  ...  1.5847  ...
                           ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
Thế giới quantized (INT8): 0      1       2      ...   158    ...
```

### 1.2 Công thức cốt lõi

```
┌─────────────────────────────────────────────────────────────┐
│  QUANTIZE (số thực → số nguyên):                           │
│                                                             │
│    x_int = clamp(round(x_real / scale) + zero_point,       │
│                  q_min, q_max)                              │
│                                                             │
│  DEQUANTIZE (số nguyên → số thực):                         │
│                                                             │
│    x_real ≈ (x_int - zero_point) × scale                   │
│                                                             │
│  Trong đó:                                                  │
│    scale      = (max_val - min_val) / (q_max - q_min)       │
│    zero_point = q_min - round(min_val / scale)              │
│    q_min, q_max = phạm vi INT8 (ví dụ: 0..255 hoặc -128..127) │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Ví dụ bằng số

Giả sử một tensor weight có giá trị thuộc khoảng `[-0.5, 1.2]`, và ta dùng **unsigned INT8** (`q_min=0, q_max=255`):

```
scale      = (1.2 - (-0.5)) / (255 - 0) = 1.7 / 255 ≈ 0.00667

zero_point = 0 - round(-0.5 / 0.00667) = 0 - round(-75) = 75

Quantize giá trị 0.73:
  x_int = clamp(round(0.73 / 0.00667) + 75, 0, 255)
        = clamp(round(109.4) + 75, 0, 255)
        = clamp(109 + 75, 0, 255)
        = clamp(184, 0, 255)
        = 184  ✅

Dequantize:
  x_real ≈ (184 - 75) × 0.00667 = 109 × 0.00667 ≈ 0.7270

Sai lệch: |0.73 - 0.727| = 0.003  ← đây là quantization error
```

> **Quantization error** là sai lệch do phép làm tròn. Nó nhỏ cho một tham số, nhưng tích luỹ qua hàng triệu tham số và nhiều layer.

---

## 2. Tại sao cần Quantization?

```
┌─────────────────────────────────────────────────────────────┐
│                    So sánh FP32 vs INT8                     │
├──────────────────────┬──────────────┬───────────────────────┤
│ Tiêu chí             │ FP32         │ INT8                  │
├──────────────────────┼──────────────┼───────────────────────┤
│ Kích thước/param     │ 4 bytes      │ 1 byte                │
│ Mô hình 10M params  │ ~40 MB       │ ~10 MB                │
│ Phép nhân            │ FP multiply  │ INT multiply (nhanh!) │
│ Phần cứng tối ưu     │ GPU          │ CPU, NPU, MCU         │
│ Deploy mobile        │ Khó          │ Dễ                    │
│ Tiêu thụ điện        │ Cao          │ Thấp                  │
└──────────────────────┴──────────────┴───────────────────────┘
```

Kết luận: Quantization giúp mô hình **nhỏ hơn ~4×**, **nhanh hơn ~2×** trên CPU, và **tiết kiệm năng lượng** — lý tưởng cho edge deployment.

---

## 3. Fake Quantize — trái tim của QAT

### 3.1 Vấn đề: `round()` giết gradient

Trong training, ta cần **backpropagation** — tức cần gradient. Nhưng phép `round()` có vấn đề:

```
round(x):    ──┐  ┌──┐  ┌──┐  ┌──     (bậc thang)
               └──┘  └──┘  └──┘

d/dx round(x) = 0   (gần như mọi nơi)

→ Gradient = 0 → Weight KHÔNG được cập nhật → KHÔNG THỂ TRAIN!
```

### 3.2 Giải pháp: Fake Quantize

**Ý tưởng cốt lõi**: Trong **forward pass**, ta mô phỏng quantization (round + clamp). Trong **backward pass**, ta **giả vờ** phép round không tồn tại → gradient chảy thẳng qua.

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  MÃ GIẢ — FAKE QUANTIZE:                                        │
│                                                                  │
│  function FAKE_QUANTIZE(x, scale, zero_point, q_min, q_max):    │
│                                                                  │
│      ──── FORWARD ────                                           │
│      x_int  = clamp(round(x / scale) + zero_point, q_min, q_max)│
│      x_fake = (x_int - zero_point) * scale                      │
│      return x_fake                                               │
│      // Kết quả: vẫn là FP32, nhưng chỉ chứa giá trị           │
│      // "có thể biểu diễn" bởi INT8                             │
│      // Tức là x_fake ∈ {(i - zp) * s | i ∈ [q_min, q_max]}    │
│                                                                  │
│      ──── BACKWARD ────                                          │
│      // Straight-Through Estimator:                              │
│      q_min_real = (q_min - zero_point) * scale                   │
│      q_max_real = (q_max - zero_point) * scale                   │
│      mask = (x >= q_min_real) AND (x <= q_max_real)              │
│      grad_input = grad_output * mask                             │
│      return grad_input                                           │
│      // Gradient được chuyển nguyên vẹn nếu x nằm trong vùng    │
│      // quantizable, và bị chặn (=0) nếu x bị clamp            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 Tại sao gọi là "fake"?

| Thuộc tính | Quantize thật | Fake quantize |
|-----------|:---:|:---:|
| Kiểu dữ liệu output | INT8 | FP32 |
| Giá trị bị làm tròn? | ✅ | ✅ |
| Gradient chảy qua? | ❌ | ✅ (nhờ STE) |
| Dùng khi nào? | Inference | Training (QAT) |

Nói cách khác: **fake quantize = quantize + dequantize ngay lập tức**, giữ kiểu FP32 để gradient vẫn chảy, nhưng giá trị đã bị "làm hỏng" giống như khi bị quantize thật.

### 3.4 Hình dung trực quan

```
             Giá trị gốc (FP32)
             ┃
    0.01  0.03  0.07  0.08  0.12  0.15  0.18  0.22
     │     │     │     │     │     │     │     │
     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
   ┌─────────────────────────────────────────────────┐
   │            FAKE QUANTIZE (scale=0.05)           │
   │                                                  │
   │  round(0.01/0.05)=0  → dequant → 0.00          │
   │  round(0.03/0.05)=1  → dequant → 0.05          │
   │  round(0.07/0.05)=1  → dequant → 0.05          │
   │  round(0.08/0.05)=2  → dequant → 0.10          │
   │  round(0.12/0.05)=2  → dequant → 0.10          │
   │  round(0.15/0.05)=3  → dequant → 0.15          │
   │  round(0.18/0.05)=4  → dequant → 0.20          │
   │  round(0.22/0.05)=4  → dequant → 0.20          │
   └─────────────────────────────────────────────────┘
     │     │     │     │     │     │     │     │
     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
    0.00  0.05  0.05  0.10  0.10  0.15  0.20  0.20
             ┃
             Giá trị sau fake quantize (vẫn FP32, nhưng "bậc thang")
```

> **Giá trị output vẫn là FP32** nhưng đã bị "kéo" vào các mức rời rạc — chính xác giống như khi quantize sang INT8 rồi dequantize lại.

---

## 4. Mối liên hệ PTQ ↔ QAT

### 4.1 PTQ (Post-Training Quantization) — Quantize SAU khi train

```
┌────────────────────────────────────────────────────────────────┐
│  MÃ GIẢ — POST-TRAINING QUANTIZATION (PTQ):                  │
│                                                                │
│  // Bước 1: Train mô hình bình thường                         │
│  model = train_fp32(dataset)        // output: FP32 model      │
│                                                                │
│  // Bước 2: Calibrate — chạy vài batch qua model              │
│  //   để observer thu thập min/max của mỗi layer              │
│  for batch in calibration_data:                                │
│      model.forward(batch)           // forward only, no grad   │
│      observer.update(activations)   // ghi nhận min/max        │
│                                                                │
│  // Bước 3: Tính scale/zero_point từ statistics               │
│  for each layer:                                               │
│      scale, zp = compute_qparams(observer.min, observer.max)  │
│                                                                │
│  // Bước 4: Quantize weight trực tiếp                         │
│  for each layer:                                               │
│      layer.weight_int8 = quantize(layer.weight_fp32, scale, zp)│
│                                                                │
│  // Kết quả: mô hình INT8 — KHÔNG cần thêm training          │
│  // Nhưng accuracy có thể giảm vì model chưa bao giờ "thấy"  │
│  // quantization error trong quá trình training                │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 QAT (Quantization-Aware Training) — Quantize TRONG LÚC train

```
┌────────────────────────────────────────────────────────────────┐
│  MÃ GIẢ — QUANTIZATION-AWARE TRAINING (QAT):                 │
│                                                                │
│  // Bước 1: Bắt đầu từ mô hình FP32 đã pre-train (hoặc mới) │
│  model = load_pretrained_fp32()                                │
│                                                                │
│  // Bước 2: Chèn fake_quantize vào mỗi layer                 │
│  for each layer in model:                                      │
│      layer = wrap_with_fake_quantize(layer)                    │
│                                                                │
│  // Bước 3: Fine-tune VỚI fake quantization                   │
│  for epoch in range(few_epochs):                               │
│      for batch in training_data:                               │
│                                                                │
│          // FORWARD: mỗi layer tự động fake-quantize           │
│          //   activation và weight                             │
│          output = model.forward(batch)                         │
│          loss = criterion(output, labels)                      │
│                                                                │
│          // BACKWARD: STE cho phép gradient chảy qua           │
│          //   fake_quantize → weight được cập nhật (FP32)      │
│          //   → epoch tiếp, weight lại bị fake_quantize        │
│          //   → model học cách TỐI ƯU ĐỂ CHỊU ĐỰNG quant    │
│          loss.backward()                                       │
│          optimizer.step()                                      │
│                                                                │
│  // Bước 4: Convert sang INT8 thật                             │
│  //   Dùng scale/zp từ observer → quantize weight thành INT8  │
│  final_model = convert_to_int8(model)                         │
└────────────────────────────────────────────────────────────────┘
```

### 4.3 So sánh — CÙNG CƠ CHẾ, KHÁC THỜI ĐIỂM

```
                PTQ                              QAT
                ═══                              ═══

  ┌──────────┐                       ┌──────────┐
  │ Train    │ ← FP32 thuần         │ Train    │ ← FP32 thuần
  │ (bình   │   (không biết gì     │ (bình   │   (pre-training)
  │ thường)  │    về quantization)   │ thường)  │
  └────┬─────┘                       └────┬─────┘
       │                                   │
       ▼                                   ▼
  ┌──────────┐                       ┌──────────────┐
  │Calibrate │ ← observer xem       │ Fine-tune    │ ← fake quantize
  │(forward  │   min/max, tính      │ VỚI fake     │   trong forward,
  │ only)    │   scale/zp           │ quantize     │   STE trong backward
  └────┬─────┘                       │ (vài epoch) │
       │                              └────┬────────┘
       │                                   │
       ▼                                   ▼
  ┌──────────┐                       ┌──────────┐
  │ Convert  │ ← quantize weight    │ Convert  │ ← quantize weight
  │ to INT8  │   trực tiếp          │ to INT8  │   (model ĐÃ quen)
  └──────────┘                       └──────────┘

 ✦ Cùng dùng scale/zp              ✦ Cùng dùng scale/zp
 ✦ Cùng dùng observer              ✦ Cùng dùng observer
 ✦ NHƯNG: model CHƯA BAO GIỜ      ✦ NHƯNG: model ĐÃ LUYỆN TẬP
   thấy quantization error            với quantization error
   → accuracy giảm nhiều hơn          → accuracy giảm ít
```

> **Kết luận**: PTQ và QAT **dùng cùng cơ chế toán học** (scale, zero_point, quantize, dequantize). Sự khác biệt duy nhất là QAT cho mô hình **cơ hội thích nghi** với quantization error thông qua fake quantize + fine-tuning.

---

## 5. Activation và Weight trong QAT

### 5.1 Activation là gì?

```
Trong neural network, "activation" = OUTPUT của mỗi layer:

  Input → [Conv2d] → activation₁ → [ReLU] → activation₂ → [MaxPool] → activation₃
                       ^^^^^^^^               ^^^^^^^^                   ^^^^^^^^
                       đây là                 đây là                    đây là
                       activation             activation                activation
```

Activation là **dữ liệu chảy qua mạng** — nó thay đổi mỗi batch. Khác với weight (cố định trong inference), activation **phụ thuộc vào input**.

### 5.2 Weight vs Activation — khác nhau cơ bản

```
┌────────────────────────┬──────────────────┬──────────────────────┐
│                        │ WEIGHT           │ ACTIVATION           │
├────────────────────────┼──────────────────┼──────────────────────┤
│ Là gì?                 │ Tham số học được │ Output của mỗi layer │
│ Thay đổi mỗi batch?   │ ❌ Không         │ ✅ Có                │
│ Biết trước phân phối?  │ ✅ Có (sau train)│ ❌ Không (phụ thuộc  │
│                        │                  │    vào input)        │
│ Quantize scheme        │ Per-channel      │ Per-tensor           │
│ Observer cần gì?       │ Đọc trực tiếp    │ Thu thập qua nhiều   │
│                        │ weight values    │ batch → running stats│
│ q_range thường dùng    │ [-128, 127]      │ [0, 255]             │
│                        │ (signed, vì      │ (unsigned, vì sau    │
│                        │  weight có âm)   │  ReLU luôn ≥ 0)     │
└────────────────────────┴──────────────────┴──────────────────────┘
```

### 5.3 Activation bị đối xử thế nào trong FORWARD?

```
┌──────────────────────────────────────────────────────────────────┐
│  MÃ GIẢ — XỬ LÝ ACTIVATION TRONG FORWARD:                      │
│                                                                  │
│  function QAT_CONV2D_FORWARD(input_activation, weight, bias):   │
│                                                                  │
│      // ① Observer THEO DÕI activation                          │
│      //    Cập nhật running_min, running_max                    │
│      if training:                                                │
│          act_observer.running_min = min(running_min, act.min())  │
│          act_observer.running_max = max(running_max, act.max()) │
│                                                                  │
│      // ② Tính scale, zero_point cho activation                 │
│      act_scale, act_zp = act_observer.compute_qparams()         │
│                                                                  │
│      // ③ FAKE QUANTIZE activation                              │
│      //    quantize → dequantize ngay → vẫn FP32 nhưng "noisy" │
│      act_fq = fake_quantize(input_activation,                    │
│                              act_scale, act_zp,                  │
│                              q_min=0, q_max=255)                │
│                                                                  │
│      // ④ FAKE QUANTIZE weight (tương tự)                       │
│      w_fq = fake_quantize(weight, w_scale, w_zp,                │
│                            q_min=-128, q_max=127)               │
│                                                                  │
│      // ⑤ Tính output bằng giá trị ĐÃ fake-quantize            │
│      output = conv2d(act_fq, w_fq, bias)                        │
│                                                                  │
│      return output // output chứa "quantization noise"          │
│                    // → loss sẽ phản ánh ảnh hưởng              │
│                    //   của quantization                        │
│                    // → backward sẽ cập nhật weight             │
│                    //   để bù đắp noise này                     │
└──────────────────────────────────────────────────────────────────┘
```

### 5.4 Activation và Weight bị đối xử thế nào trong BACKWARD?

```
┌──────────────────────────────────────────────────────────────────┐
│  MÃ GIẢ — BACKWARD PASS TRONG QAT:                              │
│                                                                  │
│  // Loss đã tính ở forward, bây giờ backward:                   │
│  // Gradient chảy ngược từ loss → output → ... → input          │
│                                                                  │
│  ════════════════════════════════════════════════════════════     │
│  GRADIENT QUA ACTIVATION FAKE QUANTIZE:                          │
│  ════════════════════════════════════════════════════════════     │
│                                                                  │
│  // Phép round() có gradient = 0 → không thể train              │
│  // → Dùng STE: "giả vờ" round() là identity function          │
│                                                                  │
│  grad_act = grad_output                   // STE: chuyển thẳng  │
│  NHƯNG: nếu activation bị CLAMP (ngoài [q_min, q_max]):        │
│    grad_act = 0                           // chặn gradient      │
│                                                                  │
│  // Nói cách khác:                                              │
│  if activation nằm TRONG vùng quantizable:                      │
│      grad chảy qua bình thường ← model có thể học               │
│  else (activation bị saturate):                                  │
│      grad = 0 ← "đây là vùng chết, không cập nhật"              │
│                                                                  │
│  ════════════════════════════════════════════════════════════     │
│  GRADIENT QUA WEIGHT FAKE QUANTIZE:                              │
│  ════════════════════════════════════════════════════════════     │
│                                                                  │
│  // Cùng cơ chế STE:                                            │
│  grad_weight = grad_output   // STE: chuyển thẳng               │
│  if weight BỊ CLAMP:                                             │
│      grad_weight = 0         // chặn gradient                   │
│                                                                  │
│  // SAU KHI có gradient, optimizer cập nhật weight:              │
│  weight_fp32 = weight_fp32 - lr * grad_weight    // FP32!       │
│                                                                  │
│  // Ở epoch tiếp theo, weight_fp32 sẽ lại bị fake_quantize     │
│  // → model dần dần học cách điều chỉnh weight                  │
│  //   sao cho KHI BỊ QUANTIZE vẫn cho kết quả tốt              │
│                                                                  │
│  ════════════════════════════════════════════════════════════     │
│  TÓM TẮT:                                                       │
│  ════════════════════════════════════════════════════════════     │
│                                                                  │
│  ① Weight được LƯU TRỮ ở FP32 (master copy)                    │
│  ② Forward: weight bị fake_quantize → round → noisy output      │
│  ③ Backward: gradient chảy QUA fake_quantize nhờ STE            │
│  ④ Optimizer cập nhật FP32 weight bằng gradient                 │
│  ⑤ Vòng lặp tiếp: weight FP32 mới → lại bị fake_quantize      │
│  → Model học cách: "weight nào bền vững khi bị round?"          │
│                                                                  │
│  Giống như tập luyện chạy với quả tạ ở chân (fake quantize noise)│
│  → khi bỏ tạ (deploy INT8), chạy nhanh mà vẫn khoẻ (accurate) │
└──────────────────────────────────────────────────────────────────┘
```

### 5.5 Hình dung toàn cảnh: một training step QAT

```
┌═══════════════════════════════════════════════════════════════════════┐
║                   MỘT TRAINING STEP TRONG QAT                       ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  ┌─────────┐                                                         ║
║  │  Batch  │  images (FP32)                                          ║
║  └────┬────┘                                                         ║
║       │                                                               ║
║       ▼                                                               ║
║  ╔════════════════════╗                                               ║
║  ║  QATConv2d Layer 1 ║                                               ║
║  ╠════════════════════╣                                               ║
║  ║                    ║                                               ║
║  ║  act_fq = FQ(x)   ║ ← activation bị fake quantize               ║
║  ║  w_fq = FQ(w)     ║ ← weight bị fake quantize                   ║
║  ║  out = conv(act_fq,║                                              ║
║  ║          w_fq)     ║ ← tính với giá trị "noisy"                  ║
║  ║                    ║                                               ║
║  ╚════════╤═══════════╝                                               ║
║           │                                                           ║
║           ▼                                                           ║
║       ReLU + Pool                                                     ║
║           │                                                           ║
║           ▼                                                           ║
║  ╔════════════════════╗                                               ║
║  ║  QATConv2d Layer 2 ║  (tương tự Layer 1)                          ║
║  ╚════════╤═══════════╝                                               ║
║           │                                                           ║
║           ▼                                                           ║
║  ╔════════════════════╗                                               ║
║  ║  QATLinear Layer 3 ║  (tương tự, FC)                              ║
║  ╚════════╤═══════════╝                                               ║
║           │                                                           ║
║           ▼                                                           ║
║  ╔════════════════════╗                                               ║
║  ║  QATLinear Layer 4 ║  (output layer)                              ║
║  ╚════════╤═══════════╝                                               ║
║           │                                                           ║
║           ▼                                                           ║
║     ┌──────────┐                                                      ║
║     │   Loss   │  = CrossEntropy(output, labels)                     ║
║     └────┬─────┘                                                      ║
║          │                                                            ║
║          │ loss.backward()                                            ║
║          │                                                            ║
║          ▼ ▼ ▼                                                        ║
║                                                                       ║
║     ╔═══════════════════════════════════════════╗                     ║
║     ║  BACKWARD — gradient chảy ngược          ║                     ║
║     ╠═══════════════════════════════════════════╣                     ║
║     ║                                           ║                     ║
║     ║  Qua FQ(weight):                          ║                     ║
║     ║    ∂L/∂w = ∂L/∂out × STE_mask             ║                     ║
║     ║    w_fp32 -= lr × ∂L/∂w  ← CẬP NHẬT FP32 ║                     ║
║     ║                                           ║                     ║
║     ║  Qua FQ(activation):                      ║                     ║
║     ║    ∂L/∂x = ∂L/∂out × STE_mask             ║                     ║
║     ║    → chảy tiếp về layer trước             ║                     ║
║     ║                                           ║                     ║
║     ╚═══════════════════════════════════════════╝                     ║
║                                                                       ║
║     optimizer.step()  →  weights cập nhật (FP32)                     ║
║     → Epoch tiếp → weight lại bị FQ → model thích nghi dần          ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
```

---

## 6. Straight-Through Estimator (STE)

### 6.1 Vấn đề

```
y = round(x)

dy/dx = ?

Toán học: dy/dx = 0   (hầu như mọi nơi)
                       (vì round là hàm bậc thang, đạo hàm = 0)

→ Nếu dùng gradient thật: mọi gradient = 0 → training KHÔNG HỌC GÌ
```

### 6.2 STE — giải pháp "hack" nhưng hiệu quả

```
┌──────────────────────────────────────────────────────────────────┐
│  STRAIGHT-THROUGH ESTIMATOR:                                     │
│                                                                  │
│  Forward:   y = round(x)           ← dùng round() thật         │
│  Backward:  ∂L/∂x = ∂L/∂y × 1     ← GIẢ VỜ round là identity  │
│                                                                  │
│  Tức là:                                                         │
│    Forward:   f(x) = round(x)     // hàm bậc thang              │
│    Backward:  f'(x) ≈ 1           // giả vờ f(x) = x           │
│                                                                  │
│  Gradient "đi thẳng qua" (straight through) phép round          │
└──────────────────────────────────────────────────────────────────┘
```

### 6.3 STE có thêm clamping

Trong thực tế, STE cũng **chặn gradient** ở vùng bị clamp:

```
                    ┌── gradient = 0 (vùng clamp trên)
                    │
  grad ────────────[████████████████]──────────── grad = 0
                   ↑                ↑
                 q_min            q_max
                 (vùng chết)    (vùng chết)
                   │                │
                   └── gradient chảy qua bình thường ──┘

  Nếu x ∈ [q_min_real, q_max_real]: grad qua bình thường
  Nếu x ∉ [q_min_real, q_max_real]: grad = 0  (bị clip)
```

### 6.4 Implement trong code (project này)

```python
class FakeQuantizeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, zero_point, q_min, q_max):
        x_int = torch.clamp(torch.round(x / scale) + zero_point, q_min, q_max)
        x_fake = (x_int - zero_point) * scale

        # Lưu lại để backward biết vùng nào bị clamp
        q_min_real = (q_min - zero_point) * scale
        q_max_real = (q_max - zero_point) * scale
        ctx.save_for_backward(x, q_min_real, q_max_real)

        return x_fake

    @staticmethod
    def backward(ctx, grad_output):
        x, q_min_real, q_max_real = ctx.saved_tensors

        # STE: chỉ cho gradient qua vùng KHÔNG bị clamp
        mask = (x >= q_min_real) & (x <= q_max_real)
        grad_input = grad_output * mask.float()

        return grad_input, None, None, None, None
```

---

## 7. Observer — thu thập thống kê

### 7.1 Vai trò

Observer **theo dõi** giá trị thực tế của activation/weight qua nhiều batch để tính **scale** và **zero_point** tối ưu.

```
┌──────────────────────────────────────────────────────────────────┐
│  MÃ GIẢ — MIN-MAX OBSERVER:                                     │
│                                                                  │
│  class MinMaxObserver:                                           │
│      running_min = +∞                                            │
│      running_max = -∞                                            │
│                                                                  │
│      function OBSERVE(tensor):                                   │
│          running_min = min(running_min, tensor.min())            │
│          running_max = max(running_max, tensor.max())            │
│                                                                  │
│      function COMPUTE_QPARAMS():                                │
│          range = running_max - running_min                       │
│          scale = range / (q_max - q_min)                        │
│          zero_point = q_min - round(running_min / scale)        │
│          zero_point = clamp(zero_point, q_min, q_max)           │
│          return scale, zero_point                                │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2 Tại sao cần observer cho activation?

- **Weight**: đã biết giá trị → có thể tính scale/zp ngay
- **Activation**: thay đổi theo input → phải **chạy nhiều batch** để estimate phân phối

```
Batch 1:  activation range = [-0.3, 0.8]
Batch 2:  activation range = [-0.2, 1.1]   ← max mới!
Batch 3:  activation range = [-0.5, 0.9]   ← min mới!
...
Sau N batch: running_min = -0.5, running_max = 1.1
           → scale = (1.1 - (-0.5)) / 255 ≈ 0.00627
```

---

## 8. Kiến trúc from-scratch trong project

### 8.1 Cây module

```
model.py — KHÔNG dùng torch.ao.quantization
│
├── FakeQuantizeFunction      ← torch.autograd.Function (STE)
│     Implement quantize + dequantize trong forward
│     Implement STE trong backward
│
├── MinMaxObserver            ← nn.Module
│     Theo dõi running min/max
│     Tính scale, zero_point
│
├── FakeQuantizeModule        ← nn.Module (bọc Function + Observer)
│     Training: observe → compute_qparams → fake_quantize
│     Eval:     compute_qparams → fake_quantize (không observe)
│
├── SimpleCNN                 ← nn.Module (baseline FP32)
│     Conv2d → ReLU → Pool → Conv2d → ReLU → Pool → FC → FC
│
├── QATConv2d                 ← nn.Module (thay thế Conv2d)
│     ├── activation_fake_quant: FakeQuantizeModule [0, 255]
│     ├── weight_fake_quant:     FakeQuantizeModule [-128, 127]
│     └── conv:                  nn.Conv2d
│
├── QATLinear                 ← nn.Module (thay thế Linear)
│     ├── activation_fake_quant: FakeQuantizeModule [0, 255]
│     ├── weight_fake_quant:     FakeQuantizeModule [-128, 127]
│     └── linear:                nn.Linear
│
└── QATCNN                    ← nn.Module (QAT version of SimpleCNN)
      Sử dụng QATConv2d/QATLinear thay cho Conv2d/Linear
      + load_pretrained()       ← map keys từ SimpleCNN
      + get_quantization_stats() ← in scale/zp cho giáo dục
```

### 8.2 Tại sao không dùng `torch.ao.quantization`?

| `torch.ao.quantization` | From-scratch (project này) |
|:---:|:---:|
| "Black box" — gọi API, không thấy bên trong | Thấy **mọi dòng code** quantize/dequantize |
| Observer, FakeQuantize là C++ internals | Observer, FakeQuantize là **Python thuần** |
| Khó debug & hiểu flow | **Đặt breakpoint**, print từng bước |
| Tốt cho production | Tốt cho **học tập** |

---

## 9. Giải thích code từng bước

### 9.1 `train_baseline.py` — baseline FP32

```
1. Load MNIST → normalize (mean=0.1307, std=0.3081)
2. SimpleCNN: Conv(1→32) → ReLU → Pool → Conv(32→64) → ReLU → Pool → FC(3136→128) → ReLU → FC(128→10)
3. Train 5 epochs, Adam(lr=1e-3)
4. Save state_dict → baseline_model.pth
5. Expected: ~99% accuracy
```

### 9.2 `train_qat.py` — QAT from scratch

```
┌──────────────────────────────────────────────────────────────────┐
│  PSEUDOCODE — TOÀN BỘ QAT PIPELINE:                             │
│                                                                  │
│  // 1. Tạo QATCNN (mỗi Conv2d/Linear → QATConv2d/QATLinear)    │
│  qat_model = QATCNN()                                           │
│                                                                  │
│  // 2. Load weight FP32 từ baseline                              │
│  qat_model.load_pretrained(baseline_state_dict)                  │
│  //    mapping: "conv1.weight" → "conv1.conv.weight"             │
│  //             "fc1.weight"   → "fc1.linear.weight"             │
│                                                                  │
│  // 3. Fine-tune 3 epochs                                        │
│  for epoch = 1 to 3:                                             │
│      for batch in train_data:                                    │
│          output = qat_model(batch)      // forward với FQ        │
│          loss = cross_entropy(output)                            │
│          loss.backward()                // STE backward          │
│          optimizer.step()               // cập nhật FP32 weight  │
│                                                                  │
│  // 4. In quantization stats                                     │
│  for each FakeQuantizeModule in model:                           │
│      print(layer_name, scale, zero_point, observed_min, max)     │
│                                                                  │
│  // 5. Save model                                                │
│  save(qat_model.state_dict())   // vẫn FP32 weights +           │
│                                  // quantization parameters      │
│                                  // (scale, zp) để convert sau  │
└──────────────────────────────────────────────────────────────────┘
```

### 9.3 `compare.py` — so sánh

So sánh baseline (SimpleCNN) và QAT (QATCNN): file size, accuracy, latency.

---

## 10. FAQ

### Q1: QAT model vẫn lưu FP32 — thì nhỏ hơn ở đâu?

Đúng — trong project này, `qat_model.pth` vẫn lưu FP32 vì ta chưa implement conversion thật sang INT8. Trong production:
1. Sau QAT, bạn **quantize weight thật** bằng scale/zp đã học
2. Lưu weight dưới dạng INT8 → file nhỏ ~4×
3. Dùng INT8 inference runtime (ONNX Runtime, TFLite, etc.)

Project này tập trung vào **quá trình training** (fake quantize + STE), không phải inference runtime.

### Q2: Tại sao activation dùng [0, 255] mà weight dùng [-128, 127]?

- **Activation sau ReLU**: luôn ≥ 0 → dùng **unsigned** INT8 `[0, 255]`
- **Weight**: có cả giá trị âm → dùng **signed** INT8 `[-128, 127]`

### Q3: Per-channel vs per-tensor quantization?

- **Per-tensor**: một cặp (scale, zp) cho **toàn bộ tensor** → đơn giản, project này dùng
- **Per-channel**: mỗi output channel có (scale, zp) riêng → chính xác hơn, production dùng

### Q4: Có cần pre-train baseline trước QAT không?

**Khuyến khích** nhưng không bắt buộc:
- Có pre-train: QAT chỉ cần vài epoch fine-tuning → nhanh
- Không pre-train: QAT phải train từ đầu → lâu hơn, nhưng vẫn hoạt động

---

## 11. Tài liệu tham khảo

### Papers
1. **Jacob et al. (2018)** — [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) — Paper gốc QAT từ Google
2. **Krishnamoorthi (2018)** — [Quantizing deep convolutional networks for efficient inference](https://arxiv.org/abs/1806.08342) — Tổng quan quantization

### Documentation
3. [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
4. [PyTorch QAT Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)

### Blog
5. [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
6. [Practical Quantization in PyTorch](https://pytorch.org/blog/quantization-in-practice/)
