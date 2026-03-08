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
post_qat_convert.py — Xử lý sau QAT (Post-QAT Processing)

Mục đích:
    Sau lệnh train_qat.py, mô hình sinh ra (qat_model.pth) CHƯA THỰC SỰ
    giảm dung lượng, vì weights vẫn đang lưu dưới dạng FP32 (dù chúng đã
    bị hội tụ về các giá trị rời rạc thông qua fake quantize).

    Script này sẽ:
    1. Load weights FP32 từ `qat_model.pth`.
    2. Áp dụng scale & zero_point đã học để ÉP KIỂU (cast) các tensor này
       sang `torch.int8` (1 byte/param).
    3. Lưu lại dictionary mới chứa các tensor INT8 vào `qat_model_int8.pth`.
       => Đây mới là bước cắt giảm dung lượng ổ cứng.
    4. Cung cấp hàm load & chạy inference bằng cách giải nén (dequantize) on-the-fly.
"""

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import QATCNN

# ── Hằng số ──────────────────────────────────────────────────────────────────
QAT_PATH = "qat_model.pth"
INT8_PATH = "qat_model_int8.pth"
BATCH_SIZE = 128


# ── Post-QAT Conversion (FP32 -> INT8) ───────────────────────────────────────
def convert_to_int8(fp32_state_dict):
    """
    Biến đổi một state_dict QAT (chứa weights FP32 và các params Observer)
    thành một state_dict chỉ chứa weights INT8.
    """
    int8_state_dict = {}

    # Chúng ta quét qua các cặp module trong QATCNN.
    # Trong model.py, ta có:
    # - {layer}.conv.weight (FP32 gốc)
    # - {layer}.weight_fake_quant.observer.running_min / running_max
    # Ta phải lấy min/max này, tính scale/zp, rồi ròng qua công thức quantize.

    # Các layer cần convert
    layers = ["conv1", "conv2", "fc1", "fc2"]

    for layer in layers:
        # Lấy FP32 weight
        if f"{layer}.conv.weight" in fp32_state_dict:
            weight = fp32_state_dict[f"{layer}.conv.weight"]
            bias = fp32_state_dict.get(f"{layer}.conv.bias", None)
            base_name = f"{layer}.conv"
        elif f"{layer}.linear.weight" in fp32_state_dict:
            weight = fp32_state_dict[f"{layer}.linear.weight"]
            bias = fp32_state_dict.get(f"{layer}.linear.bias", None)
            base_name = f"{layer}.linear"
        else:
            continue

        # Lấy Observer params để tính scale/zp
        rmin = fp32_state_dict[f"{layer}.weight_fake_quant.observer.running_min"]
        rmax = fp32_state_dict[f"{layer}.weight_fake_quant.observer.running_max"]

        q_min, q_max = -128, 127
        scale = (rmax - rmin) / (q_max - q_min)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = q_min - torch.round(rmin / scale)
        zero_point = torch.clamp(zero_point, q_min, q_max)

        # -------------------------------------------------------------
        # ✅ ÉP KIỂU SANG INT8: x_int8 = round(x_fp32 / scale) + zp
        # -------------------------------------------------------------
        w_int = torch.round(weight / scale) + zero_point
        w_int = torch.clamp(w_int, q_min, q_max)
        w_int8 = w_int.to(torch.int8)  # ÉP KIỂU Ở ĐÂY! (từ 4 bytes xuống 1 byte)

        # Lưu vào dict gọn gàng
        int8_state_dict[f"{layer}.weight"] = w_int8
        int8_state_dict[f"{layer}.scale"] = scale
        int8_state_dict[f"{layer}.zero_point"] = zero_point

        # Bias thường không quantize (lưu ở FP32/FP16)
        if bias is not None:
            int8_state_dict[f"{layer}.bias"] = bias

    return int8_state_dict


# ── Inference bằng cách On-the-fly Dequantize ───────────────────────────────
class RealInt8CNN(torch.nn.Module):
    """
    Một model cực mỏng nhẹ dùng để chứa weights INT8.
    Hiện tại PyTorch eager mode thiếu native ops cho `int8 x int8` (nếu không gọi C++ backend),
    nên trong lúc inference ta đành: load int8 -> giải nén thành fp32 -> tính toán.
    Cái này tiết kiệm RAM lưu trữ (Storage), nhưng thời gian chạy (Compute) thì không
    tăng tốc vì ta bù lại bằng bước giải nén.
    """

    def __init__(self, state_dict):
        super().__init__()
        # Load các tensor int8 vào attribute
        for k, v in state_dict.items():
            self.register_buffer(k.replace(".", "_"), v)

    def _dequantize(self, prefix):
        w_int8 = getattr(self, f"{prefix}_weight")
        scale = getattr(self, f"{prefix}_scale")
        zp = getattr(self, f"{prefix}_zero_point")
        # fp32 = (int8 - zp) * scale
        w_fp32 = (w_int8.to(torch.float32) - zp) * scale
        return w_fp32

    def forward(self, x):
        # Biến đổi layer Conv1
        w1 = self._dequantize("conv1")
        b1 = getattr(self, "conv1_bias", None)
        x = F.conv2d(x, w1, b1, stride=1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Conv2
        w2 = self._dequantize("conv2")
        b2 = getattr(self, "conv2_bias", None)
        x = F.conv2d(x, w2, b2, stride=1, padding=1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)

        # FC1
        w3 = self._dequantize("fc1")
        b3 = getattr(self, "fc1_bias", None)
        x = F.linear(x, w3, b3)
        x = F.relu(x)

        # FC2
        w4 = self._dequantize("fc2")
        b4 = getattr(self, "fc2_bias", None)
        x = F.linear(x, w4, b4)

        return x


# ── Code chính ───────────────────────────────────────────────────────────────
def get_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    return DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def main():
    print("=" * 70)
    print("  Post-QAT: Ép kiểu weights FP32 sang INT8 (Tiết kiệm ổ cứng)")
    print("=" * 70)

    if not os.path.exists(QAT_PATH):
        print(f"❌ Không tìm thấy '{QAT_PATH}'. Trở về train_qat.py.")
        return

    # 1. Đọc weights FP32 (bản QAT).
    fp32_state_dict = torch.load(QAT_PATH, map_location="cpu", weights_only=True)
    size_before = os.path.getsize(QAT_PATH) / 1024
    print(f"📦 Kích thước file (Fake-QAT/FP32):\t{size_before:.1f} KB")

    # 2. Xử lý convert
    int8_state_dict = convert_to_int8(fp32_state_dict)

    # 3. Lưu xuống đĩa
    torch.save(int8_state_dict, INT8_PATH)
    size_after = os.path.getsize(INT8_PATH) / 1024
    print(f"📦 Kích thước file sau ép (Real-INT8):\t{size_after:.1f} KB")
    print("-" * 60)
    print(f"📉 Tỷ lệ tối ưu: Giảm {size_before / size_after:.1f} lần!")
    print("-" * 60)

    # Khởi tạo mô hình gọn nhẹ chuyên load int8.
    print("⚙️  Chạy thử inference với mô hình INT8 (Giải nén on-the-fly)...")
    int8_model = RealInt8CNN(int8_state_dict)

    test_loader = get_test_loader()
    accuracy = evaluate(int8_model, test_loader)
    print(f"🎯 Độ chính xác Test: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
