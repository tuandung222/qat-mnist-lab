# Chứng minh Toán học: Cơ chế Lượng tử hoá (Quantization Math Proof)

Tài liệu này trình bày các bước suy luận toán học đằng sau hai công thức cốt lõi trong Quantization-Aware Training (QAT) và Post-Training Quantization (PTQ): **Quantize** (Lượng tử hoá) và **Dequantize** (Giải lượng tử).

---

## 1. Bài toán đặt ra

Ta cần ánh xạ (map) một dải số thực (floating-point) $x_{real} \in [min\_val, max\_val]$ sang một dải số nguyên (integer) $x_{int} \in [q\_min, q\_max]$.

Ví dụ với INT8:
- Unsigned INT8: $q\_min = 0$, $q\_max = 255$
- Signed INT8: $q\_min = -128$, $q\_max = 127$

Vì đây là **Lượng tử hoá tuyến tính (Linear Quantization)** hay *Affine Quantization*, phép biến đổi có dạng một phương trình đường thẳng:
$$ x_{int} \approx a \cdot x_{real} + b $$

Nhiệm vụ của chúng ta là tìm các hằng số $a$ và $b$.

---

## 2. Tìm hệ số góc ($a$) và định nghĩa `scale`

Để dải số thực ánh xạ khít vào dải số nguyên, hai điểm đầu mút phải tương ứng với nhau:
1. Giá trị nhỏ nhất: $q\_min = a \cdot min\_val + b \quad (1)$
2. Giá trị lớn nhất: $q\_max = a \cdot max\_val + b \quad (2)$

Lấy phương trình (2) trừ phương trình (1):
$$ q\_max - q\_min = a \cdot (max\_val - min\_val) $$
$$ \Rightarrow a = \frac{q\_max - q\_min}{max\_val - min\_val} $$

Trong Quantization, người ta thường dùng nghịch đảo của $a$, gọi là **`scale`** (bước nhảy lượng tử). Khái niệm lượng tử có nghĩa là "mỗi 1 đơn vị số nguyên tương đương với bao nhiêu đơn vị số thực".
$$ scale = \frac{1}{a} = \frac{max\_val - min\_val}{q\_max - q\_min} $$

Thay $a = \frac{1}{scale}$ vào lại phương trình biến đổi ban đầu, ta có:
$$ x_{int} \approx \frac{x_{real}}{scale} + b $$

---

## 3. Tìm tung độ gốc ($b$) và định nghĩa `zero_point`

Từ phương trình (1), ta thay $a = \frac{1}{scale}$ vào để tìm $b$:
$$ q\_min = \frac{min\_val}{scale} + b $$
$$ \Rightarrow b = q\_min - \frac{min\_val}{scale} $$

Theo định nghĩa, **`zero_point`** chính là giá trị nguyên $x_{int}$ đại diện cho số thực $0.0$.
Nếu ta thế $x_{real} = 0.0$ vào phương trình $x_{int} = a \cdot x_{real} + b$, ta sẽ ra $x_{int} = b$. Vậy $b$ đóng vai trò là `zero_point`.

Tuy nhiên, định dạng số mà phần cứng tính toán là số nguyên (INT8), nên `zero_point` cũng bắt buộc phải là một số nguyên. Do đó, ta phải làm tròn (round) giá trị của $b$:
$$ zero\_point = \text{round}(b) = \text{round}\left( q\_min - \frac{min\_val}{scale} \right) $$
*(Do $q\_min$ đã là số nguyên, ta có thể đưa nó ra ngoài hàm làm tròn)*:
$$ zero\_point = q\_min - \text{round}\left( \frac{min\_val}{scale} \right) $$

*(Lưu ý: Sau khi tính xong, `zero_point` thường được kẹp chặn (clamp) để đảm bảo nó không vượt quá dải $[q\_min, q\_max]$ nhằm xử lý trường hợp $0.0$ nằm hoàn toàn ngoài dải quan sát).*

---

## 4. Công thức Quantize (Giai đoạn Forward)

Bây giờ ta thay `scale` và `zero_point` (thay thế cho $b$) vào phương trình biến đổi:
$$ x_{int} \approx \frac{x_{real}}{scale} + zero\_point $$

Vì $x_{int}$ phải là số nguyên, ta thực hiện phép làm tròn gần nhất (Round-to-Nearest):
$$ x_{int} = \text{round}\left( \frac{x_{real}}{scale} + zero\_point \right) $$

**Cắt gọt (Clamping / Clipping):**
Để ngăn chặn các giá trị thực bất thường (outliers) đẩy $x_{int}$ vượt khỏi giới hạn biểu diễn của kiểu dữ liệu (ví dụ: tràn số trên 255 hoặc dưới 0 đối với INT8), ta dùng hàm `clamp`:
$$ x_{int\_final} = \text{clamp}\left( x_{int}, \ q\_min, \ q\_max \right) $$

**📍 KẾT LUẬN CÔNG THỨC QUANTIZE:**
$$ \mathbf{x_{int} = \text{clamp}\left( \text{round}\left( \frac{x_{real}}{scale} \right) + zero\_point, \ q\_min, \ q\_max \right)} $$

---

## 5. Công thức Dequantize (Giai đoạn Tính toán / Backward)

Dequantize là thao tác khôi phục lại giá trị gần đúng của $x_{real}$ từ số nguyên $x_{int}$ đã được lưu ở trong bộ nhớ.

Ta đi ngược lại từ phương trình tuyến tính vế trên (bỏ qua hàm round và clamp do không thể đảo nghịch chính xác):
$$ x_{int} = \frac{x_{real}}{scale} + zero\_point $$
$$ \Rightarrow \frac{x_{real}}{scale} = x_{int} - zero\_point $$
$$ \Rightarrow x_{real} \approx (x_{int} - zero\_point) \times scale $$

Việc dùng $\approx$ là do ta đã đánh mất thông tin lượng tử hóa tại bước `round()`. Sự sai lệch này chính là **Quantization Error** (Nhiễu lượng tử).

**📍 KẾT LUẬN CÔNG THỨC DEQUANTIZE:**
$$ \mathbf{x_{real} \approx (x_{int} - zero\_point) \times scale} $$

---

## Tóm tắt Hệ Phương Trình

| Thành phần | Công thức Toán học | Ý nghĩa Vật lý |
| :--- | :--- | :--- |
| **`scale`** | $\frac{max\_val - min\_val}{q\_max - q\_min}$ | Độ phân giải của 1 bit nguyên so với số thực. |
| **`zero_point`** | $q\_min - \text{round}(\frac{min\_val}{scale})$ | Tọa độ INT8 nơi giá trị thực bằng đúng $0.0$. |
| **`Quantize`** | $\text{clamp}(\text{round}(\frac{x_{real}}{scale}) + zp, \dots)$ | Nén weight/activation vào file siêu nhỏ. |
| **`Dequantize`** | $(x_{int} - zero\_point) \times scale$ | Khôi phục số thực khi Inference on-the-fly. |
