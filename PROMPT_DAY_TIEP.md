# PROMPT — Dán toàn bộ nội dung dưới đây vào Claude chat để dạy tiếp

---

## 0. Cách trả lời tôi mong muốn
- Giải thích bằng **tiếng Việt đơn giản**, câu ngắn, dễ hiểu.
- **Tên biến / hàm / thuật ngữ kỹ thuật để bằng tiếng Anh.**
- Tôi xuất thân **code C++** (rành DSA, OOP, toán đại số tuyến tính + giải tích), mới chuyển qua Python cho AI → giải thích kỹ chỗ khác biệt Python.
- Ngắn gọn, đi thẳng vào việc. Cho code ví dụ chạy được luôn.

---

## 1. Tôi là ai & cuộc thi
- Sinh viên, thi **AI Challenge TP.HCM 2026 — Bảng A**.
- Đội 4 người. Đầu tư 30+ giờ/tuần.
- **Đề thi:** Trợ lý ảo truy xuất thông tin multimedia (ảnh/âm thanh/văn bản). Bản chất = **Information Retrieval**: nhập câu mô tả → tìm đúng khung hình trong kho dữ liệu lớn.
- Lõi hệ thống: **CLIP** (biến ảnh + text thành vector) + **FAISS** (tìm vector gần nhất bằng cosine similarity).
- Sơ tuyển tháng 8, chung kết tháng 9. **Giữa tháng 8 phải có hệ thống end-to-end.**

---

## 2. Lộ trình 5 giai đoạn
- **GĐ0 ✅ XONG** — Môi trường conda `aic` (Python 3.11, PyTorch, CLIP, FAISS). Repo GitHub đã push.
- **GĐ1 (đang học) — Nền tảng ML/DL:** ML cơ bản, gradient descent, neural network, CNN.
- **GĐ2 — CV + NLP + Multimodal:** CLIP, ViT, Transformer, FAISS. Mini-project: text-to-image search.
- **GĐ3 — Xây hệ thống:** keyframe extraction + CLIP + FAISS + OCR + ASR + UI + DRES.
- **GĐ4 — Tối ưu & chung kết.**

---

## 3. Những gì tôi ĐÃ HỌC

### Bài 1 — ML là gì + tư duy cốt lõi
- **North star:** Cả hệ thống thi = biến ảnh & text thành vector → tìm vector gần nhất (cosine similarity). CLIP sinh vector, FAISS tìm nhanh.
- **ML vs lập trình thường:** lập trình thường = bạn viết luật. ML = đưa data, máy tự tìm hàm.
- **Model = hàm có parameters.** VD: `y = w*x + b`, máy tìm w, b.
- **Loss function** = thước đo sai (MSE). Train = giảm loss.
- **Gradient Descent** = lặp: tính gradient → đi ngược → loss giảm dần. Learning rate = bước đi.

### Bài 2 — Gradient descent bằng Python thuần
- Code linear regression (giá nhà theo diện tích) bằng vòng lặp + phép toán. Không thư viện.

### Bài 3 — NumPy + PyTorch
- **NumPy array** = mảng C chạy bên dưới, phép tính trên cả mảng không cần vòng lặp (vector hoá). Nhanh gấp 10x+ so với list.
- **PyTorch tensor** = numpy array + GPU + autograd.
- `requires_grad=True` = bật theo dõi đạo hàm cho parameter.
- `loss.backward()` = tính tất cả gradient tự động.
- `torch.no_grad()` = tắt ghi khi cập nhật parameter.
- Cú pháp numpy ↔ pytorch gần y hệt.

### Bài 4 — Neural Network (nn.Module, MNIST)
- `nn.Module`: class chứa model, override `forward()`. Giống kế thừa class trong C++.
- `nn.Linear(in, out)`: 1 layer tuyến tính (nhân ma trận + bias).
- `nn.ReLU()`: activation phi tuyến = `max(0, x)`. Không có activation thì xếp bao nhiêu Linear cũng vẫn tuyến tính.
- `DataLoader`: chia data thành batch 64 ảnh → mini-batch gradient descent.
- `CrossEntropyLoss`: loss cho phân loại nhiều nhãn.
- `Adam optimizer`: gradient descent thông minh hơn (tự chỉnh lr).
- Train loop chuẩn: `forward → loss → zero_grad → backward → step`.
- Kết quả MNIST: **96.8% accuracy** với model 109K params, 3 epoch.
- Giới hạn: Linear NN làm phẳng ảnh 28×28 → 784 số → mất thông tin vị trí pixel.

### Bài 5 — CNN (đang chạy thử)
- Đã có file `learn/nn2_cnn_mnist.py`. Đang chạy.
- `Conv2d`: bộ lọc 3×3 trượt qua ảnh, phát hiện đặc trưng cục bộ (cạnh, góc, đường cong).
- `MaxPool2d`: thu nhỏ ảnh 2x, giữ đặc trưng mạnh nhất.
- Dự kiến accuracy ~99%.

---

## 4. DẠY TIẾP TỪ ĐÂY

Nếu tôi đã chạy xong CNN (bài 5), hãy:
1. Giải thích kết quả CNN so với Linear NN
2. Dạy **Bài 6: Transfer Learning + CIFAR-10** — dùng model pretrained (ResNet) thay vì train từ đầu. Đây là cầu nối sang GĐ2 (CLIP cũng là pretrained model).
3. Sau đó chuyển sang **GĐ2**: CLIP, ViT, Transformer, FAISS, mini-project text-to-image search.

Nếu tôi chưa chạy CNN, hãy giải thích Conv2d, MaxPool2d, và cấu trúc CNN trước.
