# Bài 17: Query Strategy — mẹo viết query cho CLIP

## Tại sao quan trọng?
Khi thi, mỗi câu hỏi chỉ có ~5 phút. Viết query đúng = tìm ra ngay.
Viết query sai = mất 5 phút mò trong vô vọng.

---

## 6 QUY TẮC VÀNG

### 1. Viết như CAPTION của ảnh, không phải câu hỏi
CLIP được train trên cặp (ảnh + caption mô tả). Nó hiểu CÂU MÔ TẢ, không hiểu CÂU HỎI.

- ❌ SAI:  `where is the man cooking?`
- ✅ ĐÚNG: `a man cooking food in a kitchen`

### 2. Bắt đầu bằng "a photo of" hoặc "a/an"
Caption trên mạng thường bắt đầu kiểu này → CLIP quen thuộc.

- OK:    `man riding motorbike`
- TỐT:   `a man riding a motorbike`
- TỐT+:  `a photo of a man riding a motorbike on a busy street`

### 3. Mô tả ĐIỀU NHÌN THẤY, không phải điều suy ra
CLIP chỉ "nhìn" ảnh — không biết ngữ cảnh, tên riêng, địa danh.

- ❌ SAI:  `Ho Chi Minh city traffic` (CLIP không biết đó là HCM)
- ✅ ĐÚNG: `crowded street with many motorbikes in Vietnam`

- ❌ SAI:  `a sad woman` (cảm xúc khó thấy)
- ✅ ĐÚNG: `a woman crying, covering her face`

### 4. Thêm MÀU SẮC + VỊ TRÍ + SỐ LƯỢNG
Càng cụ thể càng lọc được kết quả tốt.

- Mơ hồ:  `a person with a hat`
- Cụ thể: `an old man wearing a green conical hat sitting on a red plastic chair`

### 5. Query NGẮN khi tìm rộng, DÀI khi lọc hẹp
- Bước 1 (tìm rộng): `a food market` → xem top 20
- Bước 2 (lọc hẹp):  `a woman selling fruits at an outdoor market, yellow umbrella`

### 6. Nếu không ra → ĐỔI GÓC NHÌN, đừng lặp lại
Cùng 1 cảnh có nhiều cách mô tả. CLIP có thể "thích" cách này hơn cách kia.

Cảnh: người bán phở ngoài đường
- Thử 1: `a street food vendor`
- Thử 2: `a person cooking noodle soup at a food stall`
- Thử 3: `steaming pot of soup, plastic tables on sidewalk`

---

## CÔNG THỨC TỔNG QUÁT

```
a photo of [SỐ LƯỢNG] [CHỦ THỂ] [HÀNH ĐỘNG] [VỊ TRÍ/BỐI CẢNH], [CHI TIẾT MÀU SẮC/ĐẶC ĐIỂM]
```

Ví dụ:
- `a photo of two children eating at a dining table, white bowls`
- `a photo of a yellow motorbike parked on a sidewalk`
- `a photo of green rice terraces with mountains in the background`

---

## KẾT HỢP CÁC MODE (chiến thuật thi đấu)

| Câu hỏi của giám khảo | Mode nên dùng |
|---|---|
| "Tìm cảnh X" (mô tả chung) | CLIP |
| "Tìm cảnh có chữ/biển hiệu Y" | OCR |
| "Tìm cảnh người nói về Z" | ASR |
| "Tìm cảnh có đúng N người/vật" | YOLO |
| "Tìm chuỗi: A rồi B rồi C" | Temporal |
| "Câu hỏi chi tiết khó" | CLIP lọc top 20 → VQA xác nhận |

**Mẹo phối hợp:** CLIP tìm rộng trước → YOLO/OCR lọc lại → mất ít thời gian nhất.

---

## TỪ VỰNG HAY DÙNG KHI THI (học thuộc!)

**Người:** man, woman, child, elderly person, crowd, vendor (người bán), pedestrian (người đi bộ)

**Hành động:** walking, riding, cooking, eating, selling, carrying, holding, pointing, waving

**Vị trí:** in the foreground (phía trước), in the background (phía sau), on the left/right, next to, behind

**Bối cảnh:** street, market, kitchen, riverside, rice field, temple, beach, indoor, outdoor

**Màu + vật:** conical hat (nón lá), plastic chair, motorbike, umbrella, basket, street stall

---

## BÀI TẬP THỬ NGAY

Mở UI lên, thử các query này với mode CLIP:
1. `a photo of a family eating dinner together at a table`
2. `a yellow motorcycle parked outside`
3. `two children sitting at a table`

Rồi tự viết query tìm 1 cảnh bất kỳ trong video — nếu top 5 có cảnh đó = đạt!
