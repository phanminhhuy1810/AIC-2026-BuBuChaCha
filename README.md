# AIC-2026-BuBuChaCha — AI Challenge TP.HCM 2026 (Bảng A)

Hệ thống truy xuất thông tin trong dữ liệu lớn multimedia (ảnh / âm thanh / văn bản).
Thể thức theo LSC & VBS. Lõi: CLIP + FAISS + OCR + ASR + LLM.

## Thành viên
- Nguyễn Huỳnh Đệ
- Nguyễn Minh Anh
- Phan Minh Huy
- ... Anh Khoa
  
## Cài môi trường
Cần Anaconda. Tạo môi trường `aic` (Python 3.11):

```
conda create -n aic python=3.11 -y
conda activate aic
pip install -r requirements.txt
python verify_setup.py
```

Verify in ra "Tat ca SAN SANG" là OK. Việc nặng (embedding nhiều ảnh/video) chạy trên Google Colab GPU.

## Cấu trúc dự kiến
```
data/            # du lieu (KHONG commit)
preprocess/      # keyframe, OCR, ASR
embedding/       # CLIP -> vector
index/           # FAISS index
search/          # query + retrieval
ui/              # giao dien tim kiem
```

## Quy ước làm việc nhóm
- Mỗi người làm trên branch riêng, xong mở Pull Request.
- KHÔNG commit: model weights, dữ liệu, token/API key, môi trường ảo.
