"""
Bai 10: OCR — doc chu trong anh.

TAI SAO CAN?
  CLIP gioi tim "a cat on a chair" nhung TE khi tim chu cu the:
  "tim anh co chu TOYOTA" → CLIP khong doc duoc chu
  → Can OCR (Optical Character Recognition) doc chu trong anh
  → Index chu do → search bang text match

CACH DUNG:
  1. Cai:  pip install easyocr
  2. Chay: conda activate aic && python learn/ocr1_demo.py
"""
import os
import glob
import time

print("Dang tai EasyOCR...")
import easyocr

# =================================================================
# BUOC 1: TAO OCR READER
# =================================================================
# EasyOCR ho tro nhieu ngon ngu. Lan dau chay tai model (~100MB)
reader = easyocr.Reader(['en', 'vi'], gpu=False)
print("OK! OCR san sang.\n")

# =================================================================
# BUOC 2: TIM KEYFRAME
# =================================================================
keyframe_dir = "data/keyframes"
image_paths = sorted(glob.glob(f"{keyframe_dir}/**/*.jpg", recursive=True))

if not image_paths:
    print("KHONG TIM THAY KEYFRAME. Chay keyframe1_extract.py truoc.")
    exit()

print(f"Tim thay {len(image_paths)} keyframe. Dang doc chu...\n")

# =================================================================
# BUOC 3: DOC CHU TRONG TUNG KEYFRAME
# =================================================================
# Ket qua: dict {ten_file: [list cac chu doc duoc]}
ocr_results = {}
total_texts = 0
t0 = time.time()

for i, path in enumerate(image_paths):
    name = os.path.basename(path)

    # readtext tra ve list cac vung chu:
    # [(bbox, text, confidence), ...]
    # bbox = 4 goc cua vung chu
    # text = noi dung doc duoc
    # confidence = do tin cay (0-1)
    results = reader.readtext(path)

    texts = []
    for (bbox, text, conf) in results:
        if conf > 0.3:  # chi lay chu co do tin cay > 30%
            texts.append(text)

    if texts:
        ocr_results[name] = texts
        total_texts += len(texts)

    # In tien trinh
    if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
        print(f"  [{i+1}/{len(image_paths)}] Da xu ly...")

t1 = time.time()

# =================================================================
# BUOC 4: KET QUA
# =================================================================
print(f"\nXong! {t1-t0:.1f}s")
print(f"  {len(ocr_results)}/{len(image_paths)} keyframe co chu")
print(f"  Tong: {total_texts} doan chu doc duoc\n")

# In mau vai keyframe co chu
print("--- Mau vai keyframe co chu ---")
count = 0
for name, texts in ocr_results.items():
    print(f"  {name}:")
    for t in texts:
        print(f"    \"{t}\"")
    count += 1
    if count >= 5:
        print(f"  ... va {len(ocr_results) - 5} keyframe khac co chu")
        break

# =================================================================
# BUOC 5: LUU KET QUA
# =================================================================
import json

ocr_path = "data/ocr_results.json"
with open(ocr_path, "w", encoding="utf-8") as f:
    json.dump(ocr_results, f, ensure_ascii=False, indent=2)

print(f"\nDa luu ket qua OCR vao: {ocr_path}")
print()
print("=" * 50)
print("CACH DUNG TRONG HE THONG THI:")
print("  User hoi: 'tim anh co chu TOYOTA'")
print("  → Khong dung CLIP (CLIP khong doc duoc chu)")
print("  → Tim trong ocr_results: keyframe nao co 'TOYOTA'")
print("  → Tra ve keyframe do")
print("=" * 50)
