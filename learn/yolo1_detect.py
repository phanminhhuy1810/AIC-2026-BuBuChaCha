"""
Bai 13: YOLO — phat hien vat the trong anh (Object Detection).

CLIP:  "anh nay GIONG canh gi?" (toan cuc)
YOLO:  "anh nay CO GI trong do?" (tung vat the cu the)

Vi du:
  CLIP:  "a busy street" → tim canh duong pho
  YOLO:  "3 nguoi + 2 xe may + 1 o to" → tim frame co dung so luong do

CACH CHAY:
  pip install ultralytics
  conda activate aic && python learn/yolo1_detect.py
"""
import os
import glob
import json
import time
from ultralytics import YOLO
from collections import Counter

# =================================================================
# BUOC 1: TAI MODEL YOLO
# =================================================================
# YOLOv8n = phien ban nho (nano), nhanh, du chinh xac
# Phat hien 80 loai vat the: person, car, dog, cat, chair, ...

print("Dang tai YOLOv8...")
model = YOLO("yolov8n.pt")  # tu dong tai lan dau (~6MB)
print("OK!\n")

# =================================================================
# BUOC 2: TIM KEYFRAME
# =================================================================
keyframe_dir = "data/keyframes"
image_paths = sorted(glob.glob(f"{keyframe_dir}/**/*.jpg", recursive=True))

if not image_paths:
    print("KHONG TIM THAY KEYFRAME. Chay keyframe1_extract.py truoc.")
    exit()

print(f"Tim thay {len(image_paths)} keyframe. Dang detect...\n")

# =================================================================
# BUOC 3: DETECT TUNG KEYFRAME
# =================================================================
# Ket qua: dict {ten_file: {ten_vat_the: so_luong}}
# VD: {"frame_001.jpg": {"person": 3, "car": 2, "motorcycle": 1}}

detect_results = {}
t0 = time.time()

for i, path in enumerate(image_paths):
    name = os.path.basename(path)

    # YOLO detect — 1 dong
    results = model(path, verbose=False)

    # Dem so luong tung loai vat the
    counts = Counter()
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]  # "person", "car", ...
        confidence = float(box.conf[0])
        if confidence > 0.3:  # chi lay do tin cay > 30%
            counts[class_name] += 1

    if counts:
        detect_results[name] = dict(counts)

    if (i + 1) % 10 == 0 or i == len(image_paths) - 1:
        print(f"  [{i+1}/{len(image_paths)}] Da xu ly...")

t1 = time.time()

# =================================================================
# BUOC 4: KET QUA
# =================================================================
print(f"\nXong! {t1-t0:.1f}s")
print(f"  {len(detect_results)}/{len(image_paths)} keyframe co vat the\n")

# Thong ke tong
all_objects = Counter()
for counts in detect_results.values():
    for obj, count in counts.items():
        all_objects[obj] += count

print("--- Tong so vat the phat hien ---")
for obj, count in all_objects.most_common(15):
    print(f"  {obj:20s}: {count}")

# In mau vai keyframe
print("\n--- Mau vai keyframe ---")
count = 0
for name, objects in detect_results.items():
    obj_str = ", ".join(f"{v} {k}" for k, v in objects.items())
    print(f"  {name}: {obj_str}")
    count += 1
    if count >= 5:
        print(f"  ... va {len(detect_results) - 5} keyframe khac")
        break

# =================================================================
# BUOC 5: LUU KET QUA
# =================================================================
detect_path = "data/detect_results.json"
with open(detect_path, "w", encoding="utf-8") as f:
    json.dump(detect_results, f, ensure_ascii=False, indent=2)

print(f"\nDa luu vao: {detect_path}")
print()
print("=" * 50)
print("CACH DUNG TRONG HE THONG THI:")
print("  User hoi: 'tim frame co 3 nguoi va 1 xe may'")
print("  → Tim trong detect_results:")
print("    keyframe nao co person >= 3 va motorcycle >= 1")
print("  → Tra ve keyframe do")
print("=" * 50)
