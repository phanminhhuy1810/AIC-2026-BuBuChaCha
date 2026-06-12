"""
Bai 15: Object Tracking — theo doi vat the qua nhieu frame.

VAN DE:
  YOLO: "frame nay co 3 nguoi" (moi frame doc lap)
  Tracking: "nguoi #1 xuat hien tu frame 10 den frame 50"
           "co 5 nguoi KHAC NHAU trong video"

  YOLO chi dem, Tracking theo doi AI CU THE

SO SANH:
  YOLO     = "co bao nhieu nguoi trong anh?"
  Tracking = "nguoi ao do di tu trai sang phai"

CACH LAM (ByteTrack):
  1. YOLO detect tung frame → danh sach box
  2. ByteTrack ghep box giua cac frame:
     - Frame 1: box A o (100, 200)
     - Frame 2: box B o (105, 202)  ← gan A → cung 1 nguoi!
     - Frame 3: box C o (110, 205)  ← van nguoi do
  3. Moi nguoi co 1 track_id duy nhat

CACH CHAY:
  pip install supervision
  conda activate aic && python learn/track1_bytetrack.py
"""
import os
import glob
import json
import time
import numpy as np
from ultralytics import YOLO

# supervision = thu vien ho tro tracking, ve box, annotation
# ByteTrack co san trong supervision
try:
    import supervision as sv
except ImportError:
    print("Can cai supervision:")
    print("  pip install supervision")
    exit()

# =================================================================
# BUOC 1: TAI MODEL + TIM KEYFRAME
# =================================================================
print("Dang tai YOLOv8...")
model = YOLO("yolov8n.pt")
print("OK!\n")

keyframe_dir = "data/keyframes"
image_paths = sorted(glob.glob(f"{keyframe_dir}/**/*.jpg", recursive=True))

if not image_paths:
    print("KHONG TIM THAY KEYFRAME. Chay keyframe1_extract.py truoc.")
    exit()

print(f"Tim thay {len(image_paths)} keyframe.\n")

# =================================================================
# BUOC 2: TRACKING VOI BYTETRACK
# =================================================================
# ByteTrack: moi frame YOLO detect → ByteTrack ghep thanh track
# Moi track = 1 vat the di chuyen qua nhieu frame

tracker = sv.ByteTrack()

# Luu ket qua: {track_id: [frame_indices]}
track_history = {}
t0 = time.time()

print("Dang tracking...")
for i, path in enumerate(image_paths):
    # YOLO detect
    results = model(path, verbose=False)
    detections = sv.Detections.from_ultralytics(results[0])

    # ByteTrack update — ghep detection voi track hien co
    tracked = tracker.update_with_detections(detections)

    # Luu lai track_id xuat hien o frame nay
    if tracked.tracker_id is not None:
        for tid in tracked.tracker_id:
            tid = int(tid)
            if tid not in track_history:
                track_history[tid] = []
            track_history[tid].append(i)

    if (i + 1) % 20 == 0 or i == len(image_paths) - 1:
        print(f"  [{i+1}/{len(image_paths)}] Tracks hien tai: {len(track_history)}")

t1 = time.time()

# =================================================================
# BUOC 3: PHAN TICH KET QUA
# =================================================================
print(f"\nXong! {t1-t0:.1f}s")
print(f"Tong so track (vat the duy nhat): {len(track_history)}")

# Loc track co it nhat 2 frame (vat the thuc su di chuyen)
moving_tracks = {
    tid: frames for tid, frames in track_history.items()
    if len(frames) >= 2
}
print(f"Track xuat hien >= 2 frame: {len(moving_tracks)}")

# In top 10 track dai nhat
print("\n--- Top 10 track dai nhat ---")
sorted_tracks = sorted(moving_tracks.items(), key=lambda x: -len(x[1]))
for tid, frames in sorted_tracks[:10]:
    first_frame = os.path.basename(image_paths[frames[0]])
    last_frame = os.path.basename(image_paths[frames[-1]])
    print(f"  Track #{tid}: {len(frames)} frame ({first_frame} → {last_frame})")

# =================================================================
# BUOC 4: LUU KET QUA
# =================================================================
track_results = {}
for tid, frames in track_history.items():
    track_results[str(tid)] = {
        "num_frames": len(frames),
        "first_frame": os.path.basename(image_paths[frames[0]]),
        "last_frame": os.path.basename(image_paths[frames[-1]]),
        "frame_names": [os.path.basename(image_paths[f]) for f in frames]
    }

track_path = "data/track_results.json"
with open(track_path, "w", encoding="utf-8") as f:
    json.dump(track_results, f, ensure_ascii=False, indent=2)

print(f"\nDa luu vao: {track_path}")
print()
print("=" * 50)
print("CACH DUNG TRONG HE THONG THI:")
print("  User hoi: 'co bao nhieu nguoi KHAC NHAU trong video?'")
print("  → Dem so track co class=person")
print("  User hoi: 'nguoi nao xuat hien lau nhat?'")
print("  → Tim track co nhieu frame nhat")
print("  User hoi: 'theo doi nguoi ao do'")
print("  → Ket hop CLIP + tracking: tim track co dac diem 'red shirt'")
print("=" * 50)
