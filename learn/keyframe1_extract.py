"""
Bai 8: Trich keyframe tu video.

Video = hang nghin frame/giay. Khong can het.
Chi lay 1 frame moi N giay → giam tu 54,000 frame con ~300 frame.

CACH DUNG:
  1. Bo file video (.mp4) vao data/videos/
  2. Chay:  conda activate aic && python learn/keyframe1_extract.py
  3. Keyframe xuat ra:  data/keyframes/

CAN CAI THEM:
  pip install opencv-python   (thuong da co trong aic env)
"""
import cv2
import os
import sys
import time

# =================================================================
# CAU HINH
# =================================================================
VIDEO_DIR = "data/videos"
OUTPUT_DIR = "data/keyframes"
INTERVAL_SEC = 2    # lay 1 frame moi 2 giay (doi tuy bai thi)

# =================================================================
# TIM VIDEO
# =================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

video_files = []
for f in sorted(os.listdir(VIDEO_DIR)):
    if f.endswith((".mp4", ".avi", ".mkv", ".mov")):
        video_files.append(os.path.join(VIDEO_DIR, f))

if not video_files:
    print(f"KHONG TIM THAY VIDEO trong {os.path.abspath(VIDEO_DIR)}/")
    print("Hay bo file .mp4 vao do roi chay lai.")
    sys.exit(1)

print(f"Tim thay {len(video_files)} video:")
for v in video_files:
    print(f"  - {v}")
print()

# =================================================================
# TRICH KEYFRAME
# =================================================================
# cv2.VideoCapture = doc video frame-by-frame
# Giong fopen() doc file tung dong trong C++

total_frames_saved = 0

for video_path in video_files:
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Mo video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"LOI: Khong mo duoc {video_path}")
        continue

    # Lay thong tin video
    fps = cap.get(cv2.CAP_PROP_FPS)                    # frame per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_name}")
    print(f"  FPS: {fps:.0f} | Tong frame: {total_frames:,} | Thoi luong: {duration:.1f}s")

    # Tinh: cu moi bao nhieu frame thi lay 1 keyframe
    frame_interval = int(fps * INTERVAL_SEC)
    # VD: fps=30, interval=2s => moi 60 frame lay 1 cai

    # Tao thu muc rieng cho video nay
    video_out_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(video_out_dir, exist_ok=True)

    frame_idx = 0
    saved = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        # ret = True/False (con frame khong)
        # frame = anh (numpy array, shape: H x W x 3)

        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Luu keyframe
            timestamp = frame_idx / fps
            filename = f"{video_name}_{frame_idx:06d}_{timestamp:.1f}s.jpg"
            filepath = os.path.join(video_out_dir, filename)

            cv2.imwrite(filepath, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    t1 = time.time()

    total_frames_saved += saved
    print(f"  Trich {saved} keyframe (moi {INTERVAL_SEC}s) -> {video_out_dir}/")
    print(f"  Thoi gian xu ly: {t1-t0:.1f}s")
    print()

# =================================================================
# KET QUA
# =================================================================
print("=" * 50)
print(f"TONG: {total_frames_saved} keyframe da luu vao {OUTPUT_DIR}/")
print()
print("BUOC TIEP THEO:")
print("  Dung CLIP encode cac keyframe nay thanh vector")
print("  Roi dua vao FAISS de tim kiem")
print("=" * 50)
