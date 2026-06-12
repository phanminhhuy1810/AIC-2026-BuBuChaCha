"""
Bai 11: ASR — tu dong nhan dang giong noi tu video.

Video co tieng noi → Whisper nghe → ra text
→ Luu lai → user hoi "tim canh noi ve X" → tim trong text

Whisper = model cua OpenAI, ho tro 99 ngon ngu (ca tieng Viet).

CACH CHAY:
  pip install openai-whisper
  conda activate aic && python learn/asr1_demo.py

LAN DAU tai model ~1GB. Can ffmpeg (thuong Mac da co san).
"""
import os
import glob
import json
import time

print("Dang tai Whisper...")
import whisper

# =================================================================
# BUOC 1: TAI MODEL WHISPER
# =================================================================
# Cac model: tiny (39M) < base (74M) < small (244M) < medium (769M) < large (1550M)
# tiny:  nhanh, do chinh xac thap
# base:  can bang toc do vs chinh xac (dung cai nay)
# large: chinh xac nhat nhung can GPU

model = whisper.load_model("base")
print("OK! Whisper san sang.\n")

# =================================================================
# BUOC 2: TIM VIDEO
# =================================================================
video_dir = "data/videos"
video_files = []
for f in sorted(os.listdir(video_dir)):
    if f.endswith((".mp4", ".avi", ".mkv", ".mov")):
        video_files.append(os.path.join(video_dir, f))

if not video_files:
    print(f"KHONG TIM THAY VIDEO trong {video_dir}/")
    exit()

print(f"Tim thay {len(video_files)} video.")

# =================================================================
# BUOC 3: TRANSCRIBE (NGHE → TEXT)
# =================================================================
all_results = {}

for video_path in video_files:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\nDang nghe: {video_name}...")
    t0 = time.time()

    # 1 DONG = transcribe toan bo video
    result = model.transcribe(video_path)

    t1 = time.time()

    # result["text"] = toan bo text
    # result["segments"] = tung doan voi timestamp
    #   moi segment co: start, end, text

    print(f"  Thoi gian: {t1-t0:.1f}s")
    print(f"  Ngon ngu phat hien: {result['language']}")
    print(f"  So doan (segment): {len(result['segments'])}")

    # In vai doan dau
    print(f"\n  --- Vai doan dau ---")
    for seg in result["segments"][:5]:
        start = seg["start"]
        end = seg["end"]
        text = seg["text"].strip()
        print(f"  [{start:.1f}s - {end:.1f}s]: {text}")

    if len(result["segments"]) > 5:
        print(f"  ... va {len(result['segments']) - 5} doan nua")

    # Luu ket qua
    all_results[video_name] = {
        "language": result["language"],
        "full_text": result["text"],
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            }
            for seg in result["segments"]
        ]
    }

# =================================================================
# BUOC 4: LUU KET QUA
# =================================================================
asr_path = "data/asr_results.json"
with open(asr_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\nDa luu ket qua ASR vao: {asr_path}")
print()
print("=" * 50)
print("CACH DUNG TRONG HE THONG THI:")
print("  User hoi: 'tim canh noi ve Da Lat'")
print("  → Tim trong asr_results: segment nao co 'Da Lat'")
print("  → Lay timestamp (VD: 45.2s)")
print("  → Tim keyframe gan 45.2s nhat")
print("  → Tra ve keyframe do")
print("=" * 50)
