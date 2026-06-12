"""
Bai 14: Temporal Search — tim chuoi hanh dong theo thoi gian.

VAN DE:
  CLIP tim 1 canh rieng le: "a man cooking"
  Nhung cuoc thi hay hoi CHUOI hanh dong:
    "Tim video co: nguoi di cho → nau an → an com"
  → Can tim 3 canh THEO THU TU thoi gian trong cung 1 video

CACH LAM:
  1. Query = "A then B then C"
  2. Dung CLIP tim TOP frame cho tung buoc A, B, C
  3. Loc: chi giu nhung bo (frameA, frameB, frameC) ma
     thoi gian A < B < C (dung thu tu)

CACH CHAY:
  conda activate aic && python learn/temporal1_search.py
"""
import os
import glob
import json
import re
import torch
import numpy as np
import open_clip
import faiss
from PIL import Image

# =================================================================
# BUOC 1: TAI CLIP + ENCODE KEYFRAME
# =================================================================
print("Dang tai CLIP...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()
print("OK!\n")

keyframe_dir = "data/keyframes"
image_paths = sorted(glob.glob(f"{keyframe_dir}/**/*.jpg", recursive=True))

if not image_paths:
    print("KHONG TIM THAY KEYFRAME. Chay keyframe1_extract.py truoc.")
    exit()

print(f"Dang encode {len(image_paths)} keyframe...")

all_features = []
batch_size = 32
for i in range(0, len(image_paths), batch_size):
    batch = []
    for p in image_paths[i:i+batch_size]:
        img = Image.open(p).convert("RGB")
        batch.append(preprocess(img))
    batch_tensor = torch.stack(batch)
    with torch.no_grad():
        features = model.encode_image(batch_tensor)
    all_features.append(features)

features = torch.cat(all_features, dim=0)
features = features / features.norm(dim=1, keepdim=True)
features_np = features.numpy().astype(np.float32)

index = faiss.IndexFlatIP(features_np.shape[1])
index.add(features_np)
print(f"OK! {len(image_paths)} keyframe da encode.\n")

# =================================================================
# BUOC 2: PARSE TIMESTAMP TU TEN FILE
# =================================================================
# Ten file dang: test_000000_0.0s.jpg → lay 0.0
def get_timestamp(path):
    name = os.path.basename(path)
    parts = name.replace(".jpg", "").split("_")
    for part in parts:
        if part.endswith("s"):
            try:
                return float(part[:-1])
            except ValueError:
                pass
    return -1.0

timestamps = [get_timestamp(p) for p in image_paths]

# =================================================================
# BUOC 3: TEMPORAL SEARCH
# =================================================================
def clip_search(query_text, top_k=20):
    """Tim top_k frame giong query nhat."""
    tokens = tokenizer([query_text])
    with torch.no_grad():
        text_feat = model.encode_text(tokens)
    text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)
    text_np = text_feat.numpy().astype(np.float32)
    scores, ids = index.search(text_np, top_k)
    return [(ids[0][i], scores[0][i]) for i in range(top_k)]

def temporal_search(steps, top_per_step=20):
    """
    steps = ["a market scene", "cooking food", "eating"]
    Tim chuoi frame thoa man: thoi gian tang dan
    """
    print(f"Temporal search: {len(steps)} buoc")
    for i, s in enumerate(steps):
        print(f"  Buoc {i+1}: \"{s}\"")
    print()

    # Tim top frame cho tung buoc
    step_results = []
    for i, step_query in enumerate(steps):
        hits = clip_search(step_query, top_per_step)
        step_results.append(hits)
        top_name = os.path.basename(image_paths[hits[0][0]])
        print(f"  Buoc {i+1} top-1: {top_name} (score={hits[0][1]:.3f})")

    # Tim to hop thoa man thu tu thoi gian
    # Dung greedy: chon frame tot nhat cho buoc 1,
    # roi chon frame tot nhat cho buoc 2 co timestamp > buoc 1, ...
    print("\nDang tim chuoi thoi gian hop le...")

    best_chain = []
    best_total_score = -1

    # Thu nhieu diem bat dau khac nhau
    for start_idx, start_score in step_results[0][:10]:
        chain = [(start_idx, start_score)]
        current_time = timestamps[start_idx]

        for step_i in range(1, len(steps)):
            best_next = None
            best_next_score = -1
            for idx, score in step_results[step_i]:
                t = timestamps[idx]
                if t > current_time:
                    if score > best_next_score:
                        best_next = (idx, score)
                        best_next_score = score
            if best_next is None:
                break
            chain.append(best_next)
            current_time = timestamps[best_next[0]]

        if len(chain) == len(steps):
            total_score = sum(s for _, s in chain)
            if total_score > best_total_score:
                best_total_score = total_score
                best_chain = chain

    return best_chain

# =================================================================
# BUOC 4: DEMO
# =================================================================
print("\n" + "=" * 60)
print("DEMO TEMPORAL SEARCH")
print("=" * 60)

# Thu chuoi hanh dong trong video du lich Viet Nam
test_queries = [
    ["a busy street with traffic", "green rice terraces", "local food market"],
    ["people walking", "a temple or pagoda", "sunset scenery"],
]

for steps in test_queries:
    print(f"\n--- Query: {' → '.join(steps)} ---")
    chain = temporal_search(steps)

    if chain:
        print(f"\n  KET QUA (tong score = {sum(s for _, s in chain):.3f}):")
        for i, (idx, score) in enumerate(chain):
            name = os.path.basename(image_paths[idx])
            t = timestamps[idx]
            print(f"    Buoc {i+1}: {name} (t={t:.1f}s, score={score:.3f})")
    else:
        print("  KHONG TIM THAY chuoi hop le!")

print()
print("=" * 50)
print("CACH DUNG TRONG HE THONG THI:")
print("  User hoi: 'Tim video co nguoi di cho roi nau an roi an com'")
print("  → Tach thanh steps: ['market scene', 'cooking', 'eating']")
print("  → Temporal search tim chuoi frame theo thu tu thoi gian")
print("  → Tra ve chuoi keyframe do")
print("=" * 50)
