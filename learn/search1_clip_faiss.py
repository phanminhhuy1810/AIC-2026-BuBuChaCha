"""
Bai 9: Ghep CLIP + FAISS — gõ text tim dung canh trong video.

Day la HE THONG LOI cua cuoc thi, chay that.

CACH DUNG:
  1. Chay keyframe1_extract.py truoc (de co keyframe)
  2. Chay:  conda activate aic && python learn/search1_clip_faiss.py
  3. Go mo ta → no tra ve keyframe khop nhat

LUONG HOAT DONG:
  [60 keyframe] → CLIP encode → 60 vector → FAISS index
  User go text → CLIP encode → 1 vector → FAISS tim top 5
"""
import torch
import open_clip
import faiss
import numpy as np
from PIL import Image
import os
import glob
import time

# =================================================================
# BUOC 1: TAI MODEL CLIP
# =================================================================
print("Dang tai model CLIP...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()
print("OK!\n")

# =================================================================
# BUOC 2: TIM KEYFRAME
# =================================================================
keyframe_dir = "data/keyframes"
image_paths = sorted(glob.glob(f"{keyframe_dir}/**/*.jpg", recursive=True))

if not image_paths:
    print(f"KHONG TIM THAY KEYFRAME trong {keyframe_dir}/")
    print("Chay  python learn/keyframe1_extract.py  truoc.")
    exit()

print(f"Tim thay {len(image_paths)} keyframe.")

# =================================================================
# BUOC 3: ENCODE TAT CA KEYFRAME BANG CLIP (offline)
# =================================================================
print("Dang encode keyframe bang CLIP...")
t0 = time.time()

all_features = []
batch_size = 32  # encode 32 anh 1 luc cho nhanh

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    batch_tensors = []

    for path in batch_paths:
        img = Image.open(path).convert("RGB")
        img_tensor = preprocess(img)
        batch_tensors.append(img_tensor)

    batch = torch.stack(batch_tensors)

    with torch.no_grad():
        features = model.encode_image(batch)

    all_features.append(features)

# Ghep tat ca vector lai
image_features = torch.cat(all_features, dim=0)  # (N, 512)
image_features = image_features / image_features.norm(dim=1, keepdim=True)
image_features_np = image_features.numpy().astype(np.float32)

t1 = time.time()
print(f"Encode {len(image_paths)} keyframe trong {t1-t0:.1f}s")

# =================================================================
# BUOC 4: TAO FAISS INDEX
# =================================================================
dim = image_features_np.shape[1]  # 512
index = faiss.IndexFlatIP(dim)
index.add(image_features_np)
print(f"FAISS index: {index.ntotal} vector, {dim} chieu\n")

# =================================================================
# BUOC 5: TIM KIEM BANG TEXT
# =================================================================
print("=" * 60)
print("HE THONG TIM KIEM VIDEO")
print("Go mo ta canh ban muon tim (tieng Anh).")
print("Go 'q' de thoat.")
print("=" * 60)

while True:
    query = input("\nQuery: ").strip()
    if query.lower() == 'q':
        break
    if not query:
        continue

    # Encode text
    text_tokens = tokenizer([query])
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    text_features_np = text_features.numpy().astype(np.float32)

    # FAISS search
    k = 5
    scores, ids = index.search(text_features_np, k)

    print(f"\nTop {k} ket qua cho: \"{query}\"")
    print("-" * 50)
    for rank in range(k):
        idx = ids[0][rank]
        score = scores[0][rank]
        path = image_paths[idx]
        name = os.path.basename(path)
        print(f"  #{rank+1}: {name} | score: {score:.4f}")

    print(f"\nMo anh xem:  open \"{image_paths[ids[0][0]]}\"")
