"""
Bai 6: Demo CLIP — bien anh va text thanh vector, tinh cosine similarity.
Day la DUNG CAI ma he thong cuoc thi se dung.

CACH DUNG:
  1. Bo anh bat ky vao  data/sample_images/  (jpg, png deu duoc)
  2. Chay:  conda activate aic && python learn/clip1_demo.py
  3. Xem bang diem: cau nao khop anh nao
"""
import torch
import open_clip
from PIL import Image
import os
import glob

# =================================================================
# BUOC 1: TAI MODEL CLIP
# =================================================================
print("Dang tai model CLIP...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()
print("OK! Model CLIP da san sang.\n")

# =================================================================
# BUOC 2: TIM ANH TRONG THU MUC
# =================================================================
img_dir = "data/sample_images"
os.makedirs(img_dir, exist_ok=True)

image_paths = sorted(
    glob.glob(f"{img_dir}/*.jpg") +
    glob.glob(f"{img_dir}/*.jpeg") +
    glob.glob(f"{img_dir}/*.png")
)

if len(image_paths) == 0:
    print("KHONG TIM THAY ANH!")
    print(f"Hay bo vai anh (jpg/png) vao thu muc: {os.path.abspath(img_dir)}/")
    print("Roi chay lai.")
    exit()

image_names = [os.path.basename(p) for p in image_paths]
print(f"Tim thay {len(image_paths)} anh:")
for name in image_names:
    print(f"  - {name}")
print()

# =================================================================
# BUOC 3: BIEN ANH THANH VECTOR (Image Embedding)
# =================================================================
print("--- BUOC 3: Encode anh thanh vector ---")
image_tensors = []
for path in image_paths:
    img = Image.open(path).convert("RGB")
    img_tensor = preprocess(img)
    image_tensors.append(img_tensor)

image_batch = torch.stack(image_tensors)

with torch.no_grad():
    image_features = model.encode_image(image_batch)

image_features = image_features / image_features.norm(dim=1, keepdim=True)

for i, name in enumerate(image_names):
    print(f"  {name}: vector 512 so -> [{image_features[i][:3].tolist()}...]")

# =================================================================
# BUOC 4: BIEN TEXT THANH VECTOR (Text Embedding)
# =================================================================
print("\n--- BUOC 4: Encode text thanh vector ---")

queries = [
    "a photo of a cat",
    "a cute dog",
    "a red car on the road",
    "a person reading a book",
    "a beautiful sunset",
    "food on a plate",
]

text_tokens = tokenizer(queries)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)

text_features = text_features / text_features.norm(dim=1, keepdim=True)

# =================================================================
# BUOC 5: TINH COSINE SIMILARITY
# =================================================================
print("\n--- BUOC 5: Cosine similarity (text vs image) ---")

col_width = max(len(n) for n in image_names) + 2
print(f"{'':30s}", end="")
for name in image_names:
    print(f"{name:>{col_width}s}", end="")
print()
print("-" * (30 + col_width * len(image_names)))

similarity = text_features @ image_features.T

for i, q in enumerate(queries):
    print(f"{q:30s}", end="")
    best_j = similarity[i].argmax().item()
    for j in range(len(image_paths)):
        score = similarity[i][j].item()
        marker = " ***" if j == best_j else ""
        print(f"{score:>{col_width}.3f}{marker}", end="")
    print()

print()
print("*** = diem cao nhat cho moi cau query")
print()
print("=" * 60)
print("DAY CHINH LA CACH HE THONG CUOC THI HOAT DONG:")
print("  1. Encode tat ca keyframe thanh vector (offline)")
print("  2. User nhap mo ta -> encode thanh vector")
print("  3. Tim vector anh gan nhat (cosine similarity)")
print("  4. Tra ve keyframe do")
print("=" * 60)
