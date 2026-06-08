"""
Bai 7: FAISS — tim vector gan nhat trong hang trieu vector.

VAN DE:
  Bai 6: so sanh text voi 3 anh -> for loop du
  Cuoc thi: so sanh text voi 1,000,000 anh -> for loop qua cham
  FAISS: tim top-K gan nhat trong 1 trieu vector chi mat vai millisecond

FAISS = Facebook AI Similarity Search
  Giong nhu std::map (O(log n) thay vi O(n)) nhung cho vector.

Chay:  conda activate aic && python learn/faiss1_demo.py
"""
import numpy as np
import faiss
import time

# =================================================================
# PHAN 1: FAISS CO BAN — 5 DONG LA XONG
# =================================================================

print("=" * 60)
print("PHAN 1: FAISS co ban")
print("=" * 60)

# Gia su da co 10,000 anh, moi anh da encode thanh vector 512 so
# (trong thuc te dung CLIP encode, o day gia lap bang random)
n_images = 10_000
dim = 512

# Tao 10,000 vector ngau nhien (gia lap 10,000 anh da encode)
np.random.seed(42)
image_vectors = np.random.randn(n_images, dim).astype(np.float32)

# === BUOC 1: Tao FAISS index ===
# IndexFlatIP = brute-force inner product (cosine similarity khi vector da normalize)
# Giong tao 1 cai "database" chua vector
index = faiss.IndexFlatIP(dim)

# === BUOC 2: Them vector vao index ===
# Normalize truoc (de inner product = cosine similarity)
faiss.normalize_L2(image_vectors)
index.add(image_vectors)

print(f"Da them {index.ntotal} vector vao FAISS index")

# === BUOC 3: Tim kiem ===
# Gia lap 1 text query da encode thanh vector
query = np.random.randn(1, dim).astype(np.float32)
faiss.normalize_L2(query)

k = 5  # tim top 5 gan nhat

t0 = time.time()
scores, ids = index.search(query, k)
t1 = time.time()

# scores = diem cosine similarity cua top 5
# ids    = index (vi tri) cua top 5 anh trong database

print(f"\nQuery tim top {k} anh gan nhat trong {n_images:,} anh:")
print(f"  Thoi gian: {(t1-t0)*1000:.2f} ms")
print(f"  Ket qua:")
for i in range(k):
    print(f"    #{i+1}: anh thu {ids[0][i]:5d} | score = {scores[0][i]:.4f}")

# =================================================================
# PHAN 2: SO SANH TOC DO — FOR LOOP vs FAISS
# =================================================================

print(f"\n{'=' * 60}")
print("PHAN 2: So sanh toc do")
print("=" * 60)

# Cach 1: For loop (giong bai 6)
t0 = time.time()
# Tinh cosine similarity voi tat ca 10,000 vector
all_scores = query @ image_vectors.T     # (1, 10000)
top_k_ids = np.argsort(all_scores[0])[::-1][:k]
t1 = time.time()
time_loop = (t1 - t0) * 1000

# Cach 2: FAISS
t0 = time.time()
scores, ids = index.search(query, k)
t1 = time.time()
time_faiss = (t1 - t0) * 1000

print(f"  NumPy brute-force: {time_loop:.2f} ms")
print(f"  FAISS:             {time_faiss:.2f} ms")
print(f"  (Voi 10K vector, chenh lech chua nhieu)")

# =================================================================
# PHAN 3: 1 TRIEU VECTOR — FAISS TOA SANG
# =================================================================

print(f"\n{'=' * 60}")
print("PHAN 3: 1 TRIEU vector (giong cuoc thi that)")
print("=" * 60)

n_big = 1_000_000
print(f"Dang tao {n_big:,} vector gia lap...")

big_vectors = np.random.randn(n_big, dim).astype(np.float32)
faiss.normalize_L2(big_vectors)

# FAISS voi IVF index (chia vung, chi tim trong vung gan)
# Giong binary search: khong can quet het, chi quet vung lien quan
nlist = 100   # chia thanh 100 vung (cluster)
quantizer = faiss.IndexFlatIP(dim)
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

print("Dang train FAISS index (chia vung)...")
index_ivf.train(big_vectors)
index_ivf.add(big_vectors)
index_ivf.nprobe = 10  # tim trong 10/100 vung gan nhat (do chinh xac vs toc do)

print(f"FAISS index: {index_ivf.ntotal:,} vector, {nlist} vung\n")

# So sanh toc do
query = np.random.randn(1, dim).astype(np.float32)
faiss.normalize_L2(query)

# Brute-force
t0 = time.time()
all_scores = query @ big_vectors.T
top_k_ids = np.argsort(all_scores[0])[::-1][:k]
t1 = time.time()
time_brute = (t1 - t0) * 1000

# FAISS IVF
t0 = time.time()
scores, ids = index_ivf.search(query, k)
t1 = time.time()
time_ivf = (t1 - t0) * 1000

print(f"  Brute-force (1 TRIEU vector): {time_brute:.1f} ms")
print(f"  FAISS IVF   (1 TRIEU vector): {time_ivf:.1f} ms")
if time_ivf > 0:
    print(f"  => FAISS nhanh hon {time_brute/time_ivf:.0f}x!")

# =================================================================
# PHAN 4: GHEP CLIP + FAISS (CODE THUC TE)
# =================================================================

print(f"\n{'=' * 60}")
print("PHAN 4: Ghep CLIP + FAISS (giong he thong thi)")
print("=" * 60)
print("""
CODE THUC TE chi can:

  # --- OFFLINE (chay 1 lan) ---
  for frame in all_keyframes:
      vec = clip_model.encode_image(frame)    # CLIP
      index.add(vec)                           # FAISS

  # --- ONLINE (khi user hoi) ---
  query_vec = clip_model.encode_text(user_query)  # CLIP
  scores, ids = index.search(query_vec, k=10)      # FAISS
  show(keyframes[ids])                              # hien thi

=> XONG. Day la TOAN BO loi cua he thong thi.
""")

print("=" * 60)
print("TOM TAT:")
print("  CLIP  = bien anh/text thanh vector")
print("  FAISS = tim vector gan nhat cuc nhanh")
print("  CLIP + FAISS = he thong truy xuat anh/video")
print("=" * 60)
