"""
Bai 3a: Gradient Descent bang NumPy (vector hoa, khong vong lap for i).

DIEM CHINH CAN HIEU:
  - numpy array KHAC list Python o cho nao
  - "vector hoa" nghia la gi
  - tai sao nhanh hon list

Chay:  conda activate aic && python learn/gd2_numpy.py
"""
import numpy as np

# =================================================================
# PHAN 1: LIST vs NUMPY ARRAY — khac nhau the nao?
# =================================================================
#
# LIST (ban da biet tu bai 1-2):
#   a = [1, 2, 3]
#   => giong vector<int> trong C++
#   => muon nhan moi phan tu *2 phai dung vong lap:
#      [x * 2 for x in a]  ==> [2, 4, 6]
#
# NUMPY ARRAY:
#   a = np.array([1, 2, 3])
#   => giong mang C nhung CO THE lam phep tinh tren CA MANG 1 luc:
#      a * 2  ==> array([2, 4, 6])     (khong can vong lap!)
#      a + 1  ==> array([2, 3, 4])
#      a ** 2 ==> array([1, 4, 9])
#
# TAI SAO NHANH HON?
#   List: Python lap tung phan tu (cham vi Python la interpreted)
#   NumPy: goi code C ben duoi, tinh ca mang 1 phep (nhu SIMD trong C++)

# Tao numpy array tu list:
X = np.array([30, 50, 70, 90, 110, 130], dtype=np.float64)
Y = np.array([1.2, 2.1, 2.8, 3.6, 4.5, 5.2], dtype=np.float64)
# dtype=np.float64  giong  double trong C++
# dtype=np.float32  giong  float  trong C++

n = len(X)   # van la 6

# =================================================================
# PHAN 2: GRADIENT DESCENT — VECTOR HOA
# =================================================================
# "Vector hoa" = thay vong for i bang phep tinh tren ca mang.
# Logic HOAN TOAN GIONG bai 1, chi khac cach viet.

w = 0.0
b = 0.0
lr = 0.00005
epochs = 500

for epoch in range(epochs):

    # --- y_pred cho TAT CA mau cung luc ---
    # Bai 1 (list):
    #   for i in range(n):
    #       y_pred = w * X[i] + b        <-- tinh tung cai
    #
    # Numpy:
    y_pred = w * X + b
    #   w * X  ==> w nhan tung phan tu cua X  ==> array 6 phan tu
    #   + b    ==> cong b vao tung phan tu
    #   Ket qua: y_pred = array([...]) 6 phan tu, MOI phan tu = w*X[i]+b

    # --- Loss (MSE) ---
    # Bai 1:
    #   loss = 0
    #   for i in range(n):
    #       loss += (y_pred - Y[i]) ** 2
    #   loss /= n
    #
    # Numpy:
    loss = np.mean((y_pred - Y) ** 2)
    #   y_pred - Y  ==> array 6 phan tu (sai so tung mau)
    #   ** 2        ==> binh phuong tung phan tu
    #   np.mean()   ==> tinh trung binh ca mang => 1 so

    # --- Gradient ---
    # Bai 1:
    #   dw = 0
    #   for i in range(n):
    #       dw += 2 * (y_pred - Y[i]) * X[i] / n
    #
    # Numpy:
    dw = np.mean(2 * (y_pred - Y) * X)
    db = np.mean(2 * (y_pred - Y))
    #   (y_pred - Y) * X  ==> nhan tung cap tuong ung (element-wise)
    #   np.mean()          ==> trung binh = tong/n

    # --- Cap nhat (y het bai 1) ---
    w = w - lr * dw
    b = b - lr * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | loss = {loss:.4f} | w = {w:.4f} | b = {b:.4f}")

print(f"\nKet qua: y = {w:.4f} * x + {b:.4f}")

# =================================================================
# PHAN 3: DO TOC DO — LIST vs NUMPY
# =================================================================
import time

big_list = list(range(1_000_000))          # list 1 trieu phan tu
big_np = np.arange(1_000_000)              # numpy array 1 trieu phan tu

# List: phai dung list comprehension (vong lap an)
t0 = time.time()
result_list = [x * 2 + 1 for x in big_list]
t1 = time.time()
print(f"\n--- So sanh toc do: list vs numpy ---")
print(f"  List comprehension (1 trieu phan tu): {t1-t0:.4f}s")

# NumPy: 1 phep toan tren ca mang
t0 = time.time()
result_np = big_np * 2 + 1
t1 = time.time()
print(f"  NumPy (1 trieu phan tu):              {t1-t0:.4f}s")
print(f"  => NumPy nhanh hon khoang {(t1-t0) and ((time.time()-t0) or 1):.0f}x")
