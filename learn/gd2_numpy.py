"""
Bai 3a: Gradient Descent bang NumPy (vector hoa, khong vong lap for i).
So sanh voi gd1_linear_regression.py — cung bai toan, code ngan hon, nhanh hon.
Chay:  python learn/gd2_numpy.py
"""
import numpy as np

# === Du lieu — giong bai 2, nhung dung numpy array thay list ===
X = np.array([30, 50, 70, 90, 110, 130], dtype=np.float64)
Y = np.array([1.2, 2.1, 2.8, 3.6, 4.5, 5.2], dtype=np.float64)
n = len(X)

# === Khoi tao ===
w = 0.0
b = 0.0
lr = 0.00005
epochs = 500

# === Gradient Descent — KHONG co vong for i ===
for epoch in range(epochs):

    # Tinh y_pred cho TAT CA mau cung luc (vector hoa)
    y_pred = w * X + b          # numpy: nhan ca mang 1 luc

    # Loss (MSE)
    loss = np.mean((y_pred - Y) ** 2)   # np.mean = trung binh ca mang

    # Gradient — cung 1 dong thay vi for i
    dw = np.mean(2 * (y_pred - Y) * X)  # trung binh cua (2 * sai_so * x)
    db = np.mean(2 * (y_pred - Y))

    # Cap nhat
    w = w - lr * dw
    b = b - lr * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | loss = {loss:.4f} | w = {w:.4f} | b = {b:.4f}")

print(f"\nKet qua: y = {w:.4f} * x + {b:.4f}")

# === So sanh toc do ===
print("\n--- So sanh toc do: list vs numpy ---")
import time

big_list = list(range(1_000_000))
big_np = np.arange(1_000_000)

t0 = time.time()
result_list = [x * 2 + 1 for x in big_list]
t1 = time.time()
print(f"  List comprehension (1 trieu phan tu): {t1-t0:.4f}s")

t0 = time.time()
result_np = big_np * 2 + 1
t1 = time.time()
print(f"  NumPy (1 trieu phan tu):              {t1-t0:.4f}s")
