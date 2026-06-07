"""
Bai 2: Gradient Descent tu code bang Python thuan.
Bai toan: tim w, b sao cho y = w*x + b khop du lieu nha.
Chay:  conda activate aic && python learn/gd1_linear_regression.py
"""

# === BUOC 1: Du lieu gia (dien tich m2 -> gia ty dong) ===
X = [30, 50, 70, 90, 110, 130]
Y = [1.2, 2.1, 2.8, 3.6, 4.5, 5.2]

n = len(X)  # so mau = 6

# === BUOC 2: Khoi tao w, b ngau nhien ===
w = 0.0
b = 0.0

# === BUOC 3: Hyperparameters ===
lr = 0.00005   # learning rate (thu doi xem sao)
epochs = 500   # so lan lap

# === BUOC 4: Vong lap gradient descent ===
for epoch in range(epochs):

    # --- 4a: Tinh loss (MSE) ---
    loss = 0.0
    for i in range(n):
        y_pred = w * X[i] + b
        loss += (y_pred - Y[i]) ** 2
    loss = loss / n  # trung binh

    # --- 4b: Tinh gradient (dao ham loss theo w va b) ---
    dw = 0.0
    db = 0.0
    for i in range(n):
        y_pred = w * X[i] + b
        dw += 2 * (y_pred - Y[i]) * X[i] / n
        db += 2 * (y_pred - Y[i]) / n

    # --- 4c: Cap nhat w, b (di nguoc gradient) ---
    w = w - lr * dw
    b = b - lr * db

    # In moi 100 epoch
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | loss = {loss:.4f} | w = {w:.4f} | b = {b:.4f}")

# === BUOC 5: Ket qua ===
print(f"\nKet qua: y = {w:.4f} * x + {b:.4f}")
print(f"(Doc: moi m2 tang gia {w:.4f} ty, gia goc {b:.4f} ty)")

print("\n--- Thu du doan ---")
for x_test in [40, 80, 120]:
    print(f"  Nha {x_test} m2 -> du doan {w * x_test + b:.2f} ty")
