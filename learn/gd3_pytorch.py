"""
Bai 3b: Gradient Descent bang PyTorch (autograd tinh gradient tu dong).
So sanh: ban CHI viet ham tinh loss, PyTorch tu tinh dao ham.
Chay:  python learn/gd3_pytorch.py
"""
import torch

# === Du lieu — dung tensor thay numpy array ===
# tensor = giong numpy array, nhung ho tro GPU + autograd
X = torch.tensor([30, 50, 70, 90, 110, 130], dtype=torch.float32)
Y = torch.tensor([1.2, 2.1, 2.8, 3.6, 4.5, 5.2], dtype=torch.float32)

# === Khoi tao w, b — BAT requires_grad=True de PyTorch theo doi ===
# Day la diem khac duy nhat: noi PyTorch "hay tinh dao ham theo 2 bien nay"
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

lr = 0.00005
epochs = 500

for epoch in range(epochs):

    # --- Chi viet FORWARD (tinh loss) — KHONG can viet dao ham ---
    y_pred = w * X + b
    loss = torch.mean((y_pred - Y) ** 2)

    # --- PyTorch tu tinh gradient ---
    loss.backward()    # 1 lenh nay = tinh het dao ham

    # --- Cap nhat w, b (phai tat autograd tam thoi) ---
    with torch.no_grad():
        w -= lr * w.grad     # w.grad = dloss/dw (PyTorch tinh san)
        b -= lr * b.grad     # b.grad = dloss/db

    # --- Xoa gradient cu (neu khong, no cong don) ---
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | loss = {loss.item():.4f} | w = {w.item():.4f} | b = {b.item():.4f}")

print(f"\nKet qua: y = {w.item():.4f} * x + {b.item():.4f}")

print("\n" + "="*60)
print("TOM TAT 3 CACH:")
print("="*60)
print("Python thuan : TU viet vong lap + TU tinh dao ham")
print("NumPy        : BO vong lap (vector hoa) + TU tinh dao ham")
print("PyTorch      : BO vong lap + BO tinh dao ham (autograd)")
print("=> PyTorch = ban chi can viet: y_pred va loss. Het.")
