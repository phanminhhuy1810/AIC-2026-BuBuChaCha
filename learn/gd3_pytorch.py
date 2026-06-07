"""
Bai 3b: Gradient Descent bang PyTorch (autograd tinh gradient tu dong).

DIEM CHINH CAN HIEU:
  - tensor la gi (giong numpy array, nhung them 2 tinh nang)
  - requires_grad la gi
  - backward() lam gi
  - torch.no_grad() de lam gi

Chay:  conda activate aic && python learn/gd3_pytorch.py
"""
import torch

# =================================================================
# PHAN 1: TENSOR LA GI?
# =================================================================
#
# tensor = numpy array + 2 tinh nang them:
#   1. Chay duoc tren GPU (numpy chi chay CPU)
#   2. Tu dong tinh dao ham (autograd) — DAY LA CAI QUAN TRONG
#
# Cach tao tensor:
#   torch.tensor([1, 2, 3])             giong   np.array([1, 2, 3])
#   torch.tensor([1.0, 2.0, 3.0])       giong   np.array([1.0, 2.0, 3.0])
#   torch.zeros(5)                       giong   np.zeros(5)
#   torch.ones(3, 4)                     giong   np.ones((3, 4))
#
# Phep toan y het numpy:
#   a + b, a * b, a ** 2, a.mean(), ...

X = torch.tensor([30, 50, 70, 90, 110, 130], dtype=torch.float32)
Y = torch.tensor([1.2, 2.1, 2.8, 3.6, 4.5, 5.2], dtype=torch.float32)
# dtype=torch.float32 = kieu float trong C++ (pho bien nhat trong deep learning)

# =================================================================
# PHAN 2: requires_grad=True NGHIA LA GI?
# =================================================================
#
# Mac dinh, tensor la "so binh thuong" — khong theo doi gi ca.
# Khi bat requires_grad=True, ban noi voi PyTorch:
#   "Hay GHI NHO moi phep tinh lien quan den tensor nay,
#    de sau nay tao goi backward(), may tinh duoc dao ham."
#
# Chi bat cho PARAMETERS (w, b) — nhung bien can toi uu.
# KHONG bat cho data (X, Y) — vi khong can dao ham theo data.
#
# Hinh dung: nhu ban bat "Record" tren may tinh casio.
# Moi phep tinh duoc ghi lai thanh 1 cay (computational graph).
# Khi goi backward(), PyTorch di nguoc cay do de tinh dao ham.

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

lr = 0.00005
epochs = 500

for epoch in range(epochs):

    # ==============================================================
    # FORWARD PASS — chi viet cong thuc tinh loss, KHONG tinh dao ham
    # ==============================================================
    y_pred = w * X + b                      # y het numpy: w * tung phan tu X + b
    loss = torch.mean((y_pred - Y) ** 2)    # MSE — y het numpy

    # ==============================================================
    # BACKWARD PASS — PyTorch TU DONG tinh dao ham
    # ==============================================================
    # 1 dong nay thay the TOAN BO phan tinh dw, db trong bai 1 va 2.
    # PyTorch nhin lai "bang ghi" cac phep tinh o tren,
    # ap dung chain rule (quy tac chuoi — giai tich),
    # tinh ra: dloss/dw va dloss/db
    # Ket qua luu vao: w.grad va b.grad
    loss.backward()

    # SAU DONG NAY:
    #   w.grad = dao ham cua loss theo w  (giong dw trong bai 1)
    #   b.grad = dao ham cua loss theo b  (giong db trong bai 1)

    # ==============================================================
    # CAP NHAT PARAMETERS
    # ==============================================================
    # torch.no_grad() nghia la gi?
    #   Binh thuong moi phep tinh lien quan w, b deu duoc "ghi lai".
    #   Nhung buoc cap nhat (w -= lr * w.grad) KHONG PHAI la phan
    #   cua model — no chi la buoc chinh sua gia tri.
    #   Nen ta noi PyTorch: "dung ghi buoc nay, no khong can dao ham"
    #   => torch.no_grad()
    #
    # Trong C++ tuong duong: tam tat logging khi lam housekeeping.

    with torch.no_grad():
        w -= lr * w.grad     # w = w - lr * dloss/dw
        b -= lr * b.grad     # b = b - lr * dloss/db
        # w.grad luc nay CHAC CHAN co gia tri (khong phai None)
        # vi loss.backward() da tinh xong o tren

    # ==============================================================
    # XOA GRADIENT CU
    # ==============================================================
    # Mac dinh PyTorch CONG DON gradient qua moi lan backward().
    # Neu khong xoa, lan sau gradient = cu + moi => sai.
    # zero_() = dat ve 0. Dau _ cuoi = thay doi tai cho (in-place).
    #
    # Trong C++: giong   dw = 0; db = 0;  dau moi vong lap.

    w.grad.zero_()
    b.grad.zero_()

    if epoch % 100 == 0:
        # .item() = lay gia tri so tu tensor (tensor -> float)
        # Giong static_cast<float>(tensor) trong C++
        print(f"Epoch {epoch:4d} | loss = {loss.item():.4f} | w = {w.item():.4f} | b = {b.item():.4f}")

print(f"\nKet qua: y = {w.item():.4f} * x + {b.item():.4f}")

# =================================================================
# TOM TAT
# =================================================================
print("\n" + "=" * 60)
print("TOM TAT 3 CACH:")
print("=" * 60)
print("Python thuan : TU viet vong lap + TU tinh dao ham")
print("NumPy        : BO vong lap (vector hoa) + TU tinh dao ham")
print("PyTorch      : BO vong lap + BO tinh dao ham (autograd)")
print("=> PyTorch = ban chi can viet: y_pred va loss. Het.")
print()
print("SO SANH CU PHAP:")
print("-" * 60)
print("np.array([1,2,3])       <=>  torch.tensor([1,2,3])")
print("np.mean(a)              <=>  torch.mean(a)  hoac  a.mean()")
print("np.float64              <=>  torch.float32")
print("(khong co)              <=>  requires_grad=True")
print("(tu tinh dao ham)       <=>  loss.backward()")
print("(khong co)              <=>  w.grad")
