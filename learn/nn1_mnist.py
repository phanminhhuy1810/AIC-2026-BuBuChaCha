"""
Bai 4: Neural Network phan loai chu so viet tay (MNIST).
Dung torch.nn.Module — cach chuan viet model trong PyTorch.
Chay:  conda activate aic && python learn/nn1_mnist.py
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =====================================================
# BUOC 1: TAI DU LIEU MNIST
# =====================================================
# transforms.ToTensor() = chuyen anh thanh tensor + chia 255 (ve 0-1)
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)

# DataLoader = chia data thanh batch, shuffle moi epoch
# Giong nhu ban doc file theo tung block thay vi tung byte
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

print(f"Train: {len(train_data)} anh | Test: {len(test_data)} anh")
print(f"Moi anh: {train_data[0][0].shape} (1 kenh x 28 x 28 pixel)")

# =====================================================
# BUOC 2: DINH NGHIA MODEL
# =====================================================
# Trong C++ ban viet class ke thua base class.
# PyTorch y het: ke thua nn.Module, override forward().

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 layer:
        # 784 (input) -> 128 -> 64 -> 10 (output)
        self.layer1 = nn.Linear(784, 128)   # 784*128 + 128 = 100,480 params
        self.layer2 = nn.Linear(128, 64)    # 128*64 + 64   = 8,256 params
        self.layer3 = nn.Linear(64, 10)     # 64*10 + 10    = 650 params
        self.relu = nn.ReLU()

    def forward(self, x):
        # x vao co shape (batch, 1, 28, 28)
        x = x.view(x.size(0), -1)  # lam phang thanh (batch, 784)
                                     # giong reshape trong numpy
        x = self.relu(self.layer1(x))   # linear -> relu
        x = self.relu(self.layer2(x))   # linear -> relu
        x = self.layer3(x)              # layer cuoi KHONG relu (vi dung CrossEntropy)
        return x

model = SimpleNN()

# Dem tong so parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model co {total_params:,} parameters")

# =====================================================
# BUOC 3: LOSS + OPTIMIZER
# =====================================================
# CrossEntropyLoss = loss cho bai phan loai nhieu nhan (0-9)
# Adam = optimizer thong minh hon gradient descent thuong
#   (tu dieu chinh lr cho tung parameter)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =====================================================
# BUOC 4: TRAIN
# =====================================================
epochs = 3  # 3 epoch la du cho MNIST

for epoch in range(epochs):
    model.train()  # bat che do train
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # --- Forward ---
        outputs = model(images)          # model tu goi forward()
        loss = criterion(outputs, labels)

        # --- Backward + Update ---
        optimizer.zero_grad()   # xoa gradient cu  (giong w.grad.zero_())
        loss.backward()         # tinh gradient    (giong bai 3)
        optimizer.step()        # cap nhat params  (giong w -= lr * w.grad)

        # Thong ke
        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)   # lay so co xac suat cao nhat
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {acc:.1f}%")

# =====================================================
# BUOC 5: TEST
# =====================================================
model.eval()   # bat che do test (tat dropout, batch norm...)
correct = 0
total = 0

with torch.no_grad():  # khong can tinh gradient khi test
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"\nTest Accuracy: {correct/total*100:.1f}% ({correct}/{total})")
print("(Chi voi 3 epoch, ~109K params, khong GPU — da dat ~97%)")

# =====================================================
# BUOC 6: THU DOAN 1 ANH
# =====================================================
sample_image, sample_label = test_data[0]
with torch.no_grad():
    output = model(sample_image.unsqueeze(0))  # them chieu batch
    pred = output.argmax(dim=1).item()

print(f"\nThu doan 1 anh: dap an = {sample_label}, model doan = {pred}",
      "DUNG!" if pred == sample_label else "SAI!")
