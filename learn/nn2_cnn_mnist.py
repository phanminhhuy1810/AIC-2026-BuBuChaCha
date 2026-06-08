"""
Bai 5: CNN — Convolutional Neural Network tren MNIST.
So voi bai 4 (Linear), CNN hieu "vi tri pixel" => accuracy cao hon.

KHAI NIEM MOI:
  - Conv2d: bo loc truot tren anh, phat hien dac trung cuc bo (canh, goc, duong cong)
  - MaxPool2d: thu nho anh, giu lai dac trung quan trong nhat
  - Flatten: lam phang tu 2D thanh 1D truoc khi vao Linear

Chay:  conda activate aic && python learn/nn2_cnn_mnist.py
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# === Du lieu (giong bai 4) ===
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# =================================================================
# MODEL CNN
# =================================================================
# So sanh voi bai 4 (SimpleNN):
#
#   Bai 4: Input(784) -> Linear(128) -> Linear(64) -> Linear(10)
#          Anh bi lam phang NGAY TU DAU => mat thong tin vi tri
#
#   Bai 5: Input(1,28,28) -> Conv -> Conv -> Flatten -> Linear -> Output(10)
#          Giu nguyen anh 2D, dung bo loc truot de hieu hinh dang

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # === CONV LAYER 1 ===
        # nn.Conv2d(in_channels, out_channels, kernel_size)
        #
        # in_channels=1:  anh MNIST trang den, 1 kenh mau
        #                 (anh mau RGB = 3 kenh)
        # out_channels=16: tao 16 "bo loc" khac nhau
        #                  moi bo loc hoc phat hien 1 dac trung
        #                  (bo loc 1: canh doc, bo loc 2: canh ngang, ...)
        # kernel_size=3:   bo loc 3x3 pixel, truot qua toan bo anh
        # padding=1:       them vien 1 pixel de kich thuoc khong bi nho lai
        #
        # Hinh dung: cam 1 kinh lup 3x3, truot qua anh,
        #            tai moi vi tri ghi lai "co canh o day khong?"
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        # === CONV LAYER 2 ===
        # in_channels=16 (nhan tu conv1), out_channels=32
        # Nhin vao KET QUA cua conv1 (cac canh, goc)
        # va ghep thanh dac trung phuc tap hon (duong cong, vong tron)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # === MAX POOL ===
        # Thu nho anh 2x: lay gia tri lon nhat trong moi o 2x2
        # 28x28 -> 14x14 (sau pool 1)
        # 14x14 -> 7x7   (sau pool 2)
        # Tac dung: giam kich thuoc + giu dac trung manh nhat
        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU()

        # === FLATTEN + LINEAR ===
        # Sau 2 conv + 2 pool: anh con 7x7, co 32 kenh => 32*7*7 = 1568 so
        # Lam phang roi dua vao Linear (giong bai 4)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x shape: (batch, 1, 28, 28)

        x = self.pool(self.relu(self.conv1(x)))   # -> (batch, 16, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))   # -> (batch, 32, 7, 7)

        x = x.view(x.size(0), -1)                 # -> (batch, 1568) flatten

        x = self.relu(self.fc1(x))                 # -> (batch, 128)
        x = self.fc2(x)                            # -> (batch, 10)
        return x

model = CNN()
total_params = sum(p.numel() for p in model.parameters())
print(f"CNN co {total_params:,} parameters")
print(f"(Bai 4 SimpleNN co 109,386 params)")

# === Train (giong hệt bai 4 — chi doi model) ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {acc:.1f}%")

# === Test ===
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"\nTest Accuracy: {correct/total*100:.1f}% ({correct}/{total})")
print()
print("=" * 50)
print("SO SANH:")
print(f"  Bai 4 (Linear NN):  ~97%")
print(f"  Bai 5 (CNN):        ~99%")
print("=> CNN hieu hinh dang anh => chinh xac hon nhieu")
print("=> CLIP (cuoc thi) dung ViT = phien ban tien tien cua CNN")
print("=" * 50)
