import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

class Conv2dCustom(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2dCustom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        batch_size, in_channels, H, W = x.shape
        KH, KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding

        x_padded = F.pad(x, (PW, PW, PH, PH))

        OH = (H + 2 * PH - KH) // SH + 1
        OW = (W + 2 * PW - KW) // SW + 1

        out = torch.zeros((batch_size, self.out_channels, OH, OW), device=x.device)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(OH):
                    for j in range(OW):
                        h_start = i * SH
                        h_end = h_start + KH
                        w_start = j * SW
                        w_end = w_start + KW
                        region = x_padded[b, :, h_start:h_end, w_start:w_end]
                        out[b, oc, i, j] = torch.sum(region * self.weight[oc]) + self.bias[oc]
        return out

class MaxPool2dCustom(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(MaxPool2dCustom, self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        batch_size, channels, H, W = x.shape
        KH, KW = self.kernel_size
        SH = SW = self.stride

        OH = (H - KH) // SH + 1
        OW = (W - KW) // SW + 1

        out = torch.zeros((batch_size, channels, OH, OW), device=x.device)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(OH):
                    for j in range(OW):
                        h_start = i * SH
                        h_end = h_start + KH
                        w_start = j * SW
                        w_end = w_start + KW
                        region = x[b, c, h_start:h_end, w_start:w_end]
                        out[b, c, i, j] = torch.max(region)
        return out

# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Output: 32x32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64x32x32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64x16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate on the test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
