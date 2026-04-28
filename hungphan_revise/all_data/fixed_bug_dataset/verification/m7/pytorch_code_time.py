import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=(device.type == "cuda"),
)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    pin_memory=(device.type == "cuda"),
)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


overall_start_time = time.perf_counter()

epochs = 5
for epoch in range(epochs):
    start_time = time.time()
    for images, labels in train_loader:
        images = images.to(device, non_blocking=(device.type == "cuda"))
        labels = labels.to(device, non_blocking=(device.type == "cuda"))

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    training_time = end_time - start_time
    print(
        f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Time: {training_time:.4f}s")


correct = 0
total = 0
start_time = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=(device.type == "cuda"))
        labels = labels.to(device, non_blocking=(device.type == "cuda"))

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

if device.type == "cuda":
    torch.cuda.synchronize()

end_time = time.time()
testing_time = end_time - start_time
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%, Testing Time: {testing_time:.4f}s")

overall_end_time = time.perf_counter()
print(f"Execution time: {overall_end_time - overall_start_time:.4f} seconds")
