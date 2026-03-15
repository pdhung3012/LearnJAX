import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input: 28x28 pixels, Output: 128 neurons
        self.fc2 = nn.Linear(128, 10)  # Output: 10 classes (digits 0-9)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop with benchmarking
epochs = 5
for epoch in range(epochs):
    start_time = time.time()  # Start time for training
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_time = time.time()  # End time for training
    training_time = end_time - start_time
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Time: {training_time:.4f}s")

# Evaluate the model on the test set and benchmark the accuracy
correct = 0
total = 0
start_time = time.time()  # Start time for testing
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

end_time = time.time()  # End time for testing
testing_time = end_time - start_time
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%, Testing Time: {testing_time:.4f}s")