import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic data
torch.manual_seed(42)
X = torch.rand(100, 2) * 10  # 100 data points with 2 features
y = (X[:, 0] + X[:, 1] * 2).unsqueeze(1) + torch.randn(100, 1)  # Non-linear relationship with noise

# Define the Deep Neural Network Model
class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # Input layer to hidden layer
        self.relu = nn.ReLU()       # Activation function
        self.fc2 = nn.Linear(10, 1) # Hidden layer to output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# Initialize the model, loss function, and optimizer
model = DNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Testing on new data
X_test = torch.tensor([[4.0, 3.0], [7.0, 8.0]])
with torch.no_grad():
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
    