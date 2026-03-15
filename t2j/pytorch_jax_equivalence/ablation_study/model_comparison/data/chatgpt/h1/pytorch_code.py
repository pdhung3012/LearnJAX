import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# Create and train the model
torch.manual_seed(42)
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
X = torch.rand(100, 1)
y = 3 * X + 2 + torch.randn(100, 1) * 0.1
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()

# Save the model to a file named "model.pth"
torch.save(model.state_dict(), "model.pth")

# Load the model back from "model.pth"
loaded_model = SimpleModel()
loaded_model.load_state_dict(torch.load("model.pth"))
loaded_model.eval()

# Verify the model works after loading
X_test = torch.tensor([[0.5], [1.0], [1.5]])
with torch.no_grad():
    predictions = loaded_model(X_test)
    print(f"Predictions after loading: {predictions}")