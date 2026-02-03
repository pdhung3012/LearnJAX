import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic data
torch.manual_seed(42)
X = torch.rand(100, 1) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + torch.randn(100, 1)  # Linear relationship with noise

class LearnedSiLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        # Save the input tensor and slope for backward computation
        ctx.save_for_backward(x)
        ctx.slope = slope
        return slope * x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the input and slope saved in the forward pass
        x, = ctx.saved_tensors
        slope = ctx.slope
        sigmoid_x = torch.sigmoid(x)

        # Compute the gradient with respect to input (x)
        grad_input = grad_output * slope * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))

        # Compute the gradient with respect to slope
        grad_slope = grad_output * x * sigmoid_x

        return grad_input, grad_slope


# Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, slope=1):
        super().__init__()
        self.slope = nn.Parameter(torch.ones(1) * slope)

    def forward(self, x):
        # Use the custom LearnedSiLUFunction
        return LearnedSiLUFunction.apply(x, self.slope)

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

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

# Display the learned parameters
[w, b] = model.linear.parameters()
print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

# Testing on new data
X_test = torch.tensor([[4.0], [7.0]])
with torch.no_grad():
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
