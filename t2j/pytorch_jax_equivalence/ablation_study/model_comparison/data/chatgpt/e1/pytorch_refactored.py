import torch
import torch.nn as nn
import torch.optim as optim


def generate_data():
    """
    Generate synthetic data.

    IMPORTANT: This function must be called after torch.manual_seed(42)
    to preserve the exact random draw order from the original script.
    """
    X = torch.rand(100, 1) * 10  # 100 data points between 0 and 10
    y = 2 * X + 3 + torch.randn(100, 1)  # Linear relationship with noise
    return X, y


# Define the Linear Regression Model (module-level and importable)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single input and single output

    def forward(self, x):
        return self.linear(x)


def make_model():
    """
    Instantiate and return the model.

    Do NOT change weight initialization: nn.Linear initializes weights/biases
    internally using PyTorch defaults, same as the original.
    """
    return LinearRegressionModel()


def make_criterion():
    """Return the loss function (same as original)."""
    return nn.MSELoss()


def make_optimizer(model):
    """Return the optimizer with identical type and hyperparameters."""
    return optim.SGD(model.parameters(), lr=0.01)


def train_model(X, y, model, optimizer, criterion, num_epochs):
    """
    Training loop with identical logic to the original script.
    Prints progress every 100 epochs with the same formatting.
    """
    for epoch in range(num_epochs):
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


def main():
    # Seed must be at the top of main() (per rules).
    # Call order preserved vs original module-level execution:
    # manual_seed -> generate_data -> make_model -> make_criterion -> make_optimizer -> train loop -> prints
    torch.manual_seed(42)

    X, y = generate_data()

    model = make_model()
    criterion = make_criterion()
    optimizer = make_optimizer(model)

    epochs = 1000
    train_model(X, y, model, optimizer, criterion, epochs)

    # Display the learned parameters (same unpacking and formatting)
    [w, b] = model.linear.parameters()
    print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

    # Testing on new data
    X_test = torch.tensor([[4.0], [7.0]])
    with torch.no_grad():
        predictions = model(X_test)
        print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


if __name__ == "__main__":
    main()