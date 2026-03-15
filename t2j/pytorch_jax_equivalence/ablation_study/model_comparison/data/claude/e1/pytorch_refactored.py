import torch
import torch.nn as nn
import torch.optim as optim


# Define the Linear Regression Model at module level so it is importable.
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single input and single output

    def forward(self, x):
        return self.linear(x)


def generate_data():
    """Generate synthetic data. Returns (X, y)."""
    X = torch.rand(100, 1) * 10  # 100 data points between 0 and 10
    y = 2 * X + 3 + torch.randn(100, 1)  # Linear relationship with noise
    return X, y


def make_model():
    """Instantiate and return the model. Weight initialisation is unchanged."""
    return LinearRegressionModel()


def make_criterion():
    """Return the loss function."""
    return nn.MSELoss()


def make_optimizer(model):
    """Return the optimizer with the same type and hyperparameters as the original."""
    return optim.SGD(model.parameters(), lr=0.01)


def train_model(X, y, model, optimizer, criterion, num_epochs):
    """Training loop with identical logic to the original script."""
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
    # Seed placed at top of main(), before any random-initialising operations.
    # Call order matches the original: seed -> generate_data -> make_model ->
    # make_criterion -> make_optimizer -> train_model -> display & test.
    torch.manual_seed(42)

    X, y = generate_data()
    model = make_model()
    criterion = make_criterion()
    optimizer = make_optimizer(model)

    train_model(X, y, model, optimizer, criterion, num_epochs=1000)

    # Display the learned parameters
    [w, b] = model.linear.parameters()
    print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

    # Testing on new data
    X_test = torch.tensor([[4.0], [7.0]])
    with torch.no_grad():
        predictions = model(X_test)
        print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


if __name__ == "__main__":
    main()