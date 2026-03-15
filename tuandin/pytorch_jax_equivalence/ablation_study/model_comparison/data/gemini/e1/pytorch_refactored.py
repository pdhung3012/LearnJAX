import torch
import torch.nn as nn
import torch.optim as optim

## --- Model Definition ---

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single input and single output

    def forward(self, x):
        return self.linear(x)

## --- Component Factories ---

def generate_data():
    """Generates synthetic data: y = 2X + 3 + noise."""
    X = torch.rand(100, 1) * 10  # 100 data points between 0 and 10
    y = 2 * X + 3 + torch.randn(100, 1)  # Linear relationship with noise
    return X, y

def make_model():
    """Instantiates the Linear Regression model."""
    return LinearRegressionModel()

def make_criterion():
    """Returns the loss function."""
    return nn.MSELoss()

def make_optimizer(model):
    """Returns the SGD optimizer with original hyperparameters."""
    return optim.SGD(model.parameters(), lr=0.01)

## --- Training Logic ---

def train_model(X, y, model, optimizer, criterion, num_epochs):
    """Executes the training loop and logs progress."""
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
    return model

## --- Execution ---

def main():
    # 1. Set seed first to ensure reproducibility
    torch.manual_seed(42)

    # 2. Call order preserved: Data -> Model -> Criterion -> Optimizer
    # This matches the original script's global execution flow.
    X, y = generate_data()
    model = make_model()
    criterion = make_criterion()
    optimizer = make_optimizer(model)

    # 3. Train
    epochs = 1000
    train_model(X, y, model, optimizer, criterion, epochs)

    # 4. Display learned parameters
    [w, b] = model.linear.parameters()
    print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

    # 5. Testing
    X_test = torch.tensor([[4.0], [7.0]])
    with torch.no_grad():
        predictions = model(X_test)
        print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == '__main__':
    main()