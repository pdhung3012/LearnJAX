import torch
import torch.nn as nn
import torch.optim as optim


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


def generate_data():
    """Generate training data. Returns (X, y)."""
    X = torch.rand(100, 1)
    y = 3 * X + 2 + torch.randn(100, 1) * 0.1
    return X, y


def make_model():
    """Instantiate and return a SimpleModel with default weight initialisation."""
    return SimpleModel()


def make_criterion():
    """Return the loss function."""
    return nn.MSELoss()


def make_optimizer(model):
    """Return an SGD optimizer with the original hyperparameters."""
    return optim.SGD(model.parameters(), lr=0.01)


def train_model(X, y, model, optimizer, criterion, num_epochs):
    """Run the training loop with identical logic to the original script."""
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()


def main():
    # Seed set first, matching the original module-level position.
    torch.manual_seed(42)

    # Original call order: model init (consumes RNG), then data generation.
    model = make_model()
    criterion = make_criterion()
    optimizer = make_optimizer(model)

    X, y = generate_data()

    train_model(X, y, model, optimizer, criterion, num_epochs=100)

    torch.save(model.state_dict(), "model.pth")

    loaded_model = make_model()
    loaded_model.load_state_dict(torch.load("model.pth"))
    loaded_model.eval()

    X_test = torch.tensor([[0.5], [1.0], [1.5]])
    with torch.no_grad():
        predictions = loaded_model(X_test)
        print(f"Predictions after loading: {predictions}")


if __name__ == "__main__":
    main()