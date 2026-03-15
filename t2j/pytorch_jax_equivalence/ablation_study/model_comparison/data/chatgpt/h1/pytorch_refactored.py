# refactored_script.py
import torch
import torch.nn as nn
import torch.optim as optim


# 3) Model class at module level and importable
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


# 2) Wrap data generation
def generate_data():
    X = torch.rand(100, 1)
    y = 3 * X + 2 + torch.randn(100, 1) * 0.1
    return X, y


# 4) Factory for model (keeps same initialization path)
def make_model():
    return SimpleModel()


# 5) Factory for criterion
def make_criterion():
    return nn.MSELoss()


# 6) Factory for optimizer (same type + hyperparams)
def make_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01)


# 7) Wrap training loop with identical logic
def train_model(X, y, model, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()


def main():
    # 8) Seed at the top of main and preserve call order of RNG operations.
    # Original RNG order after seeding was:
    #   model init -> X rand -> y randn
    torch.manual_seed(42)

    model = make_model()
    criterion = make_criterion()
    optimizer = make_optimizer(model)

    X, y = generate_data()

    epochs = 100
    train_model(X, y, model, optimizer, criterion, epochs)

    # Save the model to a file named "model.pth"
    torch.save(model.state_dict(), "model.pth")

    # Load the model back from "model.pth"
    loaded_model = make_model()
    loaded_model.load_state_dict(torch.load("model.pth"))
    loaded_model.eval()

    # Verify the model works after loading
    X_test = torch.tensor([[0.5], [1.0], [1.5]])
    with torch.no_grad():
        predictions = loaded_model(X_test)
        print(f"Predictions after loading: {predictions}")


# 9) All script execution guarded
if __name__ == "__main__":
    main()