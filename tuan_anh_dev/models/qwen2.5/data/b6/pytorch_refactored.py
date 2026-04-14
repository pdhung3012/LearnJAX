import torch
import torch.nn as nn
import torch.optim as optim


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


def make_model():
    return SimpleModel()


def make_criterion():
    return nn.MSELoss()


def make_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01)


def generate_data():
    X = torch.rand(100, 1)
    y = 3 * X + 2 + torch.randn(100, 1) * 0.1
    return X, y


def train_model(X, y, model, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()


def main():
    torch.manual_seed(42)
    # NOTE: model is initialised before data to match original module-level
    # execution order (make_model consumes RNG for weight init before generate_data).
    model = make_model()
    criterion = make_criterion()
    optimizer = make_optimizer(model)
    X, y = generate_data()
    train_model(X, y, model, optimizer, criterion, num_epochs=100)

    torch.save(model.state_dict(), "model.pth")

    loaded_model = SimpleModel()
    loaded_model.load_state_dict(torch.load("model.pth"))
    loaded_model.eval()

    X_test = torch.tensor([[0.5], [1.0], [1.5]])
    with torch.no_grad():
        predictions = loaded_model(X_test)
        print(f"Predictions after loading: {predictions}")


if __name__ == '__main__':
    main()
