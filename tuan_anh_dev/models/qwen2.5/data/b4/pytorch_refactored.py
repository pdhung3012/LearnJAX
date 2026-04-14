import torch
import torch.nn as nn
import torch.optim as optim


def generate_data():
    X = torch.rand(100, 1) * 10
    y = 2 * X + 3 + torch.randn(100, 1)
    return X, y


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        error = torch.abs(y_pred - y_true)
        loss = torch.where(error <= self.delta,
                           0.5 * error**2,
                           self.delta * (error - 0.5 * self.delta))
        return loss.mean()


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def make_model():
    return LinearRegressionModel()


def make_criterion():
    return HuberLoss(delta=1.0)


def make_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01)


def train_model(X, y, model, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        predictions = model(X)
        loss = criterion(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


def main():
    torch.manual_seed(42)
    X, y = generate_data()
    model = make_model()
    criterion = make_criterion()
    optimizer = make_optimizer(model)
    train_model(X, y, model, optimizer, criterion, num_epochs=1000)

    [w, b] = model.linear.parameters()
    print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

    X_test = torch.tensor([[4.0], [7.0]])
    with torch.no_grad():
        predictions = model(X_test)
        print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


if __name__ == '__main__':
    main()
