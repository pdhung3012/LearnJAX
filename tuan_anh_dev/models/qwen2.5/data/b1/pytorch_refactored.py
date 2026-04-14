import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt


def generate_data():
    X = torch.rand(100, 1) * 10
    y = 2 * X + 3 + torch.randn(100, 1)
    return X, y


class CustomActivationModel(nn.Module):
    def __init__(self):
        super(CustomActivationModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def custom_activation(self, x):
        return torch.tanh(x) + x

    def forward(self, x):
        return self.custom_activation(self.linear(x))


def make_model():
    return CustomActivationModel()


def make_criterion():
    return nn.MSELoss()


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

    plt.figure(figsize=(4, 4))
    plt.scatter(X, y, label='Training Data')
    plt.plot(X, w.item()*X + b.item(), 'r', label='Model Fit')
    plt.legend()
    plt.show()

    X_test = torch.tensor([[4.0], [7.0]])
    with torch.no_grad():
        predictions = model(X_test)
        print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


if __name__ == '__main__':
    main()
