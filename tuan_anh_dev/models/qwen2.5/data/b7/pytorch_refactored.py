import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def generate_data():
    X = torch.rand(100, 1) * 10
    y = 3 * X + 5 + torch.randn(100, 1)
    return X, y


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def make_model():
    return LinearRegressionModel()


def make_criterion():
    return nn.MSELoss()


def make_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01)


def train_model(X, y, model, optimizer, criterion, num_epochs, writer):
    for epoch in range(num_epochs):
        predictions = model(X)
        loss = criterion(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/train", loss.item(), epoch)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


def main():
    torch.manual_seed(42)
    X, y = generate_data()
    # NOTE: writer created before model to match original module-level execution order.
    writer = SummaryWriter(log_dir="runs/linear_regression")
    model = make_model()
    criterion = make_criterion()
    optimizer = make_optimizer(model)
    train_model(X, y, model, optimizer, criterion, num_epochs=100, writer=writer)
    writer.close()


if __name__ == '__main__':
    main()
