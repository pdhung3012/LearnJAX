import torch
import torch.nn as nn
import torch.optim as optim


def generate_data():
    X = torch.rand(100, 2) * 10
    y = (X[:, 0] + X[:, 1] * 2).unsqueeze(1) + torch.randn(100, 1)
    return X, y


class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def make_model():
    return DNNModel()


def make_criterion():
    return nn.MSELoss()


def make_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.01)


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

    X_test = torch.tensor([[4.0, 3.0], [7.0, 8.0]])
    with torch.no_grad():
        predictions = model(X_test)
        print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")


if __name__ == '__main__':
    main()
