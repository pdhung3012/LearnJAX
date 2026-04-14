import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd


def generate_data():
    X = torch.rand(100, 1) * 10
    y = 2 * X + 3 + torch.randn(100, 1)
    data = torch.cat((X, y), dim=1)
    df = pd.DataFrame(data.numpy(), columns=['X', 'y'])
    df.to_csv('data.csv', index=False)
    return X, y


class LinearRegressionDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.X = torch.tensor(self.data['X'].values, dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(self.data['y'].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def train_model(X, y, model, optimizer, criterion, num_epochs):
    # DataLoader created here to preserve original logic; shuffle=True uses
    # global RNG only during iteration, so RNG state at first batch is identical
    # to the original regardless of where DataLoader is instantiated.
    dataset = LinearRegressionDataset('data.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(num_epochs):
        for batch_X, batch_y in dataloader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
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
