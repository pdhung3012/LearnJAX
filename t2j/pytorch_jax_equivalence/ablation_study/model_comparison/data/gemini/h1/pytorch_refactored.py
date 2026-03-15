import torch
import torch.nn as nn
import torch.optim as optim

# Rule 3: Model class defined at module level
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# Rule 2: Wrap data generation
def generate_data():
    X = torch.rand(100, 1)
    y = 3 * X + 2 + torch.randn(100, 1) * 0.1
    return X, y

# Rule 4: Instantiate model
def make_model():
    return SimpleModel()

# Rule 5: Return loss function
def make_criterion():
    return nn.MSELoss()

# Rule 6: Return optimizer with same hyperparameters
def make_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01)

# Rule 7: Wrap training loop with identical logic
def train_model(X, y, model, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

def main():
    # Rule 8: Manual seed at the top
    torch.manual_seed(42)
    
    # Preservation of random-initialization order:
    # 1. Model weights are initialized first in the original code.
    model = make_model()
    
    # 2. Data tensors (rand and randn) are generated second.
    X, y = generate_data()
    
    criterion = make_criterion()
    optimizer = make_optimizer(model)
    
    # Training
    epochs = 100
    train_model(X, y, model, optimizer, criterion, epochs)

    # Save the model
    torch.save(model.state_dict(), "model.pth")

    # Load the model back
    loaded_model = SimpleModel()
    loaded_model.load_state_dict(torch.load("model.pth"))
    loaded_model.eval()

    # Rule 10: Verify identical output
    X_test = torch.tensor([[0.5], [1.0], [1.5]])
    with torch.no_grad():
        predictions = loaded_model(X_test)
        print(f"Predictions after loading: {predictions}")

# Rule 9: Script execution guard
if __name__ == '__main__':
    main()