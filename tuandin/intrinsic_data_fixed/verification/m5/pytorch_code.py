import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic sequential data
torch.manual_seed(42)
sequence_length = 10
num_samples = 100

# Create a sine wave dataset
X = torch.linspace(0, 4 * 3.14159, steps=num_samples).unsqueeze(1)
y = torch.sin(X)

# Prepare data for RNN
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return torch.stack(in_seq), torch.stack(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

# Define the RNN Model
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.fc = nn.Linear(50, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the RNN
        return out

# Initialize the model, loss function, and optimizer
model = RNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 500
for epoch in range(epochs):
    for sequences, labels in zip(X_seq, y_seq):
        sequences = sequences.unsqueeze(0)  # Add batch dimension
        labels = labels.unsqueeze(0)  # Add batch dimension

        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Testing on new data
X_test = torch.linspace(4 * 3.14159, 5 * 3.14159, steps=10).unsqueeze(1)

# Reshape to (batch_size, sequence_length, input_size)
X_test = X_test.unsqueeze(0)  # Add batch dimension, shape becomes (1, 10, 1)

with torch.no_grad():
    predictions = model(X_test)
    print(f"Predictions for new sequence: {predictions.tolist()}")