import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic sequential data
torch.manual_seed(42)
sequence_length = 10
num_samples = 100

# Create a sine wave dataset
y = torch.sin(torch.linspace(0, 4 * 3.14159, steps=num_samples).unsqueeze(1))


# Prepare data for RNN
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i : i + seq_length])
        out_seq.append(data[i + seq_length])
    return torch.stack(in_seq), torch.stack(out_seq)


X_seq, y_seq = create_in_out_sequences(y, sequence_length)

# Define the RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Weight matrices for input and hidden state
        self.W_ih = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Activation
        self.tanh = nn.Tanh()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = self.tanh(x_t @ self.W_ih + h_t @ self.W_hh + self.b_h)

        output = self.output_layer(h_t)
        return output

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
X_test = torch.sin(torch.linspace(4 * 3.14159, 8 * 3.14159, steps=100).unsqueeze(1))

# Reshape to (batch_size, sequence_length, input_size)
X_test = X_test.unsqueeze(0)  # Add batch dimension, shape becomes (1, 100, 1)

with torch.no_grad():
    predictions = model(X_test) # Predict the next value of the sine wave.
    print(f"Preceding three values: {X_test[:, -3:, :].tolist()}")
    print(f"Predictions for new sequence: {predictions.tolist()}")
