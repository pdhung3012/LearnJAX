import torch
import torch.nn as nn
import torch.optim as optim

# Define a Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, ff_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pooling across the sequence
        return self.output_layer(x)

# Generate synthetic data
torch.manual_seed(42)
seq_length = 10
num_samples = 100
input_dim = 1
X = torch.rand(num_samples, seq_length, input_dim)  # Random sequences
y = torch.sum(X, dim=1)  # Target is the sum of each sequence

# Initialize the model, loss function, and optimizer
input_dim = 1
embed_dim = 16
num_heads = 2
num_layers = 2
ff_dim = 64
output_dim = 1

model = TransformerModel(input_dim, embed_dim, num_heads, num_layers, ff_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Testing on new data
X_test = torch.rand(2, seq_length, input_dim)
with torch.no_grad():
    predictions = model(X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")