import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

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
    return jnp.stack(in_seq), jnp.stack(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

# Define the RNN Model
class RNNModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, _ = x.shape
        hidden_dim = 50
        output_dim = 1

        # Weight matrices for input and hidden state
        W_ih = self.param("W_ih", (1, hidden_dim), jax.nn.initializers.glorot_uniform())
        W_hh = self.param("W_hh", (hidden_dim, hidden_dim), jax.nn.initializers.glorot_uniform())
        b_h = self.param("b_h", (hidden_dim,), jax.nn.initializers.zeros)

        # Output layer
        output_layer = nn.Dense(output_dim)

        # Activation
        tanh = jax.nn.tanh

        h_t = jnp.zeros((batch_size, hidden_dim))

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = tanh(x_t @ W_ih + h_t @ W_hh + b_h)

        output = output_layer(h_t)
        return output

# Initialize the model, loss function, and optimizer
model = RNNModel()
criterion = jax.nn.MSELoss()
optimizer = optax.adam(learning_rate=0.001)

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
        grad_fn = jax.value_and_grad(loss)
        loss_value, grads = grad_fn(model.apply)
        optimizer.apply_updates(model.state, grads)

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss_value:.4f}")

# Testing on new data
X_test = torch.sin(torch.linspace(4 * 3.14159, 8 * 3.14159, steps=100).unsqueeze(1))

# Reshape to (batch_size, sequence_length, input_size)
X_test = X_test.unsqueeze(0)  # Add batch dimension, shape becomes (1, 100, 1)

with jax.no_grad():
    predictions = model(X_test) # Predict the next value of the sine wave.
    print(f"Preceding three values: {X_test[:, -3:, :].tolist()}")
    print(f"Predictions for new sequence: {predictions.tolist()}")