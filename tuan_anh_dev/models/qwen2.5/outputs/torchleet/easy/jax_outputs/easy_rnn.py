import jax
import jax.numpy as jnp
from jax import jit
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np

# Generate synthetic sequential data
sequence_length = 10
num_samples = 100

# Create a sine wave dataset
y = jnp.sin(jnp.linspace(0, 4 * 3.14159, num=num_samples)).reshape(-1, 1)

# Prepare data for RNN
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i : i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.array(in_seq), jnp.array(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

# Define the RNN Model using Flax (matching PyTorch: manual RNN cell + Linear output)
class RNNModel(nn.Module):
    hidden_dim: int = 50
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        input_dim = x.shape[-1]
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Weight matrices for input and hidden state
        W_ih = self.param('W_ih', lambda rng, shape: jax.random.normal(rng, shape) * 0.1, (input_dim, self.hidden_dim))
        W_hh = self.param('W_hh', lambda rng, shape: jax.random.normal(rng, shape) * 0.1, (self.hidden_dim, self.hidden_dim))
        b_h = self.param('b_h', nn.initializers.zeros, (self.hidden_dim,))

        h_t = jnp.zeros((batch_size, self.hidden_dim))

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = jnp.tanh(x_t @ W_ih + h_t @ W_hh + b_h)

        # Output layer (matching PyTorch nn.Linear)
        output = nn.Dense(self.output_dim)(h_t)
        return output

# Initialize the model, loss function, and optimizer
model = RNNModel(hidden_dim=50, output_dim=1)
dummy_input = jnp.ones([1, sequence_length, 1])
variables = model.init(jax.random.PRNGKey(42), dummy_input)

tx = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx
)

@jit
def train_step(state, sequences, labels):
    def loss_fn(params):
        predictions = model.apply({'params': params}, sequences)
        return jnp.mean((predictions - labels) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Training loop
epochs = 500
for epoch in range(epochs):
    for i in range(len(X_seq)):
        sequences = X_seq[i:i+1]  # Add batch dimension (1, seq_length, 1)
        labels = y_seq[i:i+1]      # Add batch dimension (1, 1)

        state, loss = train_step(state, sequences, labels)

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Testing on new data
X_test = jnp.sin(jnp.linspace(4 * 3.14159, 8 * 3.14159, num=100)).reshape(-1, 1)

# Reshape to (batch_size, sequence_length, input_size)
X_test = X_test[None, :, :]  # Add batch dimension, shape becomes (1, 100, 1)

predictions = model.apply({'params': state.params}, X_test)
print(f"Preceding three values: {X_test[:, -3:, :].tolist()}")
print(f"Predictions for new sequence: {predictions.tolist()}")
