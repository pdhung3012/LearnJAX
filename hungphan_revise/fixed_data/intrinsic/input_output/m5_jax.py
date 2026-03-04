import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np

# -------------------------------
# Data Preparation
# -------------------------------
sequence_length = 10
num_samples = 100

# Create a sine wave dataset (using 4 * pi as the end point)
X = jnp.linspace(0, 4 * jnp.pi, num=num_samples)[:, None]
y = jnp.sin(X)

def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.stack(in_seq), jnp.stack(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

# -------------------------------
# Define the RNN Model in Flax
# -------------------------------
class RNNCell(nn.Module):
    hidden_size: int = 50

    @nn.compact
    def __call__(self, carry, x):
        # carry: previous hidden state, shape (batch, hidden_size)
        # x: current input, shape (batch, input_size)
        # Compute new hidden state: tanh(W_ih*x + W_hh*carry + b)
        new_h = nn.tanh(
            nn.Dense(self.hidden_size, name="ih")(x) +
            nn.Dense(self.hidden_size, use_bias=False, name="hh")(carry)
        )
        return new_h, new_h  # returning new state as both carry and output

class RNNModel(nn.Module):
    hidden_size: int = 50

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]
        init_carry = jnp.zeros((batch_size, self.hidden_size))
        rnn_scan = nn.scan(
            RNNCell,
            in_axes=1,
            out_axes=1,
            variable_broadcast="params",
            split_rngs={"params": False},
        )(hidden_size=self.hidden_size)
        carry, ys = rnn_scan(init_carry, x)
        last_output = ys[:, -1, :]
        output = nn.Dense(1)(last_output)
        output = nn.relu(output)  # Add this line!
        return output

# -------------------------------
# Initialize the Model and Optimizer
# -------------------------------
model = RNNModel()
rng = jax.random.PRNGKey(42)
# Sample input with shape (batch=1, seq_length, input_size)
sample_input = jnp.ones((1, sequence_length, 1))
params = model.init(rng, sample_input)

# Set up the Adam optimizer from Optax
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# -------------------------------
# Define Loss and Training Step
# -------------------------------
def loss_fn(params, x, y):
    preds = model.apply(params, x)
    return jnp.mean((preds - y) ** 2)

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# -------------------------------
# Training Loop
# -------------------------------
epochs = 500
for epoch in range(epochs):
    for seq, label in zip(X_seq, y_seq):
        seq = seq[None, :, :]
        label = label[None, :]
        params, opt_state, loss = train_step(params, opt_state, seq, label)
    # loss variable now holds the last sample's loss
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# -------------------------------
# Testing on New Data
# -------------------------------
# Create new test data (from 4*pi to 5*pi)
X_test = jnp.linspace(4 * jnp.pi, 5 * jnp.pi, num=10)[:, None]
X_test = X_test[None, :, :]  # Add batch dimension: shape becomes (1, 10, 1)
predictions = model.apply(params, X_test)
print("Predictions for new sequence:", predictions)
