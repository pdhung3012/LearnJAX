# Implement mixed precision training in JAX using bfloat16 (equivalent to torch.cuda.amp)
import jax
import jax.numpy as jnp
from jax import grad, jit
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np

# Define a simple model
class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x)

# Generate synthetic data (fixed seed for reproducibility)
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
X = jax.random.normal(key1, (1000, 10))
y = jax.random.normal(key2, (1000, 1))

# Initialize model, loss function, and optimizer
model = SimpleModel()
dummy_input = jnp.ones([1, 10])
variables = model.init(jax.random.PRNGKey(0), dummy_input)

tx = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx
)

# Mixed precision: use bfloat16 for forward pass (JAX equivalent of torch.cuda.amp.autocast)
@jit
def train_step(state, inputs, labels):
    def loss_fn(params):
        # Cast inputs to bfloat16 for mixed precision forward pass
        inputs_bf16 = inputs.astype(jnp.bfloat16)
        predictions = model.apply({'params': jax.tree.map(lambda p: p.astype(jnp.bfloat16), params)}, inputs_bf16)
        predictions = predictions.astype(jnp.float32)  # Cast back to float32 for loss
        return jnp.mean((predictions - labels) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Simple data loader
def data_loader(X, y, batch_size, shuffle=True):
    n = len(X)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]

# Training loop
epochs = 5
for epoch in range(epochs):
    for inputs, labels in data_loader(X, y, batch_size=32, shuffle=True):
        state, loss = train_step(state, inputs, labels)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Test the model on new data
key_test = jax.random.PRNGKey(99)
X_test = jax.random.normal(key_test, (5, 10))
predictions = model.apply({'params': state.params}, X_test)
print("Predictions:", predictions)
