import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from flax import optax

class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, X):
        fc = self.fc
        return fc(X)

    @nn.compact
    def setup(self):
        self.fc = nn.Dense(1)

class SimpleModelInit(nn.Module):
    @nn.compact
    def __call__(self):
        return SimpleModel()

X = jnp.ones((1000, 10))
key = jr.PRNGKey(0)
y = jnp.ones((1000, 1)) + jr.uniform(key, (1000, 1))

# Generate synthetic data
params = SimpleModelInit().init(key, jnp.ones((1,)))

# Training loop
@jax.jit
def train_step(params, X, y):
    grad_fn = jax.value_and_grad(params.fc)(X)
    loss, grads = grad_fn(y)
    mse_loss = jnp.mean(jnp.square(loss))

    updates, _ = optax.adam(0.001).update(params, (-grads,))
    return updates, mse_loss

@jax.jit
def train(params, X, y, epochs):
    for epoch in range(epochs):
        for i in range(len(X) // 32):
            X_batch, y_batch = X[i*32:i*32+32], y[i*32:i*32+32]
            updates, loss = train_step(params, X_batch, y_batch)
            params = updates

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Initialize model
params = SimpleModelInit().init(key, jnp.ones((1,)))

# Training
train(params, X, y, 5)

# Test the model on new data
X_test = jnp.ones((5, 10))
predictions = params.fc.apply(X_test)
print("Predictions:", predictions)

# Test the model on new data
X_test = jnp.ones((5, 10))
predictions = params.fc.apply(X_test)
print("Predictions:", predictions)


This JAX code replicates the PyTorch model architecture, generates synthetic data, and includes a training loop using `jax.jit` and `jax.value_and_grad`. Note that the data generation is simplified using NumPy-like JAX functions. The training loop uses a specific `train_step` function to handle the forward and backward passes.