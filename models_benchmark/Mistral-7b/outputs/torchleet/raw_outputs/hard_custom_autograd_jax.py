import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from flax import optax

# Generate synthetic data
rng = jr.PRNGKey(42)
X = jnp.ones((100, 1)) * 10
y = 2 * jnp.ones((100, 1)) * jnp.asarray(jr.normal(rng, (100, 1)), dtype=jnp.float32) + 3 + jr.normal(rng, (100, 1))

class LearnedSiLUFunction(nn.Module):
    @nn.compact
    def __call__(self, x, slope):
        return slope * jnp.sigmoid(x)

class LinearRegressionModel(nn.Module):
    @nn.compact
    def __init__(self, slope=1.0):
        super().__init__()
        self.slope = self.param("slope", nn.initializers.zeros, (1,))

    @nn.compact
    def __call__(self, x):
        return self.apply(x, self.slope)

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optax.sgd(step_size=0.01)

@jax.jit
def train_step(params, X, y):
    grad_fn = jax.grad(criterion)
    loss, grads = jax.value_and_grad(criterion)(model.apply(X, params))
    updates, _ = optimizer.update(params, grads)
    return updates, loss

# Training loop
rng = jr.PRNGKey(42)
epochs = 1000
for epoch in range(epochs):
    X_batch, _ = jr.split(jr.batch(X, 32), 2)
    y_batch = jr.batch(y, 32)

    updates, loss = train_step(model.params, X_batch, y_batch)
    model = model.apply_updates(updates)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Display the learned parameters
w, b = model.slope
print(f"Learned weight: {w[0]:.4f}, Learned bias: {b[0]:.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
with jax.grad():
    predictions = model(X_test)
print(f"Predictions for {X_test}: {predictions.tolist()}")


This JAX code replicates the PyTorch code strictly using `flax.linen.Module`, `jax.jit`, and handles the state explicitly. The training loop is converted to use `jax.value_and_grad` and `@jax.jit`. The data is generated using simple `numpy` random data generators.