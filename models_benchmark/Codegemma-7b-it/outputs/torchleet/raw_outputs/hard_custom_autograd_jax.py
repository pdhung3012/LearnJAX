import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Generate synthetic data
key = jax.random.PRNGKey(42)
X = jax.random.uniform(key, (100, 1), minval=0, maxval=10)
y = 2 * X + 3 + jax.random.normal(key, (100, 1))

# Define the custom activation function
class LearnedSiLUFunction(jax.nn.Module):
    def __init__(self, slope=1):
        super().__init__()
        self.slope = slope

    def __call__(self, x):
        return self.slope * x * jax.nn.sigmoid(x)

# Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def setup(self, slope=1):
        self.slope = self.param("slope", (1,), init=lambda key: jnp.ones(key, dtype=jnp.float32) * slope)

    def __call__(self, x):
        # Use the custom LearnedSiLUFunction
        return LearnedSiLUFunction(self.slope)(x)

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = optax.l2_loss
optimizer = optax.sgd(learning_rate=0.01)

# Training loop
@jax.jit
def train_step(params, batch_stats, x, y):
    def loss_fn(params):
        predictions = model.apply({"params": params, "batch_stats": batch_stats}, x)
        return criterion(predictions, y)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(params)
    updates, new_params = optimizer.update(grads, params)
    new_params = optax.apply_updates(params, updates)
    new_batch_stats = optax.update_batch_stats(
        batch_stats,
        predictions,
        y,
        momentum=0.9,
        epsilon=1e-5,
    )
    return new_params, new_batch_stats, loss

epochs = 1000
for epoch in range(epochs):
    params, batch_stats, loss = train_step(model.init(key, X), None, X, y)

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Display the learned parameters
w, b = model.apply({"params": params, "batch_stats": batch_stats}, jnp.array([[4.0], [7.0]]))
print(f"Learned weight: {w[0][0]:.4f}, Learned bias: {b[0][0]:.4f}")

# Testing on new data
predictions = model.apply({"params": params, "batch_stats": batch_stats}, jnp.array([[4.0], [7.0]]))
print(f"Predictions for [[4.0], [7.0]]: {predictions.tolist()}")