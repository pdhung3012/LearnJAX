import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Generate synthetic data
key = jax.random.PRNGKey(42)
X = jax.random.uniform(key, (100, 1), minval=0, maxval=10)
y = 2 * X + 3 + jax.random.normal(key, (100, 1))

# Define the Huber Loss function
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def apply(self, y_pred, y_true):
        error = jnp.abs(y_pred - y_true)
        loss = jnp.where(error <= self.delta,
                           0.5 * error**2,  # L2 loss for small errors
                           self.delta * (error - 0.5 * self.delta))  # L1 loss for large errors
        return loss.mean()  # Return the mean loss across all samples

# Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1)(x)
        return x

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = HuberLoss(delta=1.0)
optimizer = optax.sgd(learning_rate=0.01)

# Training loop
@jax.jit
def train_step(params, batch_stats, x, y):
    def loss_fn(params):
        predictions = model.apply({'params': params, **batch_stats}, x)
        loss = criterion(predictions, y)
        return loss, (predictions, batch_stats)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    loss, (predictions, batch_stats) = grad_fn(params)
    updates, new_params = optimizer.update(loss, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, batch_stats, loss, predictions

@jax.jit
def update_batch_stats(batch_stats, predictions, y):
    new_batch_stats = optax.update_batch_stats(
        batch_stats,
        (predictions, y),
        momentum=0.9,
        epsilon=1e-5,
    )
    return new_batch_stats

epochs = 1000
for epoch in range(epochs):
    params, batch_stats, loss, predictions = train_step(model.init(key, X), None, X, y)
    batch_stats = update_batch_stats(batch_stats, predictions, y)

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Display the learned parameters
w, b = jax.tree_util.tree_leaves(params)[0]
print(f"Learned weight: {w[0]:.4f}, Learned bias: {b[0]:.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = model.apply({'params': params, **batch_stats}, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")