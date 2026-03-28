import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Generate synthetic data
key = jax.random.PRNGKey(42)
X = jax.random.uniform(key, (100, 1), minval=0, maxval=10)
y = 3 * X + 5 + jax.random.normal(key, (100, 1))

# Define a simple Linear Regression Model
class LinearRegressionModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1)(x)
        return x

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optax.sgd(learning_rate=0.01)

# Training loop
@jax.jit
def train_step(params, batch_stats, x, y):
    def loss_fn(params):
        predictions = model.apply({"params": params, "batch_stats": batch_stats}, x)
        loss = criterion(predictions, y)
        return loss, predictions

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    loss, predictions = grad_fn(params)
    updates, new_params = optimizer.update(grad_fn, params)
    new_params = optax.apply_updates(params, updates)
    new_batch_stats = optax.update_batch_stats(batch_stats, predictions, y)
    return new_params, new_batch_stats, loss

epochs = 100
params = model.init(key, jnp.ones((1,)))
batch_stats = optax.initialize_batch_stats(model)

for epoch in range(epochs):
    params, batch_stats, loss = train_step(params, batch_stats, X, y)

    # Log loss to TensorBoard (assuming TensorBoard is initialized)
    writer.add_scalar("Loss/train", loss.item(), epoch)

    # Log progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Close the TensorBoard writer
writer.close()