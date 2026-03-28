import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Generate random data
rng = jax.random.PRNGKey(42)
X = jax.random.uniform(rng, (100, 1), minval=0, maxval=10)
y = 2 * X + 3 + jax.random.normal(rng, (100, 1))

# Define the model
class LinearRegression(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1)(x)
        return x

# Initialize the model, loss function, and optimizer
model = LinearRegression()
loss_fn = optax.l2_loss
optimizer = optax.sgd(learning_rate=0.01)

# Training loop
@jax.jit
def train_step(params, batch_X, batch_y):
    def loss_fn_with_params(params, batch_X, batch_y):
        predictions = model.apply({'params': params}, batch_X)
        loss = jnp.mean(optax.l2_loss(predictions, batch_y))
        return loss, predictions

    grad_fn = jax.value_and_grad(loss_fn_with_params, has_aux=True)
    (loss, predictions), grads = grad_fn(params, batch_X, batch_y)
    updates, new_params = optimizer.update(grads, params)
    params = optax.apply_updates(params, updates)
    return new_params, loss, predictions

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    for batch_X, batch_y in zip(jnp.array_split(X, 10), jnp.array_split(y, 10)):
        params, loss, _ = train_step(params, batch_X, batch_y)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")

# Display the learned parameters
params = jax.tree_util.tree_map(lambda x: x.item(), params)
print(f"Learned weight: {params['params']['dense']['kernel'][0][0]:.4f}, Learned bias: {params['params']['dense']['bias'][0]:.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = model.apply({'params': params}, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")