import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Generate synthetic data
rng = jax.random.PRNGKey(42)
X = jax.random.uniform(rng, (100, 2), minval=0, maxval=10)
y = (X[:, 0] + X[:, 1] * 2).reshape(-1, 1) + jax.random.normal(rng, (100, 1))

# Define the Deep Neural Network Model
class DNNModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(10)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

# Initialize the model, loss function, and optimizer
model = DNNModel()
loss_fn = optax.l2_loss
optimizer = optax.adam(learning_rate=0.01)

# Training loop
@jax.jit
def train_step(params, batch):
    x, y = batch
    def loss_fn_with_params(params, x, y):
        predictions = model.apply({"params": params}, x)
        loss = loss_fn(predictions, y)
        return loss, predictions
    grad_fn = jax.value_and_grad(loss_fn_with_params, has_aux=True)
    (loss, predictions), grads = grad_fn(params, x, y)
    updates, new_params = optimizer.update(grads, params)
    params = optax.apply_updates(params, updates)
    return loss, predictions, new_params

@jax.jit
def run_epoch(params, data):
    losses = []
    predictions = []
    for batch in data:
        loss, prediction, params = train_step(params, batch)
        losses.append(loss)
        predictions.append(prediction)
    return jnp.mean(losses), predictions, params

num_epochs = 1000
for epoch in range(num_epochs):
    loss, predictions, params = run_epoch(params, [(X, y)])
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")

# Testing on new data
X_test = jnp.array([[4.0, 3.0], [7.0, 8.0]])
predictions = model.apply({"params": params}, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")