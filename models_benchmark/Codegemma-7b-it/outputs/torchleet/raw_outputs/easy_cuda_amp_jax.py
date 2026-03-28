import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Define a simple model
class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(10)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x

# Generate synthetic data
X = jnp.random.randn(1000, 10)
y = jnp.random.randn(1000, 1)

# Initialize model, loss function, and optimizer
model = SimpleModel()
criterion = optax.l2_loss
optimizer = optax.adam(learning_rate=0.001)

# Training loop
@jax.jit
def train_step(params, batch):
    x, y = batch
    with jax.value_and_grad(criterion, has_aux=True) as (loss_fn, grad_fn):
        predictions = model.apply(params, x)
        loss = loss_fn(predictions, y)
    updates, new_params = optimizer.update(grad_fn(params, x), params)
    params = optax.apply_updates(params, updates)
    return loss, new_params

@jax.jit
def test_step(params, x):
    predictions = model.apply(params, x)
    return predictions

epochs = 5
for epoch in range(epochs):
    for x, y in zip(X, y):
        loss, params = train_step(params, (x, y))

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Test the model on new data
X_test = jnp.random.randn(5, 10)
predictions = test_step(params, X_test)
print("Predictions:", predictions)