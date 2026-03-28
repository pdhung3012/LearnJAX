import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Generate synthetic data
key = jax.random.PRNGKey(42)
X = jax.random.uniform(key, (100, 1), minval=0, maxval=10)
y = 2 * X + 3 + jax.random.normal(key, (100, 1))

# Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1)(x)
        return x

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
loss_fn = optax.l2_loss
optimizer = optax.sgd(learning_rate=0.01)

# Training loop
@jax.jit
def train_step(params, batch_stats, x, y):
    def loss_with_grad(params):
        predictions = model.apply(params, batch_stats, x)
        loss = loss_fn(predictions, y)
        return loss, jax.grad(loss)(params)

    updates, params = optax.apply_updates(loss_with_grad(params), params)
    batch_stats = optax.update_batch_stats(updates['batch_stats'], x, updates['mean'], updates['var'])
    return params, batch_stats

@jax.jit
def evaluate(params, batch_stats, x):
    predictions = model.apply(params, batch_stats, x)
    return predictions

# Training
num_epochs = 1000
for epoch in range(num_epochs):
    params, batch_stats = train_step(params, batch_stats, X, y)

    if (epoch + 1) % 100 == 0:
        predictions = evaluate(params, batch_stats, X)
        loss = loss_fn(predictions, y)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")

# Display the learned parameters
params = jax.tree_util.tree_map(lambda x: x.item(), params)
print(f"Learned weight: {params['dense_kernel'][0]:.4f}, Learned bias: {params['dense_bias'][0]:.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = evaluate(params, batch_stats, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")