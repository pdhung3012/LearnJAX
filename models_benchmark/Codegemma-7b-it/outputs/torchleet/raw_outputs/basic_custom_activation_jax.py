import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Generate synthetic data
key = jax.random.PRNGKey(42)
X = jax.random.uniform(key, (100, 1), minval=0, maxval=10)
y = 2 * X + 3 + jax.random.normal(key, (100, 1))

# Define the Linear Regression Model within a CustomActivationModel class
class CustomActivationModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1)(x)
        return nn.tanh(x) + x

# Initialize the model, loss function, and optimizer
model = CustomActivationModel()
loss_fn = optax.l2_loss
optimizer = optax.sgd(learning_rate=0.01)

# Training loop
@jax.jit
def train_step(params, batch):
    x, y = batch
    predictions = model.apply(params, x)
    loss = loss_fn(predictions, y)
    grads = jax.grad(loss)(params)
    updates, new_params = optimizer.update(grads, params)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params

@jax.jit
def run_epoch(params, data):
    losses = []
    for batch in data:
        loss, params = train_step(params, batch)
        losses.append(loss)
    return jnp.mean(losses), params

num_epochs = 1000
params = model.init(key, jnp.ones((1,)))
for epoch in range(num_epochs):
    loss, params = run_epoch(params, [(X, y)])
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")

# Display the learned parameters
w, b = jax.tree_util.tree_map(lambda x: x[0], params)
print(f"Learned weight: {w:.4f}, Learned bias: {b:.4f}")

# Plot the model fit to the train data
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 4))
plt.scatter(X, y, label='Training Data')
plt.plot(X, w * X + b, 'r', label='Model Fit')
plt.legend()
plt.show()

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = model.apply(params, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")