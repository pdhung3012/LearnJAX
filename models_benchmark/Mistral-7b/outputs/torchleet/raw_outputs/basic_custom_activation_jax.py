import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from flax import optax
import matplotlib.pyplot as plt

# Generate synthetic data
rng = jr.PRNGKey(42)
X = jnp.linspace(0, 10, 100)  # 100 data points between 0 and 10
y = 2 * X + 3 + jr.normal(rng, (100, 1))  # Linear relationship with noise

# Define the rng key for reproducibility
rng_train = jr.PRNGKey(42)

# Define the Linear Regression Model within a CustomActivationModel class
class CustomActivationModel(nn.Module):
    @nn.compact
    def __call__(self, X):
        x = X[:, jnp.newaxis]
        w = self.params["w"]
        b = self.params["b"]
        return jnp.tanh(x * w) + x * w + b

    @nn.compact
    def setup(self):
        self.params = self.init_params()

    def init_params(self):
        rng = jr.PRNGKey(42)
        return {"w": jr.normal(rng, (1,)), "b": jr.normal(rng, (1,))}

# Initialize the model, loss function, and optimizer
model = CustomActivationModel()

# Define the loss function
def loss(params, X, y):
    predictions = model.apply(params, X)
    return jnp.mean((predictions - y) ** 2)

# Initialize the optimizer
optimizer = optax.sgd(0.01)

# Training loop
@jax.jit
def train_step(params, X, y):
    grads = jax.value_and_grad(loss)(params, X, y)
    return optimizer.update(params, grads)

# Training loop
for epoch in range(1000):
    params = train_step(params, X, y)[0]

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{1000}], Loss: {loss(params, X, y):.4f}")

# Display the learned parameters
w, b = model.params["w"], model.params["b"]
print(f"Learned weight: {w.item()}, Learned bias: {b.item()}")

# Plot the model fit to the train data
plt.figure(figsize=(4, 4))
plt.scatter(X, y, label='Training Data')
plt.plot(X, jnp.tanh(X * w) + X * w + b, 'r', label='Model Fit')
plt.legend()
plt.show()

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
with jax.device("cpu:0"):
    predictions = model.apply(model.params, X_test)
print(f"Predictions for {X_test}: {predictions.tolist()}")


This JAX code replicates the PyTorch code strictly using `flax.linen.Module`, `jax.numpy`, `jax`, and `optax`. The training loop is converted to use `jax.value_and_grad` and `@jax.jit`. The data is generated using simple `numpy` random data generators.