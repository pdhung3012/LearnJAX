import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.example_libraries import optimizers
import matplotlib.pyplot as plt

# Generate synthetic data
key = jax.random.PRNGKey(42)
X = jax.random.uniform(key, shape=(100, 1)) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + jax.random.normal(key, shape=(100, 1))  # Linear relationship with noise

# Define the Linear Regression Model within a CustomActivationModel class
class CustomActivationModel:
    def __init__(self):
        self.w = jnp.zeros((1, 1))
        self.b = jnp.zeros((1, 1))

    def custom_activation(self, x):
        return jnp.tanh(x) + x

    def forward(self, params, x):
        w, b = params
        return self.custom_activation(jnp.dot(x, w) + b)

# Initialize the model, loss function, and optimizer
model = CustomActivationModel()
def loss_fn(params, X, y):
    predictions = model.forward(params, X)
    return jnp.mean((predictions - y) ** 2)

grad_loss_fn = grad(loss_fn, argnums=0)

# Training loop
opt_init, opt_update, get_params = optimizers.sgd(step_size=0.01)
opt_state = opt_init((model.w, model.b))

@jit
def update(i, opt_state, X, y):
    params = get_params(opt_state)
    grads = grad_loss_fn(params, X, y)
    return opt_update(i, grads, opt_state)

epochs = 1000
for epoch in range(epochs):
    opt_state = update(epoch, opt_state, X, y)

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        params = get_params(opt_state)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss_fn(params, X, y):.4f}")

# Display the learned parameters
params = get_params(opt_state)
print(f"Learned weight: {params[0].item():.4f}, Learned bias: {params[1].item():.4f}")

# Plot the model fit to the train data
plt.figure(figsize=(4, 4))
plt.scatter(X, y, label='Training Data')
plt.plot(X, params[0]*X + params[1], 'r', label='Model Fit')
plt.legend()
plt.show()

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = model.forward(params, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
