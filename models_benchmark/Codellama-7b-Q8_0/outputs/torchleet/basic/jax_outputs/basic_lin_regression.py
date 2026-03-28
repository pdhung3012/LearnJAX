import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers

# Generate synthetic data
key = jax.random.PRNGKey(42)
key, key_x, key_noise = jax.random.split(key, 3)
X = jax.random.uniform(key_x, shape=(100, 1)) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + jax.random.normal(key_noise, shape=(100, 1))  # Linear relationship with noise

# Define the Linear Regression Model
def linear_regression(params, x):
    w, b = params
    return jnp.dot(x, w) + b

# Initialize the model parameters (matching nn.Linear(1, 1))
params = (jnp.zeros((1, 1)), jnp.zeros((1,)))

# Define the loss function
def mse_loss(params, x, y):
    predictions = linear_regression(params, x)
    return jnp.mean((predictions - y) ** 2)

# Define the gradient of the loss function
grad_mse_loss = grad(mse_loss)

# Training loop
opt_init, opt_update, get_params = optimizers.sgd(step_size=0.01)
opt_state = opt_init(params)

@jit
def update(i, opt_state, x, y):
    params = get_params(opt_state)
    grads = grad_mse_loss(params, x, y)
    return opt_update(i, grads, opt_state)

epochs = 1000
for epoch in range(epochs):
    opt_state = update(epoch, opt_state, X, y)

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        params = get_params(opt_state)
        loss = mse_loss(params, X, y)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Display the learned parameters
params = get_params(opt_state)
w, b = params
print(f"Learned weight: {w[0][0]:.4f}, Learned bias: {b[0]:.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = linear_regression(params, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
