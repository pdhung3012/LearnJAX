import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers

# Generate synthetic data
key = jax.random.PRNGKey(42)
key, key_x, key_noise, key_w1, key_w2 = jax.random.split(key, 5)
X = jax.random.uniform(key_x, shape=(100, 2)) * 10  # 100 data points with 2 features
y = (X[:, 0] + X[:, 1] * 2).reshape(-1, 1) + jax.random.normal(key_noise, shape=(100, 1))  # Non-linear relationship with noise

# Define the Deep Neural Network Model
def dnn_model(params, x):
    w1, b1, w2, b2 = params
    h = jnp.maximum(jnp.dot(x, w1) + b1, 0)
    return jnp.dot(h, w2) + b2

# Initialize parameters
params = (jax.random.normal(key_w1, shape=(2, 10)), jnp.zeros((10,)),
          jax.random.normal(key_w2, shape=(10, 1)), jnp.zeros((1,)))

# Define the loss function
def loss_fn(params, x, y):
    predictions = dnn_model(params, x)
    return jnp.mean((predictions - y) ** 2)

# Define the gradient of the loss function
grad_loss_fn = grad(loss_fn)

# Training loop
opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)
opt_state = opt_init(params)

@jit
def update_step(i, opt_state, x, y):
    params = get_params(opt_state)
    grads = grad_loss_fn(params, x, y)
    return opt_update(i, grads, opt_state)

epochs = 1000
for epoch in range(epochs):
    # Backward pass and optimization
    opt_state = update_step(epoch, opt_state, X, y)

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        params = get_params(opt_state)
        loss = loss_fn(params, X, y)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Testing on new data
X_test = jnp.array([[4.0, 3.0], [7.0, 8.0]])
predictions = dnn_model(get_params(opt_state), X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
