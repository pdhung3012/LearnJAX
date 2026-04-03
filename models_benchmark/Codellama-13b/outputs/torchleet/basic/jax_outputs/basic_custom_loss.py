import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers

# Generate synthetic data
key = jax.random.PRNGKey(42)
key, key_x, key_noise = jax.random.split(key, 3)
X = jax.random.uniform(key_x, shape=(100, 1)) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + jax.random.normal(key_noise, shape=(100, 1))  # Linear relationship with noise

def linear_regression_model(params, x):
    w, b = params
    return jnp.dot(x, w) + b

def huber_loss(params, X, y, delta=1.0):
    predictions = linear_regression_model(params, X)
    error = jnp.abs(predictions - y)
    loss = jnp.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    return jnp.mean(loss)

# Initialize parameters (weight and bias, matching nn.Linear(1, 1))
params = (jnp.zeros((1, 1)), jnp.zeros((1,)))
opt_init, opt_update, get_params = optimizers.sgd(step_size=0.01)

@jit
def update(i, opt_state, X, y):
    params = get_params(opt_state)
    grads = grad(huber_loss)(params, X, y)
    return opt_update(i, grads, opt_state)

opt_state = opt_init(params)

# Training loop
epochs = 1000
for epoch in range(epochs):
    opt_state = update(epoch, opt_state, X, y)

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        params = get_params(opt_state)
        loss = huber_loss(params, X, y)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Display the learned parameters
params = get_params(opt_state)
w, b = params
print(f"Learned weight: {w[0][0]:.4f}, Learned bias: {b[0]:.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = linear_regression_model(params, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
