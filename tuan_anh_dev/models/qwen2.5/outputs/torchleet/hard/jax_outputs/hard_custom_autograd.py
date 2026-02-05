import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax

# Generate synthetic data
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
X = jax.random.uniform(key1, shape=(100, 1)) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + jax.random.normal(key2, shape=(100, 1))  # Linear relationship with noise

@jax.custom_vjp
def learned_silu(x, slope):
    return slope * x * jax.nn.sigmoid(x)


def learned_silu_fwd(x, slope):
    sigmoid_x = jax.nn.sigmoid(x)
    output = slope * x * sigmoid_x
    return output, (x, slope, sigmoid_x)


def learned_silu_bwd(res, grad_output):
    x, slope, sigmoid_x = res
    grad_input = grad_output * slope * (sigmoid_x + x * sigmoid_x * (1.0 - sigmoid_x))

    # Match torch.autograd.Function backward for slope, then reduce to the broadcasted slope shape.
    grad_slope_full = grad_output * x * sigmoid_x
    reduce_axes = tuple(range(grad_slope_full.ndim - slope.ndim))
    grad_slope = jnp.sum(grad_slope_full, axis=reduce_axes) if reduce_axes else grad_slope_full
    grad_slope = grad_slope.reshape(slope.shape)
    return grad_input, grad_slope


learned_silu.defvjp(learned_silu_fwd, learned_silu_bwd)


def model(params, x):
    slope = params['slope']
    return learned_silu(x, slope)

def loss_fn(params, x, y):
    predictions = model(params, x)
    return jnp.mean((predictions - y) ** 2)

# Initialize parameters and optimizer
params = {'slope': jnp.array([1.0])}
optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(params)

@jit
def train_step(params, opt_state, x, y):
    loss, grads = value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

epochs = 1000
for epoch in range(epochs):
    params, opt_state, loss = train_step(params, opt_state, X, y)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Display the learned parameters
w = params['slope']
print(f"Learned weight: {w[0]:.4f}, Learned bias: 0.0000")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = model(params, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
