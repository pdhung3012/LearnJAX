import jax
import jax.numpy as jnp
from jax import grad, jit
from jax.example_libraries import optimizers
import tensorflow as tf

# Generate synthetic data
key = jax.random.PRNGKey(42)
key, key_x, key_noise = jax.random.split(key, 3)
X = jax.random.uniform(key_x, shape=(100, 1)) * 10  # 100 data points between 0 and 10
y = 3 * X + 5 + jax.random.normal(key_noise, shape=(100, 1))  # Linear relationship with noise

# Define a simple Linear Regression Model
def linear_model(params, x):
    w, b = params
    return jnp.dot(x, w) + b

# Initialize TensorBoard writer
writer = tf.summary.create_file_writer('runs/linear_regression')

# Initialize the model parameters, loss function, and optimizer
params = (jnp.zeros((1, 1)), jnp.zeros((1,)))

def mse_loss(params, x, y):
    predictions = linear_model(params, x)
    return jnp.mean((predictions - y) ** 2)

# Training loop
opt_init, opt_update, get_params = optimizers.sgd(step_size=0.01)
opt_state = opt_init(params)

@jit
def update(i, opt_state, x, y):
    params = get_params(opt_state)
    grads = grad(mse_loss)(params, x, y)
    return opt_update(i, grads, opt_state)

epochs = 100
for epoch in range(epochs):
    # Forward pass (compute loss before update, matching PyTorch behavior)
    params = get_params(opt_state)
    loss = mse_loss(params, X, y)

    # Backward pass and optimization
    opt_state = update(epoch, opt_state, X, y)

    # Log loss to TensorBoard
    with writer.as_default():
        tf.summary.scalar("Loss/train", float(loss), step=epoch)

    # Log progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Close the TensorBoard writer
writer.close()

# Run TensorBoard using the logs generated
# Command to run: tensorboard --logdir=runs
