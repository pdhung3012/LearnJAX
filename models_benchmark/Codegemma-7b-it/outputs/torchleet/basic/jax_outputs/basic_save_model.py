import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit
from jax.example_libraries import optimizers
import pickle

# Define a simple model
def simple_model(params, x):
    w, b = params
    return jnp.dot(x, w) + b

# Create and train the model
key = jax.random.PRNGKey(42)
key, key_x, key_noise = jax.random.split(key, 3)

# Training data
X = jax.random.uniform(key_x, shape=(100, 1))
y = 3 * X + 2 + jax.random.normal(key_noise, shape=(100, 1)) * 0.1

# Initialize parameters (matching nn.Linear(1, 1))
params = (jnp.zeros((1, 1)), jnp.zeros((1,)))

# Loss function
def mse_loss(params, x, y):
    predictions = simple_model(params, x)
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
    opt_state = update(epoch, opt_state, X, y)

# Save the model params to a file named "model.pkl"
params = get_params(opt_state)
serializable_params = jax.tree_util.tree_map(lambda x: np.array(x), params)
with open('model.pkl', 'wb') as f:
    pickle.dump(serializable_params, f)

# Load the model params back from "model.pkl"
with open('model.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
loaded_params = jax.tree_util.tree_map(lambda x: jnp.array(x), loaded_data)

# Verify the model works after loading
X_test = jnp.array([[0.5], [1.0], [1.5]])
predictions = simple_model(loaded_params, X_test)
print(f"Predictions after loading: {predictions}")
