#Strong LLM
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


# Set the random seed and create a PRNG key
key = jax.random.PRNGKey(42)

# Generate synthetic data: 100 data points between 0 and 10 with noise
key, subkey = jax.random.split(key)
X = jax.random.uniform(subkey, shape=(100, 1)) * 10
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape=(100, 1))
y = 2 * X + 3 + noise  # Linear relationship with noise


# Define the custom activation function: tanh(x) + x
def custom_activation(x):
    return jnp.tanh(x) + x

# Define the model: a linear transformation followed by custom activation
def model(params, x):
    linear_output = jnp.dot(x, params['w']) + params['b']
    return custom_activation(linear_output)

# Initialize model parameters similar to nn.Linear(1, 1)
# Here we use a uniform distribution in [-bound, bound]
bound = 1.0
key, subkey = jax.random.split(key)
w = jax.random.uniform(subkey, shape=(1, 1), minval=-bound, maxval=bound)
key, subkey = jax.random.split(key)
b = jax.random.uniform(subkey, shape=(1,), minval=-bound, maxval=bound)
params = {'w': w, 'b': b}


# Mean Squared Error loss
def loss_fn(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)

# Training hyperparameters
lr = 0.01
epochs = 1000

# Create a function to compute the loss and its gradients
loss_and_grad = jax.value_and_grad(loss_fn)


for epoch in range(epochs):
    loss, grads = loss_and_grad(params, X, y)
    # Update parameters using simple SGD
    params['w'] = params['w'] - lr * grads['w']
    params['b'] = params['b'] - lr * grads['b']
    
    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")


# Extract the learned linear parameters (without the custom activation)
learned_w = params['w'][0, 0]
learned_b = params['b'][0]
print(f"Learned weight: {learned_w:.4f}, Learned bias: {learned_b:.4f}")

# Plot the training data and the linear fit (as in the original PyTorch code)
plt.figure(figsize=(4, 4))
# Convert JAX arrays to NumPy arrays for plotting
X_np = np.array(X)
y_np = np.array(y)
plt.scatter(X_np, y_np, label='Training Data')

# Create a line using the learned linear parameters (ignoring custom activation)
X_line = np.linspace(0, 10, 100).reshape(-1, 1)
plt.plot(X_line, learned_w * X_line + learned_b, 'r', label='Model Fit')
plt.legend()
plt.show()


X_test = jnp.array([[4.0], [7.0]])
predictions = model(params, X_test)
print(f"Predictions for {np.array(X_test).tolist()}: {np.array(predictions).tolist()}")