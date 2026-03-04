#Strong LLM
import jax
import jax.numpy as jnp
import numpy as np


# Set the random seed and create a PRNG key
key = jax.random.PRNGKey(42)

# Generate synthetic data: 100 data points between 0 and 10 with noise
key, subkey = jax.random.split(key)
X = jax.random.uniform(subkey, shape=(100, 1)) * 10
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape=(100, 1))
y = 2 * X + 3 + noise  # y = 2*x + 3 + noise

# Define the linear regression model function
def predict(params, x):
    # Computes a linear transformation: x * w + b
    return jnp.dot(x, params["w"]) + params["b"]

# Define the Huber loss function
def huber_loss(params, x, y, delta=1.0):
    preds = predict(params, x)
    error = jnp.abs(preds - y)
    loss = jnp.where(error <= delta,
                     0.5 * error**2,            # L2 loss for small errors
                     delta * (error - 0.5 * delta))  # L1 loss for large errors
    return jnp.mean(loss)

# Initialize parameters for a linear layer with 1 input and 1 output.
bound = 1.0  # Using a simple uniform initialization bound
key, subkey = jax.random.split(key)
w = jax.random.uniform(subkey, shape=(1, 1), minval=-bound, maxval=bound)
key, subkey = jax.random.split(key)
b = jax.random.uniform(subkey, shape=(1,), minval=-bound, maxval=bound)
params = {"w": w, "b": b}

lr = 0.01
epochs = 1000

# Create a function that returns loss and gradients with respect to the parameters.
loss_and_grad = jax.value_and_grad(huber_loss, argnums=0)

for epoch in range(epochs):
    loss, grads = loss_and_grad(params, X, y, 1.0)
    # Parameter update using SGD
    params["w"] = params["w"] - lr * grads["w"]
    params["b"] = params["b"] - lr * grads["b"]
    
    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")


learned_w = params["w"][0, 0]
learned_b = params["b"][0]
print(f"Learned weight: {learned_w:.4f}, Learned bias: {learned_b:.4f}")


X_test = jnp.array([[4.0], [7.0]])
predictions = predict(params, X_test)
print(f"Predictions for {np.array(X_test).tolist()}: {np.array(predictions).tolist()}")
