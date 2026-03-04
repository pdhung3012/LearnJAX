# Strong LLM
import jax
import jax.numpy as jnp

# Set the random seed and create a PRNG key
key = jax.random.PRNGKey(42)

# Generate synthetic data
key, subkey = jax.random.split(key)
X = jax.random.uniform(subkey, shape=(100, 1)) * 10  # 100 data points between 0 and 10
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape=(100, 1))
y = 2 * X + 3 + noise  # Linear relationship with noise

# Initialize model parameters similar to PyTorch's nn.Linear(1, 1)
# PyTorch uses a uniform distribution in [-1/sqrt(in_features), 1/sqrt(in_features)]
bound = 1.0  # For in_features=1, bound = 1.
key, subkey = jax.random.split(key)
w = jax.random.uniform(subkey, shape=(1, 1), minval=-bound, maxval=bound)
key, subkey = jax.random.split(key)
b = jax.random.uniform(subkey, shape=(1,), minval=-bound, maxval=bound)
params = {"w": w, "b": b}

# Define the forward pass (prediction function)
def predict(params, x):
    # Using dot product to mimic nn.Linear (x is (batch, 1), w is (1, 1), b is (1,))
    return jnp.dot(x, params["w"]) + params["b"]

# Define the loss function (Mean Squared Error)
def loss_fn(params, x, y):
    preds = predict(params, x)
    return jnp.mean((preds - y) ** 2)

# Set learning rate and number of epochs
lr = 0.01
epochs = 1000

# Get a function to compute loss and its gradients
loss_and_grad = jax.value_and_grad(loss_fn)

# Training loop
for epoch in range(epochs):
    loss, grads = loss_and_grad(params, X, y)
    
    # Update parameters using SGD
    params["w"] = params["w"] - lr * grads["w"]
    params["b"] = params["b"] - lr * grads["b"]
    
    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Display the learned parameters
learned_w = params["w"][0, 0]
learned_b = params["b"][0]
print(f"Learned weight: {learned_w:.4f}, Learned bias: {learned_b:.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = predict(params, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
