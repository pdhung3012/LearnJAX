#Strong LLM
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


# Set the random seed and create a PRNG key
key = jax.random.PRNGKey(42)

# Generate synthetic data: 100 data points between 0 and 10, with noise
key, subkey = jax.random.split(key)
X = jax.random.uniform(subkey, shape=(100, 1)) * 10
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape=(100, 1))
y = 2 * X + 3 + noise  # y = 2*x + 3 + noise

# Save the generated data to 'data.csv'
data = jnp.concatenate([X, y], axis=1)
df = pd.DataFrame(np.array(data), columns=['X', 'y'])
df.to_csv('data.csv', index=False)

# Define a simple Dataset class that loads data from CSV
class LinearRegressionDataset:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # Convert the pandas Series to JAX arrays (with float32 precision)
        self.X = jnp.array(self.data['X'].values, dtype=jnp.float32).reshape(-1, 1)
        self.y = jnp.array(self.data['y'].values, dtype=jnp.float32).reshape(-1, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define a DataLoader function to yield batches
def data_loader(dataset, batch_size, shuffle=True):
    n = len(dataset)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, n, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield dataset.X[batch_indices], dataset.y[batch_indices]

# Create dataset and specify batch size
dataset = LinearRegressionDataset('data.csv')
batch_size = 32

# Initialize model parameters similar to nn.Linear(1, 1)
# Here we initialize the weight (shape (1,1)) and bias (shape (1,))
bound = 1.0  # Using a uniform distribution bound similar to PyTorch initialization
key, subkey = jax.random.split(key)
w = jax.random.uniform(subkey, shape=(1, 1), minval=-bound, maxval=bound)
key, subkey = jax.random.split(key)
b = jax.random.uniform(subkey, shape=(1,), minval=-bound, maxval=bound)
params = {"w": w, "b": b}

# Define the forward (prediction) function
def predict(params, x):
    return jnp.dot(x, params["w"]) + params["b"]

# Define the loss function (Mean Squared Error)
def loss_fn(params, x, y):
    preds = predict(params, x)
    return jnp.mean((preds - y) ** 2)

# Set hyperparameters
lr = 0.01
epochs = 1000

# Get function to compute loss and its gradients
loss_and_grad = jax.value_and_grad(loss_fn)

# Training loop over epochs and batches
for epoch in range(epochs):
    # Loop over batches using our custom data_loader
    for batch_X, batch_y in data_loader(dataset, batch_size, shuffle=True):
        loss, grads = loss_and_grad(params, batch_X, batch_y)
        # Update parameters using SGD
        params["w"] = params["w"] - lr * grads["w"]
        params["b"] = params["b"] - lr * grads["b"]
    
    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")


# Extract and print learned weight and bias
learned_w = params["w"][0, 0]
learned_b = params["b"][0]
print(f"Learned weight: {learned_w:.4f}, Learned bias: {learned_b:.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = predict(params, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
