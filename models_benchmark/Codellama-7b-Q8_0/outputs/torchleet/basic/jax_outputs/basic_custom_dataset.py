import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import grad, jit
from jax.example_libraries import optimizers

# Generate data
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
X = jax.random.uniform(key1, shape=(100, 1)) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + jax.random.normal(key2, shape=(100, 1))  # Linear relationship with noise

# Save the generated data to data.csv
data = jnp.concatenate((X, y), axis=1)
df = pd.DataFrame(np.array(data), columns=['X', 'y'])
df.to_csv('data.csv', index=False)

# Custom Dataset class (JAX equivalent of PyTorch Dataset)
class LinearRegressionDataset:
    def __init__(self, csv_file):
        # Load data from CSV file
        self.data = pd.read_csv(csv_file)
        self.X = jnp.array(self.data['X'].values, dtype=jnp.float32).reshape(-1, 1)
        self.y = jnp.array(self.data['y'].values, dtype=jnp.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DataLoader equivalent for JAX
def data_loader(dataset, batch_size, shuffle=True, key=None):
    n = len(dataset)
    indices = jnp.arange(n)
    if shuffle and key is not None:
        indices = jax.random.permutation(key, indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_X = dataset.X[batch_idx]
        batch_y = dataset.y[batch_idx]
        yield batch_X, batch_y

# Example usage of the DataLoader
dataset = LinearRegressionDataset('data.csv')

# Define the Linear Regression Model (functional approach)
def init_params():
    return (jnp.zeros((1, 1)), jnp.zeros((1, 1)))  # (w, b)

def predict(params, X):
    w, b = params
    return X @ w + b

def loss_fn(params, X, y):
    predictions = predict(params, X)
    return jnp.mean((predictions - y) ** 2)

grad_loss_fn = grad(loss_fn, argnums=0)

# Initialize the model and optimizer
params = init_params()
opt_init, opt_update, get_params = optimizers.sgd(step_size=0.01)
opt_state = opt_init(params)

@jit
def update(i, opt_state, batch_X, batch_y):
    params = get_params(opt_state)
    grads = grad_loss_fn(params, batch_X, batch_y)
    return opt_update(i, grads, opt_state)

# Training loop
epochs = 1000
batch_size = 32
step_counter = 0

for epoch in range(epochs):
    # Create a new key for shuffling each epoch
    epoch_key = jax.random.PRNGKey(epoch)

    for batch_X, batch_y in data_loader(dataset, batch_size, shuffle=True, key=epoch_key):
        opt_state = update(step_counter, opt_state, batch_X, batch_y)
        step_counter += 1

    # Log progress every 100 epochs (report last batch loss, matching PyTorch behavior)
    if (epoch + 1) % 100 == 0:
        params = get_params(opt_state)
        loss = loss_fn(params, batch_X, batch_y)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Display the learned parameters
params = get_params(opt_state)
w, b = params
print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = predict(params, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
