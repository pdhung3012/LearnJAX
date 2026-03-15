import jax
import jax.numpy as jnp
from jax import grad, jit, random
from flax import linen as nn
from flax.training import train_state
import optax
import pandas as pd
import numpy as np

# Generate synthetic data (matching PyTorch)
def generate_and_save_data():
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    X = random.uniform(subkey, shape=(100, 1)) * 10
    key, subkey = random.split(key)
    noise = random.normal(subkey, shape=(100, 1))
    y = 2 * X + 3 + noise
    
    # Save to CSV
    data = jnp.concatenate([X, y], axis=1)
    df = pd.DataFrame(np.array(data), columns=['X', 'y'])
    df.to_csv('data.csv', index=False)
    print("Data generated and saved to data.csv")

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    X = jnp.array(df['X'].values, dtype=jnp.float32).reshape(-1, 1)
    y = jnp.array(df['y'].values, dtype=jnp.float32).reshape(-1, 1)
    return X, y

def data_loader(X, y, batch_size, shuffle=True, seed=None):
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield {'x': X[batch_idx], 'y': y[batch_idx]}

class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1)(x)
        return x

def create_train_state(rng, learning_rate):
    model = SimpleNN()
    params = model.init(rng, jnp.ones([1, 1]))  # Initialize with dummy input
    tx = optax.sgd(learning_rate)  # Changed from adam to sgd
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jit
def train_step(state, batch):
    def loss_fn(params):
        predictions = state.apply_fn(params, batch['x'])
        return jnp.mean((predictions - batch['y']) ** 2)

    grads = grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    loss = loss_fn(state.params)
    return new_state, loss

def main():
    # Generate data first
    generate_and_save_data()
    
    rng = random.PRNGKey(42)  # Changed from 0 to 42
    learning_rate = 0.01  # Changed from 0.001 to 0.01
    state = create_train_state(rng, learning_rate)
    
    X, y = load_data('data.csv')
    batch_size = 32
    epochs = 1000

    for epoch in range(epochs):
        # Seed shuffle for reproducibility
        for batch in data_loader(X, y, batch_size, shuffle=True, seed=42):
            state, loss = train_step(state, batch)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

    # Output learned parameters
    w = state.params['params']['Dense_0']['kernel'].flatten()[0]
    b = state.params['params']['Dense_0']['bias'].flatten()[0]
    print(f"Learned weight: {w:.4f}, Learned bias: {b:.4f}")

    # Testing on new data
    X_test = jnp.array([[4.0], [7.0]])
    predictions = state.apply_fn(state.params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == "__main__":
    main()