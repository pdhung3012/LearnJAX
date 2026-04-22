import jax
import jax.numpy as jnp  # MODIFIED: Consistently import jax.numpy as jnp
from jax import grad, jit, random, vmap
import flax.linen as nn
import optax
import pickle

class SimpleModel(nn.Module):
    """A simple neural network model using Flax."""
    
    def setup(self):
        """Define the layers of the model."""
        self.dense = nn.Dense(features=1)  # A layer with one output feature

    def __call__(self, x):
        """Forward pass of the model."""
        return self.dense(x)

def train_model(X, y, key):
    """Train the model with the given data."""
    model = SimpleModel()
    params = model.init(key, X)
    # Loss function and optimization setup
    loss_fn = lambda params: jnp.mean((model.apply(params, X) - y) ** 2)
    optimizer = optax.sgd(0.01)
    opt_state = optimizer.init(params)
    
    for epoch in range(100):  # Simple training loop
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    
    return params

def main():
    """Main function to execute the training and evaluation of the model."""
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    X_train = random.uniform(subkey, (100, 1))
    key, subkey = random.split(key)
    noise = random.normal(subkey, (100, 1)) * 0.1
    y_train = 3 * X_train + 2 + noise

    # Train the model
    trained_params = train_model(X_train, y_train, key)
    
    # Save model parameters to file
    with open("model.pkl", "wb") as f:
        pickle.dump(trained_params, f)

    # Load model parameters from file
    with open("model.pkl", "rb") as f:
        loaded_params = pickle.load(f)

    # Verify the model works after loading
    X_test = jnp.array([[0.5], [1.0], [1.5]])  # Test data
    model = SimpleModel()  # Initialize model
    predictions = model.apply(loaded_params, X_test)  # Get predictions
    print(f"Predictions after training: {predictions}")

if __name__ == "__main__":
    main()  # Execute the main function