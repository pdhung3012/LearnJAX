import jax
import jax.numpy as jnp
from jax import grad, jit, random, vmap
from flax import linen as nn
import optax
import numpy as np
from tensorboardX import SummaryWriter

# Linear regression model definition
class LinearRegressionModel(nn.Module):
    input_dim: int

    def setup(self):
        self.w = self.param('w', nn.initializers.xavier_uniform(), (self.input_dim, 1))
        self.b = self.param('b', nn.initializers.zeros, (1,))

    def __call__(self, x):
        return jnp.dot(x, self.w) + self.b

# Loss function
def loss_fn(params, inputs, targets, model):
    predictions = model.apply(params, inputs)
    return jnp.mean((predictions - targets) ** 2)

# Jitted gradient computation using vectorization
def compute_gradients(params, inputs, targets, model):
    return grad(loss_fn)(params, inputs, targets, model)
compute_gradients = jit(compute_gradients, static_argnums=(3,))

# Training function
def train_model(params, model, inputs, targets, num_epochs=100, learning_rate=0.01):
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(params)
    writer = SummaryWriter(log_dir="runs/linear_regression")

    for epoch in range(num_epochs):
        grads = compute_gradients(params, inputs, targets, model)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if (epoch + 1) % 10 == 0:
            current_loss = loss_fn(params, inputs, targets, model)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {current_loss:.4f}")
            writer.add_scalar("Loss/train", current_loss, epoch)
    
    writer.close()
    return params

def main():
    # Generate synthetic data
    key = jax.random.PRNGKey(42)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    inputs = jax.random.uniform(subkey1, (100, 1), minval=0.0, maxval=10.0)
    noise = jax.random.normal(subkey2, (100, 1))
    targets = 3 * inputs + 5 + noise

    # Initialize model
    model = LinearRegressionModel(input_dim=1)  # MODIFIED: Clearer initialization
    key = jax.random.PRNGKey(0)
    params = model.init(key, inputs)

    # Train the model
    trained_params = train_model(params, model, inputs, targets)
    final_predictions = model.apply(trained_params, inputs)

if __name__ == "__main__":
    main()  # Entry point of the program