import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import flax.linen as nn
from flax.training import train_state
import optax

# Constants
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 32
NUM_CLASSES = 10
INPUT_SHAPE = (28, 28, 1)

# Define model (VanillaCNNModel is assumed to be defined elsewhere)
class VanillaCNNModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Define the forward pass here
        x = nn.Conv(features=32, kernel_size=(3,3), padding='SAME')(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=NUM_CLASSES)(x)
        return x

def create_train_state(rng, model, learning_rate):
    # Initialize the model parameters
    params = model.init(rng, jnp.ones((1, *INPUT_SHAPE)))  # MODIFIED: Input shape for initialization
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jit
def loss_fn(params, x, y):
    # Compute the loss function
    model = VanillaCNNModel()
    logits = model.apply(params, x)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=y))
    return loss

@jit
def compute_gradients(params, x, y):
    # Compute gradients
    return grad(loss_fn)(params, x, y)

def update(params, grads):
    # Update parameters
    return optax.apply_updates(params, grads)  # MODIFIED: Use functional update

def train_model(x_train, y_train, num_epochs, batch_size):
    rng = random.PRNGKey(0)  # PRNG key for reproducibility
    model = VanillaCNNModel()
    state = create_train_state(rng, model, learning_rate=LEARNING_RATE)

    for epoch in range(num_epochs):
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            grads = compute_gradients(state.params, x_batch, y_batch)
            state = state.apply_gradients(grads=grads)  # MODIFIED: Use functional updates to apply gradients

    return state.params  # Return final weights

def main():
    # Sample training data (x_train, y_train should be defined appropriately)
    x_train = jnp.ones((100, *INPUT_SHAPE))  # Placeholder, replace with actual data
    y_train = jax.nn.one_hot(jnp.zeros(100), num_classes=NUM_CLASSES)  # Placeholder, replace with actual labels
    model = VanillaCNNModel()
    final_weights = train_model(x_train, y_train, NUM_EPOCHS, BATCH_SIZE)
    print('Final weights:', final_weights)  # Display final weights after training

if __name__ == "__main__":
    main()