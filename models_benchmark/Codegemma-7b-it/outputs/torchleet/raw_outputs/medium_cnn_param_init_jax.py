import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Define the model architecture using flax.linen.Module
class VanillaCNNModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv2d(3, 32, kernel_size=3, padding=1)(x))
        x = nn.max_pool(nn.relu(nn.Conv2d(32, 64, kernel_size=3, padding=1)(x)), 2)
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(128)(x))
        x = nn.Dense(10)(x)
        return x

# Initialize the model
model = VanillaCNNModel()

# Define the loss function and optimizer
loss_fn = nn.losses.cross_entropy_with_logits
optimizer = optax.adam(learning_rate=0.001)

# Generate random data for training and testing
rng = jax.random.PRNGKey(0)
train_images = jax.random.normal(rng, (100, 3, 32, 32))
train_labels = jax.random.randint(rng, (100,), 0, 10)
test_images = jax.random.normal(rng, (20, 3, 32, 32))
test_labels = jax.random.randint(rng, (20,), 0, 10)

# Define the training step function
@jax.jit
def train_step(params, batch):
    images, labels = batch
    predictions = model.apply({"params": params}, images)
    loss = loss_fn(predictions, labels)
    grads = jax.grad(loss)(params)
    updates, new_params = optimizer.update(grads, params)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params

# Train the model
for epoch in range(10):
    for batch in zip(train_images, train_labels):
        loss, params = train_step(model.init(rng, jnp.ones((1, 3, 32, 32))), batch)
    print(f"Training loss at epoch {epoch} = {loss}")

# Evaluate the model
predictions = model.apply({"params": params}, test_images)
accuracy = jnp.mean(jnp.argmax(predictions, axis=-1) == test_labels)
print(f"Test Accuracy = {accuracy}")