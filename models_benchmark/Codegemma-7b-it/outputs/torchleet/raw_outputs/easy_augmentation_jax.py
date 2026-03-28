import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Define the model architecture
class ResNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv2d(32, (3, 3), padding='same')(x)
        x = nn.relu(x)
        x = nn.MaxPool2d((2, 2), padding='valid')(x)

        x = nn.Conv2d(32, (3, 3), padding='same')(x)
        x = nn.relu(x)
        x = nn.MaxPool2d((2, 2), padding='valid')(x)

        x = nn.Conv2d(64, (3, 3), padding='same')(x)
        x = nn.relu(x)
        x = nn.MaxPool2d((2, 2), padding='valid')(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(10)(x)
        return x

# Initialize the model
model = ResNet()

# Define the loss function
def loss_fn(params, batch):
    images, labels = batch
    logits = model.apply({'params': params}, images)
    return jnp.mean(optax.softmax_cross_entropy_with_logits(logits, labels))

# Define the optimizer
optimizer = optax.sgd(learning_rate=0.01)

# Training loop
@jax.jit
def train_step(params, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, new_params = optimizer.update(grads, params)
    params = optax.apply_updates(params, updates)
    return loss, new_params

# Generate random data for training
rng = jax.random.PRNGKey(0)
images = jax.random.normal(rng, (100, 3, 32, 32))
labels = jax.random.randint(rng, (100,), 0, 10)

# Train the model
for step in range(10):
    loss, params = train_step(params, (images, labels))
    print(f"Step {step}: Loss = {loss}")