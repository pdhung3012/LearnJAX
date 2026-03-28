import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Define the CNN Model
class CNNModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv2d(3, 32, kernel_size=3, padding=1)(x))
        x = nn.max_pool(nn.relu(nn.Conv2d(32, 64, kernel_size=3, padding=1)(x)), 2)
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(128)(x))
        x = nn.Dense(10)(x)
        return x

# Initialize the model, loss function, and optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optax.adam(learning_rate=0.001)

# Generate random data for training and testing
rng = jax.random.PRNGKey(0)
train_images = jax.random.normal(rng, (100, 3, 32, 32))
train_labels = jax.random.randint(rng, (100,), 0, 10)
test_images = jax.random.normal(rng, (20, 3, 32, 32))
test_labels = jax.random.randint(rng, (20,), 0, 10)

# Training loop
for epoch in range(10):
    # Forward pass
    outputs = model(train_images)
    loss = criterion(outputs, train_labels)

    # Backward pass and optimization
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 3, 32, 32)))
    updates, _ = optax.value_and_grad(loss)(params)
    params = optax.apply_updates(params, updates)

    print(f"Epoch [{epoch + 1}/{10}], Loss: {loss.item():.4f}")

# Evaluate on the test set
outputs = model(test_images)
_, predicted = jnp.argmax(outputs, axis=-1)
accuracy = jnp.mean(predicted == test_labels)

print(f"Test Accuracy: {accuracy * 100:.2f}%")