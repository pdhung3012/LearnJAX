import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Define a simple neural network model using flax.linen
class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((-1, 28*28))  # Flatten the input
        x = nn.relu(nn.Dense(128)(x))
        x = nn.Dense(10)(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
loss_fn = jax.nn.softmax_cross_entropy_with_logits
optimizer = optax.sgd(learning_rate=0.01)

# Generate random training and test data
rng = jax.random.PRNGKey(0)
train_images = jax.random.normal(rng, (1000, 28, 28))
train_labels = jax.random.randint(rng, (1000,), 0, 9)
test_images = jax.random.normal(rng, (200, 28, 28))
test_labels = jax.random.randint(rng, (200,), 0, 9)

# Training loop with benchmarking
for epoch in range(5):
    start_time = time.time()  # Start time for training
    for images, labels in zip(train_images, train_labels):
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        params = model.init(jax.random.PRNGKey(0), jnp.ones_like(images))
        updates, _ = optimizer.update(loss, params)
        params = optax.apply_updates(params, updates)

    end_time = time.time()  # End time for training
    training_time = end_time - start_time
    print(f"Epoch [{epoch + 1}/5], Loss: {loss:.4f}, Time: {training_time:.4f}s")

# Evaluate the model on the test set and benchmark the accuracy
start_time = time.time()  # Start time for testing
params = model.init(jax.random.PRNGKey(0), jnp.ones_like(test_images))
outputs = model.apply(params, test_images)
predictions = jnp.argmax(outputs, axis=-1)
accuracy = jnp.mean(predictions == test_labels)

end_time = time.time()  # End time for testing
testing_time = end_time - start_time
print(f"Test Accuracy: {accuracy:.2f}, Testing Time: {testing_time:.4f}s")