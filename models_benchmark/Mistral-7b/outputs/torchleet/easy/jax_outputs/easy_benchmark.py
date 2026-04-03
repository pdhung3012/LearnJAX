import jax
import jax.numpy as jnp
from jax import grad, jit
from flax import linen as nn
from flax.training import train_state
import optax
import torchvision
import numpy as np
import time

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

# Convert to numpy, normalize to [-1, 1] (matching transforms.Normalize((0.5,), (0.5,)))
train_images = np.array(train_dataset.data.numpy(), dtype=np.float32) / 255.0
train_images = (train_images - 0.5) / 0.5
train_images = train_images.reshape(-1, 28 * 28)  # Flatten to match nn.Linear(28*28, 128)
train_labels = np.array(train_dataset.targets.numpy(), dtype=np.int32)

test_images = np.array(test_dataset.data.numpy(), dtype=np.float32) / 255.0
test_images = (test_images - 0.5) / 0.5
test_images = test_images.reshape(-1, 28 * 28)
test_labels = np.array(test_dataset.targets.numpy(), dtype=np.int32)

def data_loader(images, labels, batch_size, shuffle=True):
    n = len(images)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield jnp.array(images[batch_idx]), jnp.array(labels[batch_idx])

# Define a simple neural network model
class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN()
dummy_input = jnp.ones([1, 28 * 28])
variables = model.init(jax.random.PRNGKey(0), dummy_input)

optimizer = optax.sgd(learning_rate=0.01)
state = train_state.TrainState.create(
    apply_fn=model.apply, params=variables['params'], tx=optimizer
)

@jit
def train_step(state, images, labels):
    def loss_fn(params):
        logits = model.apply({'params': params}, images)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Training loop with benchmarking
epochs = 5
for epoch in range(epochs):
    start_time = time.time()
    for images, labels in data_loader(train_images, train_labels, batch_size=64, shuffle=True):
        state, loss = train_step(state, images, labels)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Time: {training_time:.4f}s")

# Evaluate the model on the test set and benchmark the accuracy
correct = 0
total = 0
start_time = time.time()
for images, labels in data_loader(test_images, test_labels, batch_size=64, shuffle=False):
    logits = model.apply({'params': state.params}, images)
    predictions = jnp.argmax(logits, axis=-1)
    total += labels.shape[0]
    correct += int((predictions == labels).sum())
end_time = time.time()
testing_time = end_time - start_time
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%, Testing Time: {testing_time:.4f}s")
