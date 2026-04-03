import jax
import jax.numpy as jnp
from jax import random, grad, jit
from flax import linen as nn
from flax.training import train_state
import optax
import torchvision
import numpy as np

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# Convert to numpy, normalize to [-1, 1] (matching transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)))
train_images = np.array(train_dataset.data, dtype=np.float32) / 255.0
train_images = (train_images - 0.5) / 0.5  # (N, 32, 32, 3) HWC format
train_labels = np.array(train_dataset.targets, dtype=np.int32)

test_images = np.array(test_dataset.data, dtype=np.float32) / 255.0
test_images = (test_images - 0.5) / 0.5
test_labels = np.array(test_dataset.targets, dtype=np.int32)

def data_loader(images, labels, batch_size, shuffle=True):
    n = len(images)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield jnp.array(images[batch_idx]), jnp.array(labels[batch_idx])

# Define the CNN Model
# PyTorch architecture: conv1(3→32) → ReLU → conv2(32→64) → ReLU → MaxPool(2,2) → flatten(64*16*16) → fc1(128) → ReLU → fc2(10)
class CNNModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten: 64 * 16 * 16
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

# Initialize the model, loss function, and optimizer
model = CNNModel()
dummy_input = jnp.ones([1, 32, 32, 3])
variables = model.init(random.PRNGKey(0), dummy_input)

tx = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx
)

@jit
def train_step(state, images, labels):
    def loss_fn(params):
        logits = model.apply({'params': params}, images)
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in data_loader(train_images, train_labels, batch_size=64, shuffle=True):
        state, loss = train_step(state, images, labels)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluate on the test set
correct = 0
total = 0
for images, labels in data_loader(test_images, test_labels, batch_size=64, shuffle=False):
    logits = model.apply({'params': state.params}, images)
    predictions = jnp.argmax(logits, axis=-1)
    total += labels.shape[0]
    correct += int((predictions == labels).sum())

print(f"Test Accuracy: {100 * correct / total:.2f}%")
