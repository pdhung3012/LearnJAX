import jax
import jax.numpy as jnp
from jax import random, jit
from flax import linen as nn
from flax.training import train_state
import optax
import torchvision
import numpy as np

KERNEL_INIT = nn.initializers.normal(stddev=0.01)
BIAS_INIT = nn.initializers.zeros

# Load data
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# Normalize to [-1, 1] (matching transforms.Normalize((0.5,), (0.5,)))
train_images = np.array(train_dataset.data, dtype=np.float32) / 255.0
train_images = (train_images - 0.5) / 0.5  # (N, 32, 32, 3) HWC
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
        batch_imgs = jnp.array(images[batch_idx])
        # Resize from 32x32 to 224x224 (matching transforms.Resize(224))
        batch_imgs = jax.image.resize(batch_imgs, (batch_imgs.shape[0], 224, 224, 3), method='bilinear')
        yield batch_imgs, jnp.array(labels[batch_idx])

# Define AlexNet (NHWC format for JAX)
# PyTorch architecture: Conv2d(3,96,11,4,pad=2)->ReLU->MaxPool(3,2) -> Conv2d(96,256,5,pad=2)->ReLU->MaxPool(3,2)
# -> Conv2d(256,384,3,pad=1)->ReLU -> Conv2d(384,384,3,pad=1)->ReLU -> Conv2d(384,256,3,pad=1)->ReLU->MaxPool(3,2)
# -> Dropout->Dense(256*6*6,4096)->ReLU->Dropout->Dense(4096,4096)->ReLU->Dense(4096,10)
class AlexNet(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, training=False):
        # Features
        x = nn.Conv(
            features=96,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding=((2, 2), (2, 2)),
            kernel_init=KERNEL_INIT,
            bias_init=BIAS_INIT,
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        x = nn.Conv(
            features=256,
            kernel_size=(5, 5),
            padding=((2, 2), (2, 2)),
            kernel_init=KERNEL_INIT,
            bias_init=BIAS_INIT,
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        x = nn.Conv(
            features=384,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            kernel_init=KERNEL_INIT,
            bias_init=BIAS_INIT,
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=384,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            kernel_init=KERNEL_INIT,
            bias_init=BIAS_INIT,
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=256,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            kernel_init=KERNEL_INIT,
            bias_init=BIAS_INIT,
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        # Classifier
        x = x.reshape((x.shape[0], -1))  # Flatten: 256 * 6 * 6
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        x = nn.Dense(features=4096, kernel_init=KERNEL_INIT, bias_init=BIAS_INIT)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        x = nn.Dense(features=4096, kernel_init=KERNEL_INIT, bias_init=BIAS_INIT)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes, kernel_init=KERNEL_INIT, bias_init=BIAS_INIT)(x)
        return x

# Training setup
model = AlexNet(num_classes=10)
dummy_input = jnp.ones([1, 224, 224, 3])
variables = model.init(random.PRNGKey(0), dummy_input, training=False)

tx = optax.adam(learning_rate=0.0001)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx
)

@jit
def train_step(state, images, labels, dropout_rng):
    def loss_fn(params):
        logits = model.apply({'params': params}, images, training=True, rngs={'dropout': dropout_rng})
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean(), logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, logits

# Training loop
rng = random.PRNGKey(0)
num_epochs = 500
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader(train_images, train_labels, batch_size=64, shuffle=True):
        rng, dropout_rng = random.split(rng)
        state, loss, logits = train_step(state, images, labels, dropout_rng)

        running_loss += loss.item()
        predictions = jnp.argmax(logits, axis=-1)
        total += labels.shape[0]
        correct += int((predictions == labels).sum())

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

# Evaluation
correct = 0
total = 0
for images, labels in data_loader(test_images, test_labels, batch_size=64, shuffle=False):
    logits = model.apply({'params': state.params}, images, training=False)
    predictions = jnp.argmax(logits, axis=-1)
    total += labels.shape[0]
    correct += int((predictions == labels).sum())

print(f'Test Accuracy: {100 * correct / total:.2f}%')
