import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
import torchvision


def load_data():
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

    train_images = np.asarray(train_dataset.data, dtype=np.float32) / 255.0
    train_images = (train_images - 0.5) / 0.5
    train_labels = np.asarray(train_dataset.targets, dtype=np.int32)

    test_images = np.asarray(test_dataset.data, dtype=np.float32) / 255.0
    test_images = (test_images - 0.5) / 0.5
    test_labels = np.asarray(test_dataset.targets, dtype=np.int32)

    return (jnp.asarray(train_images), jnp.asarray(train_labels)), (
        jnp.asarray(test_images),
        jnp.asarray(test_labels),
    )


def data_loader(images, labels, batch_size=64, shuffle=True, key=None):
    n = images.shape[0]
    if shuffle:
        if key is None:
            key = jax.random.PRNGKey(0)
        indices = np.asarray(jax.random.permutation(key, n))
    else:
        indices = np.arange(n)

    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        yield images[batch_idx], labels[batch_idx]


class Conv2dCustom(nn.Module):
    out_channels: int
    kernel_size: tuple
    stride: int = 1
    padding: int = 0

    @nn.compact
    def __call__(self, x):
        in_channels = x.shape[-1]
        kh, kw = self.kernel_size
        sh = sw = self.stride
        ph = pw = self.padding

        weight = self.param(
            "weight",
            lambda rng, shape: jax.random.normal(rng, shape) * 0.1,
            (kh, kw, in_channels, self.out_channels),
        )
        bias = self.param("bias", nn.initializers.zeros, (self.out_channels,))

        out = jax.lax.conv_general_dilated(
            x,
            weight,
            window_strides=(sh, sw),
            padding=[(ph, ph), (pw, pw)],
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        return out + bias


class MaxPool2dCustom(nn.Module):
    kernel_size: tuple
    stride: int = None

    @nn.compact
    def __call__(self, x):
        kh, kw = self.kernel_size
        stride = self.stride if self.stride is not None else kh
        return nn.max_pool(x, window_shape=(kh, kw), strides=(stride, stride), padding="VALID")


class CNNModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding="SAME", name="conv1")(x))
        x = nn.relu(nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", name="conv2")(x))
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(128, name="fc1")(x))
        x = nn.Dense(10, name="fc2")(x)
        return x


model = CNNModel()
key = jax.random.PRNGKey(0)
key, init_key = jax.random.split(key)
dummy_input = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
params = model.init(init_key, dummy_input)["params"]

optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)


@jax.jit
def train_step(curr_params, curr_opt_state, images, labels):
    def loss_fn(p):
        logits = model.apply({"params": p}, images)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels))

    loss, grads = jax.value_and_grad(loss_fn)(curr_params)
    updates, next_opt_state = optimizer.update(grads, curr_opt_state)
    next_params = optax.apply_updates(curr_params, updates)
    return next_params, next_opt_state, loss


@jax.jit
def eval_step(curr_params, images):
    logits = model.apply({"params": curr_params}, images)
    return jnp.argmax(logits, axis=-1)


(train_images, train_labels), (test_images, test_labels) = load_data()

epochs = 10
batch_size = 64
train_key = jax.random.PRNGKey(42)

for epoch in range(epochs):
    train_key, epoch_key = jax.random.split(train_key)
    for images, labels in data_loader(train_images, train_labels, batch_size=batch_size, shuffle=True, key=epoch_key):
        params, opt_state, loss = train_step(params, opt_state, images, labels)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {float(loss):.4f}")

correct = 0
total = 0
for images, labels in data_loader(test_images, test_labels, batch_size=batch_size, shuffle=False):
    predicted = eval_step(params, images)
    total += labels.shape[0]
    correct += int((predicted == labels).sum())

print(f"Test Accuracy: {100 * correct / total:.2f}%")
