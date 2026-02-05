import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import optax
import numpy as np
import torchvision


def load_data():
    """Load CIFAR-10 with torchvision and apply PyTorch-equivalent normalization."""
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


def data_loader(images, labels, batch_size=32, shuffle=True, key=None):
    """Numpy/JAX array loader mirroring torch DataLoader batching/shuffling."""
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


class VanillaCNNModel(nn.Module):
    """conv1 -> relu -> conv2 -> relu -> pool -> flatten -> fc1 -> relu -> fc2."""

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME", name="conv1")(x))
        x = nn.relu(nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", name="conv2")(x))
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(features=128, name="fc1")(x))
        x = nn.Dense(features=10, name="fc2")(x)
        return x


def _compute_fan_in(kernel_shape):
    if len(kernel_shape) < 2:
        return int(kernel_shape[0]) if kernel_shape else 1
    receptive_field = int(np.prod(kernel_shape[:-2])) if len(kernel_shape) > 2 else 1
    return int(kernel_shape[-2] * receptive_field)


def reinit_params(params, init_type, key):
    """Reinitialize Conv/Dense weights and biases to mirror PyTorch init configs."""
    params_mut = unfreeze(params)
    new_params = {}

    for layer_name, layer_params in params_mut.items():
        new_layer = {}
        kernel = layer_params.get("kernel", None)
        kernel_shape = tuple(kernel.shape) if kernel is not None else None
        fan_in = _compute_fan_in(kernel_shape) if kernel_shape is not None else 1
        for param_name, param in layer_params.items():
            key, subkey = jax.random.split(key)
            if init_type == "vanilla":
                if param_name in ("kernel", "bias"):
                    bound = 1.0 / np.sqrt(fan_in)
                    new_layer[param_name] = jax.random.uniform(
                        subkey,
                        param.shape,
                        minval=-bound,
                        maxval=bound,
                        dtype=param.dtype,
                    )
                else:
                    new_layer[param_name] = param
            elif init_type == "kaiming":
                if param_name == "kernel":
                    mode = "fan_out" if layer_name in ("conv1", "conv2") else "fan_in"
                    new_layer[param_name] = nn.initializers.variance_scaling(2.0, mode, "normal")(
                        subkey,
                        param.shape,
                    )
                else:
                    new_layer[param_name] = jnp.zeros_like(param)
            elif init_type == "xavier":
                if param_name == "kernel":
                    new_layer[param_name] = nn.initializers.variance_scaling(1.0, "fan_avg", "normal")(
                        subkey,
                        param.shape,
                    )
                else:
                    new_layer[param_name] = jnp.zeros_like(param)
            elif init_type == "zeros":
                new_layer[param_name] = jnp.zeros_like(param)
            elif init_type == "random":
                new_layer[param_name] = jax.random.normal(subkey, param.shape)
            else:
                raise ValueError(f"Unknown init_type: {init_type}")
        new_params[layer_name] = new_layer

    return freeze(new_params)


def train_test_loop(model, params, train_images, train_labels, test_images, test_labels, epochs=10):
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)
    batch_size = 32

    @jax.jit
    def train_step(curr_params, curr_opt_state, images, labels):
        def loss_fn(p):
            logits = model.apply({"params": p}, images)
            return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels))

        loss, grads = jax.value_and_grad(loss_fn)(curr_params)
        updates, next_opt_state = optimizer.update(grads, curr_opt_state)
        next_params = optax.apply_updates(curr_params, updates)
        return next_params, next_opt_state, loss

    train_key = jax.random.PRNGKey(0)
    for epoch in range(epochs):
        train_key, epoch_key = jax.random.split(train_key)
        for image, label in data_loader(train_images, train_labels, batch_size=batch_size, shuffle=True, key=epoch_key):
            params, opt_state, loss = train_step(params, opt_state, image, label)
        print(f"Training loss at epoch {epoch} = {float(loss)}")

    correct = 0
    total = 0
    test_key = jax.random.PRNGKey(123)
    for image_test, label_test in data_loader(
        test_images,
        test_labels,
        batch_size=batch_size,
        shuffle=True,  # PyTorch source uses shuffle=True for test_loader as well.
        key=test_key,
    ):
        pred_test = model.apply({"params": params}, image_test)
        pred_test_vals = jnp.argmax(pred_test, axis=1)
        total += label_test.shape[0]
        correct += int((pred_test_vals == label_test).sum())
    print(f"Test Accuracy = {(correct * 100) / total}")


model = VanillaCNNModel()
key = jax.random.PRNGKey(0)
key, init_key = jax.random.split(key)
dummy_input = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
variables = model.init(init_key, dummy_input)
default_params = variables["params"]

key, k0, k1, k2, k3, k4 = jax.random.split(key, 6)
init_configs = {
    "Vanilla": reinit_params(default_params, "vanilla", k0),
    "Kaiming": reinit_params(default_params, "kaiming", k1),
    "Xavier": reinit_params(default_params, "xavier", k2),
    "Zeros": reinit_params(default_params, "zeros", k3),
    "Random": reinit_params(default_params, "random", k4),
}

(train_images, train_labels), (test_images, test_labels) = load_data()

for name, params in init_configs.items():
    print(f"_________{name}_______________________")
    train_test_loop(model, params, train_images, train_labels, test_images, test_labels)
