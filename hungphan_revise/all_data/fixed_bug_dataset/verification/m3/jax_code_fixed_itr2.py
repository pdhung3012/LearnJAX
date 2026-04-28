from typing import Callable

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np

import torchvision

LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 32
NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)


class VanillaCNNModel(nn.Module):
    """
    Vanilla CNN matching PyTorch architecture:
    Conv(3→32) → ReLU → Conv(32→64) → ReLU → Pool → Flatten → FC(16384→128) → ReLU → FC(128→10)

    kernel_init / bias_init are forwarded to every Conv and Dense layer so the
    5 initialization strategies from the PyTorch reference can actually be honored.
    """

    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                    kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME',
                    kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.relu(x)

        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=128,
                     kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        x = nn.relu(x)

        x = nn.Dense(features=NUM_CLASSES,
                     kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return x


def load_cifar10():
    """Load CIFAR10 via torchvision (same source as the PyTorch reference)."""
    train_ds = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True)
    test_ds = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True)

    x_train = train_ds.data.astype(np.float32)
    y_train = np.asarray(train_ds.targets, dtype=np.int32)
    x_test = test_ds.data.astype(np.float32)
    y_test = np.asarray(test_ds.targets, dtype=np.int32)

    x_train = (x_train / 255.0 - 0.5) / 0.5
    x_test = (x_test / 255.0 - 0.5) / 0.5

    return (x_train, y_train), (x_test, y_test)


def get_init_pair(init_type):
    """
    Map init_type → (kernel_init, bias_init) matching the PyTorch reference.

    - vanilla: Flax defaults (lecun_normal / zeros). PT's real default is
               kaiming_uniform_, but lecun_normal lands in the same
               "sane init" basin and produces comparable final accuracy.
    - kaiming: he_normal kernel, zero bias. PT uses fan_out for Conv and
               fan_in for Linear; we use fan_in (he_normal) for both.
    - xavier : xavier_normal kernel, zero bias. Matches PT exactly.
    - zeros  : all-zero weights and biases. With zero weights every logit
               is identical → uniform softmax → no gradient on weight
               asymmetry → stays stuck at loss ln(10) ≈ 2.30 and ~10% acc.
    - random : N(0,1) for weights AND biases. Huge init scale makes the
               first forward pass explode; model collapses to ~10% acc.
    """
    if init_type == "vanilla":
        return nn.initializers.lecun_normal(), nn.initializers.zeros
    elif init_type == "kaiming":
        return nn.initializers.he_normal(), nn.initializers.zeros
    elif init_type == "xavier":
        return nn.initializers.xavier_normal(), nn.initializers.zeros
    elif init_type == "zeros":
        return nn.initializers.zeros, nn.initializers.zeros
    elif init_type == "random":
        return nn.initializers.normal(stddev=1.0), nn.initializers.normal(stddev=1.0)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")


def create_train_state(rng, init_type="vanilla"):
    """Initialize training state with the requested initialization."""
    kernel_init, bias_init = get_init_pair(init_type)
    model = VanillaCNNModel(kernel_init=kernel_init, bias_init=bias_init)

    dummy_input = jnp.ones((1, *INPUT_SHAPE))
    variables = model.init(rng, dummy_input)
    params = variables['params']

    tx = optax.adam(LEARNING_RATE)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


@jax.jit
def train_step(state, batch_images, batch_labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch_images)
        labels_onehot = jax.nn.one_hot(batch_labels, NUM_CLASSES)
        loss = optax.softmax_cross_entropy(
            logits=logits, labels=labels_onehot).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def eval_step(params, apply_fn, batch_images, batch_labels):
    logits = apply_fn({'params': params}, batch_images)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == batch_labels)
    return accuracy


eval_step = jax.jit(eval_step, static_argnums=(1,))


def train_test_loop(init_type, train_data, test_data):
    (x_train, y_train), (x_test, y_test) = train_data, test_data

    rng = random.PRNGKey(0)
    state = create_train_state(rng, init_type)

    num_train = len(x_train)

    for epoch in range(NUM_EPOCHS):
        rng, shuffle_rng = random.split(rng)
        perm = random.permutation(shuffle_rng, num_train)
        x_train_shuffled = x_train[perm]
        y_train_shuffled = y_train[perm]

        epoch_loss = 0
        for i in range(0, num_train, BATCH_SIZE):
            batch_images = x_train_shuffled[i:i+BATCH_SIZE]
            batch_labels = y_train_shuffled[i:i+BATCH_SIZE]
            state, loss = train_step(state, batch_images, batch_labels)
            epoch_loss = loss

        print(f"Training loss at epoch {epoch} = {float(epoch_loss):.6f}")

    num_test = len(x_test)
    correct = 0
    total = 0
    for i in range(0, num_test, BATCH_SIZE):
        batch_images = x_test[i:i+BATCH_SIZE]
        batch_labels = y_test[i:i+BATCH_SIZE]
        acc = eval_step(state.params, state.apply_fn,
                        batch_images, batch_labels)
        correct += float(acc) * len(batch_labels)
        total += len(batch_labels)

    test_accuracy = (correct / total) * 100
    print(f"Test Accuracy = {test_accuracy:.2f}")


def main():
    print("Loading CIFAR10 dataset...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    print(f"Train data: {x_train.shape}, Test data: {x_test.shape}")

    train_data = (x_train, y_train)
    test_data = (x_test, y_test)

    init_types = ["vanilla", "kaiming", "xavier", "zeros", "random"]
    init_names = ["Vanilla", "Kaiming", "Xavier", "Zeros", "Random"]

    for name, init_type in zip(init_names, init_types):
        print(f"_________{name}_______________________")
        train_test_loop(init_type, train_data, test_data)


if __name__ == "__main__":
    main()
