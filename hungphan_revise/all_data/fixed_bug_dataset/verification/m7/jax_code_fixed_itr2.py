import time
from functools import partial

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds


BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.01
MNIST_TRAIN_SIZE = 60_000


def preprocess(image, label):
    """Match PyTorch ToTensor() + Normalize((0.5,), (0.5,))."""
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - 0.5) / 0.5
    return image, label


def get_datasets_streaming(batch_size=BATCH_SIZE):
    train_ds = tfds.load("mnist", split="train",
                         as_supervised=True, shuffle_files=True)
    test_ds = tfds.load("mnist", split="test",
                        as_supervised=True, shuffle_files=False)

    train_ds = train_ds.shuffle(
        buffer_size=MNIST_TRAIN_SIZE, reshuffle_each_iteration=True
    )
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds


class TorchLinear(nn.Module):
    """Dense layer with parameter init matching PyTorch nn.Linear."""

    features: int

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        bound = 1.0 / np.sqrt(in_features)

        def init_uniform(key, shape, dtype=jnp.float32):
            return jax.random.uniform(
                key, shape, dtype=dtype, minval=-bound, maxval=bound
            )

        kernel = self.param("kernel", init_uniform,
                            (in_features, self.features))
        bias = self.param("bias", init_uniform, (self.features,))
        return jnp.matmul(x, kernel) + bias


class SimpleNN(nn.Module):
    """Match PyTorch: Linear(784,128) -> ReLU -> Linear(128,10)."""

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = TorchLinear(features=128, name="fc1")(x)
        x = nn.relu(x)
        x = TorchLinear(features=10, name="fc2")(x)
        return x


def create_train_state(rng):
    model = SimpleNN()
    dummy_input = jnp.ones((1, 28, 28, 1), dtype=jnp.float32)
    params = model.init(rng, dummy_input)
    optimizer = optax.sgd(LEARNING_RATE)
    opt_state = optimizer.init(params)
    return model, params, optimizer, opt_state


def loss_fn(params, model, batch):
    images, labels = batch
    logits = model.apply(params, images)
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


@partial(jax.jit, static_argnames=("model", "optimizer"))
def train_step(params, opt_state, batch, model, optimizer):
    loss, grads = jax.value_and_grad(loss_fn)(params, model, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def main():
    train_ds, test_ds = get_datasets_streaming(BATCH_SIZE)

    rng = jax.random.PRNGKey(0)
    model, params, optimizer, opt_state = create_train_state(rng)

    for epoch in range(EPOCHS):
        start_time = time.time()

        for batch in tfds.as_numpy(train_ds):
            images, labels = batch
            batch_jax = (jnp.asarray(images), jnp.asarray(labels))
            params, opt_state, loss = train_step(
                params, opt_state, batch_jax, model, optimizer
            )

        end_time = time.time()
        training_time = end_time - start_time
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {float(loss):.4f}, Time: {training_time:.4f}s"
        )

    correct = 0
    total = 0
    start_time = time.time()

    for batch in tfds.as_numpy(test_ds):
        images, labels = batch
        logits = model.apply(params, jnp.asarray(images))
        predictions = jnp.argmax(logits, axis=1)
        correct += int(jnp.sum(predictions == jnp.asarray(labels)))
        total += images.shape[0]

    end_time = time.time()
    testing_time = end_time - start_time
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%, Testing Time: {testing_time:.4f}s")


if __name__ == "__main__":
    main()
