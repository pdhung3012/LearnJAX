import math

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds


BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
MNIST_TRAIN_SIZE = 60_000
PYTORCH_CONVTRANSPOSE_PADDING = ((1, 2), (1, 2))


def pytorch_uniform_init(bound):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)

    return init


def pytorch_conv_bound(kernel_size, in_channels):
    fan_in = int(np.prod(kernel_size)) * int(in_channels)
    return 1.0 / math.sqrt(fan_in)


class TorchConv(nn.Module):
    features: int
    kernel_size: tuple[int, int]
    strides: tuple[int, int] = (1, 1)
    padding: str | tuple = "SAME"

    @nn.compact
    def __call__(self, x):
        in_channels = x.shape[-1]
        bound = pytorch_conv_bound(self.kernel_size, in_channels)
        return nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_init=pytorch_uniform_init(bound),
            bias_init=pytorch_uniform_init(bound),
        )(x)


class TorchConvTranspose(nn.Module):
    features: int
    kernel_size: tuple[int, int]
    strides: tuple[int, int] = (1, 1)
    padding: str | tuple = "SAME"

    @nn.compact
    def __call__(self, x):
        in_channels = x.shape[-1]
        bound = pytorch_conv_bound(self.kernel_size, in_channels)
        return nn.ConvTranspose(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_init=pytorch_uniform_init(bound),
            bias_init=pytorch_uniform_init(bound),
        )(x)


def preprocess_fn(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - 0.5) / 0.5
    return image, label


def get_datasets(batch_size=BATCH_SIZE):
    train_ds, test_ds = tfds.load(
        "mnist", split=["train", "test"], as_supervised=True)

    train_ds = train_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.shuffle(
        MNIST_TRAIN_SIZE, reshuffle_each_iteration=True
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds


class Autoencoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = TorchConv(32, kernel_size=(3, 3),
                      padding="SAME", name="enc_conv1")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2),
                        strides=(2, 2), padding="VALID")
        x = TorchConv(64, kernel_size=(3, 3),
                      padding="SAME", name="enc_conv2")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2),
                        strides=(2, 2), padding="VALID")

        x = TorchConvTranspose(
            32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=PYTORCH_CONVTRANSPOSE_PADDING,
            name="dec_deconv1",
        )(x)
        x = nn.relu(x)
        x = TorchConvTranspose(
            1,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=PYTORCH_CONVTRANSPOSE_PADDING,
            name="dec_deconv2",
        )(x)
        x = nn.sigmoid(x)
        return x


model = Autoencoder()
params = model.init(jax.random.PRNGKey(
    0), jnp.ones((1, 28, 28, 1), dtype=jnp.float32))


def mse_loss(reconstructed, original):
    return jnp.mean((reconstructed - original) ** 2)


optimizer = optax.adam(learning_rate=LEARNING_RATE)
opt_state = optimizer.init(params)


@jax.jit
def update(params, opt_state, batch):
    def loss_fn(current_params):
        reconstructed = model.apply(current_params, batch)
        return mse_loss(reconstructed, batch)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


train_ds, test_ds = get_datasets(BATCH_SIZE)

for epoch in range(EPOCHS):
    for images, _ in tfds.as_numpy(train_ds):
        batch = jnp.asarray(images)
        params, opt_state, loss = update(params, opt_state, batch)

    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {float(loss):.4f}")
