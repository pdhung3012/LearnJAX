import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
def preprocess_fn(image, label):
    # Convert image to float32 and scale to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    # Normalize to [-1, 1] as (x - 0.5) / 0.5
    image = (image - 0.5) / 0.5
    # Ensure the image has a channel dimension (28,28) -> (28,28,1)
    if image.shape.rank == 2:
        image = tf.expand_dims(image, -1)
    return image

# Specify the 'split' to load both train and test datasets correctly.
train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=False)

# Apply preprocessing (ignoring the label similar to the PyTorch code)
train_ds = train_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and batch the training dataset (batch_size=64) and batch the test dataset.
train_ds = train_ds.shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(64).prefetch(tf.data.AUTOTUNE)

# ---------------------------
# Define the Autoencoder Model using Flax
# ---------------------------
class Autoencoder(nn.Module):
    def setup(self):
        # Encoder: Two Conv layers with ReLU and Max Pooling (downsampling)
        self.encoder = nn.Sequential([
            nn.Conv(32, kernel_size=(3, 3), padding='SAME'),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID'),
            nn.Conv(64, kernel_size=(3, 3), padding='SAME'),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        ])
        # Decoder: Two ConvTranspose layers with ReLU and final Sigmoid
        self.decoder = nn.Sequential([
            nn.ConvTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='SAME'),
            nn.relu,
            nn.ConvTranspose(1, kernel_size=(3, 3), strides=(2, 2), padding='SAME'),
            nn.sigmoid
        ])

    def __call__(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ---------------------------
# Initialize Model, Loss, and Optimizer
# ---------------------------
model = Autoencoder()
# Flax expects NHWC; create a dummy input of shape [1, 28, 28, 1]
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1)))

def mse_loss(reconstructed, original):
    return jnp.mean((reconstructed - original) ** 2)

optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# ---------------------------
# Training Step Function (using JIT)
# ---------------------------
@jax.jit
def update(params, opt_state, batch):
    def loss_fn(params):
        reconstructed = model.apply(params, batch)
        return mse_loss(reconstructed, batch)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

# ---------------------------
# Training Loop
# ---------------------------
epochs = 5
for epoch in range(epochs):
    for batch in tfds.as_numpy(train_ds):
        # Each batch is preprocessed to shape (batch, 28, 28, 1) already.
        if batch.ndim == 3:
            batch = np.expand_dims(batch, axis=-1)
        params, opt_state, loss = update(params, opt_state, batch)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")