"""
M8 MNIST Autoencoder - JAX/Flax Implementation

Fixed to match PyTorch exactly:
- Changed epochs from 5 to 10 (match PyTorch training duration)
- Added reshuffle_each_iteration=True for per-epoch shuffling
- Fixed preprocessing to return (image, label) tuple
- Removed redundant batch dimension check
- Verified ConvTranspose output sizes match 28x28

Note: Both PyTorch and JAX have a design issue where Sigmoid outputs [0,1]
but inputs are normalized to [-1,1]. For true equivalence, should either:
- Remove normalization (keep images in [0,1]), OR
- Use Tanh instead of Sigmoid (outputs [-1,1])
This fix maintains the original (flawed) design for exact comparison.
"""

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
def preprocess_fn(image, label):
    """
    Preprocess MNIST images to match PyTorch transform.
    
    FIXED: Returns (image, label) tuple to maintain supervised structure,
    even though autoencoder doesn't use labels.
    """
    # Convert image to float32 and scale to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    # Normalize to [-1, 1] as (x - 0.5) / 0.5
    image = (image - 0.5) / 0.5
    # Ensure the image has a channel dimension (28,28) -> (28,28,1)
    if image.shape.rank == 2:
        image = tf.expand_dims(image, -1)
    # FIXED: Return both image and label
    return image, label

# Load both train and test datasets
train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

# Apply preprocessing
train_ds = train_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

# FIXED: Add reshuffle_each_iteration=True to match PyTorch's shuffle=True
train_ds = train_ds.shuffle(10000, reshuffle_each_iteration=True).batch(64).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(64).prefetch(tf.data.AUTOTUNE)

# ---------------------------
# Define the Autoencoder Model using Flax
# ---------------------------
class Autoencoder(nn.Module):
    """
    Autoencoder matching PyTorch architecture.
    
    Encoder: Conv(32) -> ReLU -> MaxPool -> Conv(64) -> ReLU -> MaxPool
    Decoder: ConvTranspose(32) -> ReLU -> ConvTranspose(1) -> Sigmoid
    
    Input: (28, 28, 1) -> Encoded: (7, 7, 64) -> Output: (28, 28, 1)
    """
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
        # FIXED: Verified this produces 28x28 output to match PyTorch
        # From (7, 7, 64) -> (14, 14, 32) -> (28, 28, 1)
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
    """Mean Squared Error loss for autoencoder"""
    return jnp.mean((reconstructed - original) ** 2)

# Match PyTorch optimizer: Adam with lr=0.001
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# ---------------------------
# Training Step Function (using JIT)
# ---------------------------
@jax.jit
def update(params, opt_state, batch):
    """Single training step with gradient update"""
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
# FIXED: Changed from 5 to 10 to match PyTorch
epochs = 10

for epoch in range(epochs):
    # FIXED: Properly unpack (images, labels) from supervised dataset
    # Autoencoder only uses images, ignores labels
    for images, labels in tfds.as_numpy(train_ds):
        # FIXED: Removed redundant batch dimension check
        # images is already (batch, 28, 28, 1) from pipeline
        params, opt_state, loss = update(params, opt_state, images)
    
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {float(loss):.4f}")

# ---------------------------
# Optional: Verify output size matches input
# ---------------------------
# Test that decoder produces exact 28x28 output
if __name__ == "__main__":
    # Verify architecture produces correct output size
    test_input = jnp.ones((1, 28, 28, 1))
    test_output = model.apply(params, test_input)
    assert test_output.shape == (1, 28, 28, 1), f"Output shape mismatch: {test_output.shape}"
    print(f"\n✓ Architecture verified: Input {test_input.shape} -> Output {test_output.shape}")