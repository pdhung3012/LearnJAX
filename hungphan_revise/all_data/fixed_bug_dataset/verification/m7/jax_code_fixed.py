"""
M7 MNIST Simple NN - JAX/Flax Implementation

Fixed to match PyTorch exactly:
- Uses streaming data pipeline (tf.data) instead of loading all data into RAM
- Shuffles data every epoch (not just once)
- Removed commented dead code
- Proper memory-efficient approach that scales to large datasets

This follows Google's recommended JAX pattern:
- TensorFlow (tf.data, tfds) for efficient data streaming
- JAX/Flax for model computation and training
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import optax
from flax import linen as nn
from functools import partial

# ---------------------------
# Data Loading and Preprocessing (STREAMING)
# ---------------------------
def get_datasets_streaming(batch_size=64):
    """
    Load MNIST using tf.data pipeline for efficient streaming.
    This matches PyTorch's DataLoader behavior - loads batch-by-batch,
    not all data at once.
    """
    # Load datasets
    train_ds = tfds.load('mnist', split='train', shuffle_files=True)
    test_ds = tfds.load('mnist', split='test', shuffle_files=False)
    
    def preprocess(example):
        """Convert image to float32, scale to [0,1] then normalize to [-1,1]"""
        image = tf.cast(example['image'], tf.float32) / 255.0
        image = (image - 0.5) / 0.5
        label = example['label']
        return image, label
    
    # FIXED: Shuffle every epoch, batch, and prefetch - streaming pipeline
    # reshuffle_each_iteration=True matches PyTorch's shuffle=True behavior
    train_ds = train_ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    # Test dataset: no shuffling
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds

# ---------------------------
# Model Definition using Flax Linen
# ---------------------------
class SimpleNN(nn.Module):
    """
    Simple neural network matching PyTorch architecture:
    - Input: 28x28 pixels (784 features)
    - Hidden: 128 neurons with ReLU
    - Output: 10 classes
    """
    @nn.compact
    def __call__(self, x):
        # Flatten the input (28x28 pixels becomes 784)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

def create_train_state(rng, learning_rate):
    """Initialize model parameters and optimizer"""
    model = SimpleNN()
    # Initialize parameters with dummy input: shape (1, 28, 28, 1)
    dummy_input = jnp.ones((1, 28, 28, 1))
    params = model.init(rng, dummy_input)
    # Use SGD optimizer with learning rate 0.01 to match PyTorch
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(params)
    return model, params, optimizer, opt_state

# ---------------------------
# Loss Function and Training Step
# ---------------------------
def loss_fn(params, model, batch):
    """Compute cross-entropy loss"""
    images, labels = batch
    logits = model.apply(params, images)
    one_hot = jax.nn.one_hot(labels, num_classes=10)
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    return loss

# FIXED: Removed commented @jax.jit line - keep only the correct decorator
@partial(jax.jit, static_argnames=('model', 'optimizer'))
def train_step(params, opt_state, batch, model, optimizer):
    """Single training step with gradient update"""
    loss, grads = jax.value_and_grad(loss_fn)(params, model, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ---------------------------
# Main Training and Evaluation Loop
# ---------------------------
def main():
    batch_size = 64
    epochs = 5
    
    # FIXED: Load datasets with streaming pipeline
    train_ds, test_ds = get_datasets_streaming(batch_size)
    
    # Initialize model and optimizer
    rng = jax.random.PRNGKey(0)
    model, params, optimizer, opt_state = create_train_state(rng, learning_rate=0.01)
    
    # Training Loop with Benchmarking
    for epoch in range(epochs):
        start_time = time.time()
        
        # FIXED: Iterate over streaming dataset (not pre-loaded batches)
        # Each epoch gets freshly shuffled data from tf.data pipeline
        for batch in tfds.as_numpy(train_ds):
            params, opt_state, loss = train_step(params, opt_state, batch, model, optimizer)
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {float(loss):.4f}, Time: {training_time:.4f}s")
    
    # Evaluation on Test Set with Benchmarking
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch in tfds.as_numpy(test_ds):
        images, labels = batch
        logits = model.apply(params, images)
        predictions = jnp.argmax(logits, axis=1)
        correct += int(jnp.sum(predictions == labels))
        total += images.shape[0]
    
    end_time = time.time()
    testing_time = end_time - start_time
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%, Testing Time: {testing_time:.4f}s")

if __name__ == "__main__":
    main()