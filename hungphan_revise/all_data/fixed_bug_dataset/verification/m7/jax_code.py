"""
This is a version of JAX that provides more comprehensive output and
implement more utilities and complete translation that recreates the full
pipeline of the original PyTorch code (data loading, model definition,
training loop, and evaluation), but it does so using JAX's functional
style with Flax for the model, optax for optimization, and TFDS for data.

Error code will be recorded from this code version (if applicable)
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import optax
from flax import linen as nn
from functools import partial

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
def preprocess(example):
    # Convert image to float32, scale to [0,1] then normalize to [-1,1]
    image = np.array(example['image'], dtype=np.float32) / 255.0
    image = (image - 0.5) / 0.5
    # Ensure image shape is (28, 28, 1)
    if image.ndim == 2:
        image = np.expand_dims(image, -1)
    label = example['label']
    return image, label

def get_datasets(batch_size=64):
    # Load MNIST using TensorFlow Datasets
    train_ds = tfds.load('mnist', split='train', shuffle_files=True)
    test_ds  = tfds.load('mnist', split='test',  shuffle_files=False)

    # Convert training dataset to numpy arrays and create batches
    train_images, train_labels = [], []
    for example in tfds.as_numpy(train_ds):
        img, lab = preprocess(example)
        train_images.append(img)
        train_labels.append(lab)
    train_images = np.stack(train_images)
    train_labels = np.array(train_labels)

    # Convert test dataset to numpy arrays and create batches
    test_images, test_labels = [], []
    for example in tfds.as_numpy(test_ds):
        img, lab = preprocess(example)
        test_images.append(img)
        test_labels.append(lab)
    test_images = np.stack(test_images)
    test_labels = np.array(test_labels)

    # Create batches
    train_batches = [(train_images[i:i+batch_size], train_labels[i:i+batch_size])
                     for i in range(0, len(train_labels), batch_size)]
    test_batches = [(test_images[i:i+batch_size], test_labels[i:i+batch_size])
                    for i in range(0, len(test_labels), batch_size)]

    return train_batches, test_batches

# ---------------------------
# Model Definition using Flax Linen
# ---------------------------
class SimpleNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Flatten the input (28x28 pixels becomes 784)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

def create_train_state(rng, learning_rate):
    model = SimpleNN()
    # Initialize parameters with dummy input: shape (1, 28, 28, 1)
    dummy_input = jnp.ones((1, 28, 28, 1))
    params = model.init(rng, dummy_input)
    # Use SGD optimizer with learning rate 0.01
    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(params)
    return model, params, optimizer, opt_state

# ---------------------------
# Loss Function and Training Step
# ---------------------------
def loss_fn(params, model, batch):
    images, labels = batch
    logits = model.apply(params, images)
    one_hot = jax.nn.one_hot(labels, num_classes=10)
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    return loss

# @jax.jit
@partial(jax.jit, static_argnames=('model', 'optimizer'))
def train_step(params, opt_state, batch, model, optimizer):
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
    # Load datasets
    train_batches, test_batches = get_datasets(batch_size)

    # Initialize model and optimizer
    rng = jax.random.PRNGKey(0)
    model, params, optimizer, opt_state = create_train_state(rng, learning_rate=0.01)

    # Training Loop with Benchmarking
    for epoch in range(epochs):
        start_time = time.time()
        for batch in train_batches:
            params, opt_state, loss = train_step(params, opt_state, batch, model, optimizer)
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}, Time: {training_time:.4f}s")

    # Evaluation on Test Set with Benchmarking
    correct = 0
    total = 0
    start_time = time.time()
    for batch in test_batches:
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