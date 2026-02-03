import jax
import jax.numpy as jnp
from jax import random
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-10 dataset (raw, no transforms)
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

# Convert entire dataset to numpy arrays (HWC uint8 images)
train_images = np.array(train_dataset.data)  # (50000, 32, 32, 3)
train_labels = np.array(train_dataset.targets)
test_images = np.array(test_dataset.data)
test_labels = np.array(test_dataset.targets)

# Data augmentation functions in JAX
def random_horizontal_flip(key, image):
    """Randomly flip the image horizontally with 50% probability."""
    do_flip = random.uniform(key) > 0.5
    return jnp.where(do_flip, jnp.flip(image, axis=1), image)

def random_crop_with_padding(key, image, padding=4):
    """Randomly crop the image after padding (equivalent to transforms.RandomCrop(32, padding=4))."""
    # Pad the image
    padded = jnp.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    # Random crop offset
    h_offset = random.randint(key, (), 0, 2 * padding + 1)
    w_key = random.fold_in(key, 1)
    w_offset = random.randint(w_key, (), 0, 2 * padding + 1)
    # Use dynamic_slice for JIT compatibility
    return jax.lax.dynamic_slice(padded, (h_offset, w_offset, 0), (32, 32, 3))

def to_float_and_normalize(image):
    """Convert to float [0,1] then normalize with mean=0.5, std=0.5 per channel."""
    image = image.astype(jnp.float32) / 255.0
    return (image - 0.5) / 0.5

def augment_and_normalize(key, image):
    """Apply augmentation pipeline: flip, crop, normalize."""
    k1, k2 = random.split(key)
    image = random_horizontal_flip(k1, image)
    image = random_crop_with_padding(k2, image, padding=4)
    image = to_float_and_normalize(image)
    return image

def normalize_only(image):
    """Normalize without augmentation (for test set)."""
    return to_float_and_normalize(image)

# Create a simple data loader
def data_loader(images, labels, batch_size, key, shuffle=True, augment=True):
    n = len(images)
    indices = np.arange(n)
    if shuffle:
        indices = np.array(random.permutation(key, n))
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_images = jnp.array(images[batch_idx])
        batch_labels = jnp.array(labels[batch_idx])
        if augment:
            keys = random.split(random.fold_in(key, start), len(batch_idx))
            batch_images = jax.vmap(augment_and_normalize)(keys, batch_images)
        else:
            batch_images = jax.vmap(normalize_only)(batch_images)
        yield batch_images, batch_labels

# Display a batch of augmented images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = np.array(img)
    plt.imshow(npimg)
    plt.show()

def make_grid(images, nrow=8):
    """Create a grid of images similar to torchvision.utils.make_grid.
    images: (N, H, W, C) array
    """
    n, h, w, c = images.shape
    ncol = nrow
    nrow_actual = int(np.ceil(n / ncol))
    # Pad with zeros if needed
    pad_n = nrow_actual * ncol - n
    if pad_n > 0:
        padding = jnp.zeros((pad_n, h, w, c))
        images = jnp.concatenate([images, padding], axis=0)
    grid = images.reshape(nrow_actual, ncol, h, w, c)
    grid = grid.transpose(0, 2, 1, 3, 4)  # (nrow, H, ncol, W, C)
    grid = grid.reshape(nrow_actual * h, ncol * w, c)
    return grid

# Get some random training images
key = random.PRNGKey(0)
for batch_images, batch_labels in data_loader(train_images, train_labels, 64, key, shuffle=True, augment=True):
    grid = make_grid(batch_images)
    imshow(grid)
    break
