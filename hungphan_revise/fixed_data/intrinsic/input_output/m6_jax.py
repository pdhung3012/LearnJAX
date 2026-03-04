import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# -----------------------------
# Augmentation Functions in JAX
# -----------------------------
def random_flip(image, key):
    """Randomly flip the image horizontally with probability 0.5."""
    do_flip = jax.random.uniform(key) > 0.5
    return jnp.where(do_flip, jnp.flip(image, axis=1), image)

def random_crop(image, key, crop_size=32, padding=4):
    """Pad the image then randomly crop a patch of size (crop_size x crop_size)."""
    # Pad with 4 pixels on each side using constant (mimics PyTorch padding behavior)
    image_padded = jnp.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)
    # The padded image has size 32+8=40; choose a random crop location in the padded image.
    max_offset = padding * 2  # 8
    offset_x = jax.random.randint(key, (), 0, max_offset + 1)
    offset_y = jax.random.randint(key, (), 0, max_offset + 1)
    cropped = image_padded[offset_x:offset_x+crop_size, offset_y:offset_y+crop_size, :]
    return cropped

def normalize(image):
    """Normalize image with mean=0.5 and std=0.5 (per channel)."""
    return (image - 0.5) / 0.5

def augment_image(image, key):
    """Apply the series of augmentations to a single image."""
    key_flip, key_crop = jax.random.split(key)
    image = random_flip(image, key_flip)
    image = random_crop(image, key_crop, crop_size=32, padding=4)
    image = normalize(image)
    return image

# -----------------------------
# Utility Function for Plotting
# -----------------------------
def imshow_grid(images):
    """Display a grid of images. Assumes images shape is (batch, H, W, C)."""
    # Unnormalize to bring pixel values back to [0, 1] for display
    images = images * 0.5 + 0.5
    batch, h, w, c = images.shape
    grid_cols = 8  # adjust as needed
    grid_rows = int(np.ceil(batch / grid_cols))
    grid = np.zeros((grid_rows * h, grid_cols * w, c))
    for idx, image in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        grid[row*h:(row+1)*h, col*w:(col+1)*w, :] = image
    plt.figure(figsize=(grid_cols, grid_rows))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

# -----------------------------
# Dataset Loading with tfds
# -----------------------------
def load_dataset(split, batch_size=64):
    """
    Load CIFAR-10 from TensorFlow Datasets.
    The images are scaled to [0,1] and returned along with their labels.
    """
    ds = tfds.load('cifar10', split=split, as_supervised=True)

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0  # scale image to [0, 1]
        return image, label

    ds = ds.map(preprocess)
    ds = ds.cache()
    if split == 'train':
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return tfds.as_numpy(ds)

# Load training and test datasets
train_ds = load_dataset('train', batch_size=64)
test_ds = load_dataset('test', batch_size=64)

# -----------------------------
# Process One Batch of Training Images
# -----------------------------
# Get one batch of training images and labels
batch = next(iter(train_ds))
images, labels = batch  # images shape: (64, 32, 32, 3)

# Apply augmentation to each image in the batch using separate random keys.
augmented_images = []
rng = jax.random.PRNGKey(0)
for i, image in enumerate(images):
    rng, subkey = jax.random.split(rng)  # Generate fresh key each time
    aug_img = augment_image(jnp.array(image), subkey)
    augmented_images.append(np.array(aug_img))
augmented_images = np.stack(augmented_images, axis=0)

# -----------------------------
# Display the Augmented Batch
# -----------------------------
imshow_grid(augmented_images)
