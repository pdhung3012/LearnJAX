import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds  # Required for loading datasets
import flax.linen as nn
from flax.training import train_state
import matplotlib.pyplot as plt
import numpy as np

def load_cifar10(batch_size=64):
    ds = tfds.load('cifar10', split='train', as_supervised=True)

    def preprocess(image, label):
      """Added code to normalize the image output. Please note that this is not a specific error. This is added into the fixed code to make the output comprehensive"""
      image = tf.image.random_flip_left_right(image)
      # Pad image with 4 pixels on each side using reflection padding
      image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode='REFLECT')
      # Random crop a 32x32 image
      image = tf.image.random_crop(image, size=[32, 32, 3])
      # Convert image to float32 and scale to [0, 1]
      image = tf.cast(image, tf.float32) / 255.0
      # Normalize with mean=0.5 and std=0.5 to get values in roughly [-1, 1]
      image = (image - 0.5) / 0.5
      return image, label
      """End of added code"""
      # # Use TensorFlow's image resize function
      # image = tf.image.resize(image, (32, 32))
      # image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
      # return image, label

    ds = ds.map(preprocess)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)  # Improve performance with prefetching

    return tfds.as_numpy(ds)  # // MODIFIED: Convert the dataset to NumPy arrays


def imshow_grid(images):
    # Simple function to display a grid of images
    grid = np.concatenate([np.concatenate([np.array(img) for img in images[i:i+8]], axis=1) for i in range(0, len(images), 8)], axis=0)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


def main():
    try:
        batch_size = 64  # Example batch size, adjust as necessary
        cifar10_data = load_cifar10(batch_size)

        # Example of iterating through the dataset and displaying images
        for images, labels in cifar10_data:
            # Show images - this is where you would handle displaying or further processing
            print(images.shape, labels.shape)  # // MODIFIED: Print shapes to show output
            imshow_grid(images)
            break  # Remove break to process all batches

    except Exception as e:
        print("An error occurred:", e)


if __name__ == '__main__':
    main()  # Entry point to the program