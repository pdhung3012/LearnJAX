import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.linen as nn
import numpy as np
import torchvision.transforms as transforms
from flax import serialization
import matplotlib.pyplot as plt
from PIL import Image

class SomeLayer(nn.Module):
    features: int
    key: jax.random.PRNGKey
    @nn.compact
    def __call__(self, x):
        # MODIFIED: pass key explicitly to prevent global state mutation
        # subkey, self.key = jax.random.split(self.key)  # Split the key for a new operation
        subkey = self.make_rng('rng')
        return jnn.relu(nn.Dense(self.features)(x))  # Use a dense layer with relu activation


def generate_random_tensor(shape, dtype=jnp.float32, key=None):  # MODIFIED: Explicit dtype and PRNGKey
    if key is None:
        raise ValueError("PRNG key must be provided")  # Error handling for missing key
    subkey, key = jax.random.split(key)  # Split key for randomness
    return jax.random.normal(subkey, shape, dtype=dtype)  # Generate a tensor with specified dtype


def main():
    key = jax.random.PRNGKey(0)  # Initialize a PRNGKey
    input_tensor_shape = (10, 10)  # Define the shape of the input tensor
    input_tensor = generate_random_tensor(input_tensor_shape, dtype=jnp.float32, key=key)  # MODIFIED: Use the modified function

    layer = SomeLayer(features=5, key=key)  # Pass the PRNGKey explicitly
    # output = layer(input_tensor)
    params = layer.init({"params": key, "rng": key}, input_tensor)
    output = layer.apply(params, input_tensor, rngs={"rng": key})

    # Assuming we have an image that we want to display and overlay with Grad-CAM heatmap
    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_image_data)
    # heatmap = transforms.Resize(image.size)(np.array(output))  # Convert JAX array to NumPy array
    output_np = np.array(output)  # Convert JAX array to NumPy
    output_img = Image.fromarray(output_np.astype(np.uint8))  # Convert to PIL Image
    heatmap = transforms.Resize(image.size)(output_img)
    # Display the image with the Grad-CAM heatmap
    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title("Predicted Class: Example Class")  # Example title
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()