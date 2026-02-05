# Explain a CNN model's predictions using Grad-CAM in JAX/Flax
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define a simple CNN model split into feature extractor and classifier.
# Flax does not have a built-in pretrained ResNet18, so we use a simple CNN.
# The model is split so we can compute Grad-CAM using JAX's functional gradients
# (JAX/Flax does not support PyTorch-style hooks like register_backward_hook).

class FeatureExtractor(nn.Module):
    """CNN layers up to and including the target layer for Grad-CAM."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(128, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(256, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        # Target layer for Grad-CAM (analogous to model.layer4[1].conv2 in ResNet)
        x = nn.Conv(512, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        return x

class Classifier(nn.Module):
    """Classification head after the target layer."""
    num_classes: int = 1000
    @nn.compact
    def __call__(self, x):
        x = jnp.mean(x, axis=(1, 2))  # Global average pooling (NHWC: mean over H, W)
        x = nn.Dense(self.num_classes)(x)
        return x

class CNN(nn.Module):
    """Full CNN model combining feature extractor and classifier."""
    num_classes: int = 1000
    @nn.compact
    def __call__(self, x):
        features = FeatureExtractor()(x)
        logits = Classifier(num_classes=self.num_classes)(features)
        return logits

# Initialize model
key = jax.random.PRNGKey(0)
model = CNN(num_classes=1000)
dummy_input = jnp.ones((1, 224, 224, 3))
key, init_key = jax.random.split(key)
variables = model.init(init_key, dummy_input)
params = variables['params']

# Split params for feature extractor and classifier
feature_params = params['FeatureExtractor_0']
classifier_params = params['Classifier_0']

# Create module instances for separate apply calls
feature_extractor = FeatureExtractor()
classifier = Classifier(num_classes=1000)

# Generate a random sample image (equivalent to torchvision.datasets.FakeData)
key, img_key = jax.random.split(key)
raw_image = jax.random.uniform(img_key, (224, 224, 3), minval=0.0, maxval=1.0)
image = Image.fromarray((np.array(raw_image) * 255).astype(np.uint8))

# Preprocess the image for the model (ImageNet normalization, same as PyTorch original)
mean = jnp.array([0.485, 0.456, 0.406])
std = jnp.array([0.229, 0.224, 0.225])
input_tensor = (raw_image - mean) / std
input_tensor = input_tensor[None, ...]  # Add batch dimension: (1, 224, 224, 3)

# Perform a forward pass
output = model.apply({'params': params}, input_tensor)
predicted_class = jnp.argmax(output, axis=-1).item()

# Grad-CAM: Compute gradients of target class score w.r.t. target layer activations.
# This replaces PyTorch's register_backward_hook/register_forward_hook with
# JAX's functional gradient computation (jax.grad).
def compute_grad_cam(feature_params, classifier_params, input_tensor, target_class):
    # Get activations from the target layer (feature extractor output)
    activations = feature_extractor.apply({'params': feature_params}, input_tensor)

    # Compute gradient of target class score w.r.t. activations
    def class_score_fn(acts):
        logits = classifier.apply({'params': classifier_params}, acts)
        return logits[0, target_class]

    gradients = jax.grad(class_score_fn)(activations)
    return activations, gradients

activations, gradients = compute_grad_cam(feature_params, classifier_params, input_tensor, predicted_class)

# Generate Grad-CAM heatmap
# JAX uses NHWC format: activations shape is (B, H, W, C)
weights = jnp.mean(gradients, axis=(1, 2), keepdims=True)  # Mean over H, W -> (B, 1, 1, C)
heatmap = (weights * activations).sum(axis=-1).squeeze()    # Sum over C -> (H, W)
heatmap = jnp.maximum(heatmap, 0)  # ReLU

# Normalize the heatmap and overlay it on the original image
heatmap = heatmap / (heatmap.max() + 1e-8)
heatmap_resized = jax.image.resize(heatmap, (224, 224), method='bilinear')
heatmap_pil = Image.fromarray((np.array(heatmap_resized) * 255).astype(np.uint8))

# Display the image with the Grad-CAM heatmap
plt.imshow(image)
plt.imshow(heatmap_pil, alpha=0.5, cmap='jet')
plt.title(f"Predicted Class: {predicted_class}")
plt.axis('off')
plt.show()
