import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define the ResNet model using flax.linen
class ResNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv2d(3, 64, (7, 7), (2, 2), (3, 3))(x)
        x = nn.max_pool(x, (3, 3), (2, 2), (1, 1))
        x = nn.relu(x)

        for i in range(4):
            for j in range(2):
                x = self.residual_block(x, 64, 1 if i == 0 and j == 0 else 2)

        x = nn.avg_pool(x, (2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(1000)(x)
        return x

    def residual_block(self, x, channels, stride):
        residual = x

        x = nn.Conv2d(channels, channels, (3, 3), (stride, stride), (1, 1))(x)
        x = nn.BatchNorm2d(channels)(x)
        x = nn.relu(x)

        x = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))(x)
        x = nn.BatchNorm2d(channels)(x)

        if residual.shape != x.shape:
            residual = nn.Conv2d(residual.shape[-1], channels, (1, 1), (stride, stride), (0, 0))(residual)
            residual = nn.BatchNorm2d(channels)(residual)

        x = x + residual
        x = nn.relu(x)
        return x

# Create a ResNet instance
model = ResNet()

# Define variables to capture gradients and activations
gradients = None
activations = None

# Define hooks to capture gradients and activations
def save_gradients(grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def save_activations(input, output):
    global activations
    activations = output

# Attach hooks to the target layer
target_layer = model.residual_block.conv2
target_layer.grad_fn.register_hook(save_gradients)
target_layer.register_forward_hook(save_activations)

# Fetch a sample image from numpy
image = np.random.rand(224, 224, 3)
image = Image.fromarray((image * 255).astype(np.uint8))

# Preprocess the image for the model
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0)

# Perform a forward pass
output = model(input_tensor)
predicted_class = output.argmax(dim=1).item()

# Perform a backward pass for the predicted class
model.zero_grad()
output[0, predicted_class].backward()

# Generate Grad-CAM heatmap
weights = gradients.mean(dim=[2, 3], keepdim=True)
heatmap = (weights * activations).sum(dim=1).squeeze().relu()

# Normalize the heatmap and overlay it on the original image
heatmap = heatmap / heatmap.max()
heatmap = transforms.ToPILImage()(heatmap.cpu())
heatmap = heatmap.resize(image.size, resample=Image.BILINEAR)

# Display the image with the Grad-CAM heatmap
plt.imshow(image)
plt.imshow(heatmap, alpha=0.5, cmap='jet')
plt.title(f"Predicted Class: {predicted_class}")
plt.axis('off')
plt.show()