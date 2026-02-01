import jax
import jax.numpy as jnp
from jax import random, jit
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np

# Generate synthetic CT-scan data and associated segmentation masks
# Using reduced dimensions for memory (same architecture, smaller data)
key = random.PRNGKey(42)
batch = 4
num_slices = 4
channels = 3
width = 64
height = 64

# JAX uses NHWC: [B, D, H, W, C] instead of PyTorch's [B, D, C, H, W]
key1, key2 = random.split(key)
ct_images = random.normal(key1, (batch, num_slices, height, width, channels))
segmentation_masks = (random.normal(key2, (batch, num_slices, height, width, 1)) > 0).astype(jnp.float32)

print(f"CT images (train examples) shape: {ct_images.shape}")
print(f"Segmentation binary masks (labels) shape: {segmentation_masks.shape}")

# ResNet18 BasicBlock (without BatchNorm for simplicity - no pretrained weights available in JAX)
class BasicBlock(nn.Module):
    features: int
    strides: tuple = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.Conv(self.features, (3, 3), strides=self.strides, padding=((1, 1), (1, 1)), use_bias=False)(x)
        y = nn.relu(y)
        y = nn.Conv(self.features, (3, 3), padding=((1, 1), (1, 1)), use_bias=False)(y)

        if residual.shape != y.shape:
            residual = nn.Conv(self.features, (1, 1), strides=self.strides, use_bias=False)(residual)

        return nn.relu(y + residual)

# ResNet18 feature extractor (equivalent to torchvision.models.resnet18 minus avgpool+fc)
class ResNet18Backbone(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Initial conv: Conv(64, 7, stride=2, pad=3) + ReLU + MaxPool(3, stride=2, pad=1)
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding=((3, 3), (3, 3)), use_bias=False)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        # Layer 1: 2x BasicBlock(64)
        x = BasicBlock(64)(x)
        x = BasicBlock(64)(x)

        # Layer 2: 2x BasicBlock(128, stride=2)
        x = BasicBlock(128, strides=(2, 2))(x)
        x = BasicBlock(128)(x)

        # Layer 3: 2x BasicBlock(256, stride=2)
        x = BasicBlock(256, strides=(2, 2))(x)
        x = BasicBlock(256)(x)

        # Layer 4: 2x BasicBlock(512, stride=2)
        x = BasicBlock(512, strides=(2, 2))(x)
        x = BasicBlock(512)(x)

        return x

# Define the MedCNN class (matching PyTorch architecture)
# PyTorch: ResNet18 backbone -> 3DConv(512,64) -> 3DConv(64,64) -> ConvTranspose3D(64,32) -> ConvTranspose3D(32,16) -> Conv3D(16,1)
class MedCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        b = x.shape[0]
        d = x.shape[1]
        h, w, c = x.shape[2], x.shape[3], x.shape[4]
        print(f"Input shape [B, D, H, W, C]: {b, d, h, w, c}")

        # Reshape for 2D backbone: [B*D, H, W, C]
        x = x.reshape((b * d, h, w, c))
        features = ResNet18Backbone()(x)
        print(f"ResNet output shape [B*D, H, W, C]: {features.shape}")

        # Reshape back: [B, D, H', W', C']
        new_h, new_w, new_c = features.shape[1], features.shape[2], features.shape[3]
        x = features.reshape((b, d, new_h, new_w, new_c))
        print(f"Reshape ResNet output for 3DConv [B, D, H, W, C]: {x.shape}")

        # Downsampling with 3D Conv (in JAX NHWC: [B, D, H, W, C])
        x = nn.relu(nn.Conv(features=64, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)))(x))
        print(f"Output shape 3D Conv #1: {x.shape}")
        x = nn.relu(nn.Conv(features=64, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)))(x))
        print(f"Output shape 3D Conv #2: {x.shape}")

        # Upsampling with 3D Transposed Conv
        x = nn.relu(nn.ConvTranspose(features=32, kernel_size=(1, 4, 4), strides=(1, 4, 4), padding='SAME')(x))
        print(f"Output shape 3D Transposed Conv #1: {x.shape}")
        x = nn.relu(nn.ConvTranspose(features=16, kernel_size=(1, 8, 8), strides=(1, 8, 8), padding='SAME')(x))
        print(f"Output shape 3D Transposed Conv #2: {x.shape}")

        # Final segmentation
        x = jax.nn.sigmoid(nn.Conv(features=1, kernel_size=(1, 1, 1))(x))
        print(f"Final shape: {x.shape}")

        return x

def compute_dice_loss(pred, labels, eps=1e-8):
    """
    Args
    pred: [B, D, H, W, 1]
    labels: [B, D, H, W, 1]

    Returns
    dice_coefficient: scalar
    """
    numerator = 2 * jnp.sum(pred * labels)
    denominator = jnp.sum(pred) + jnp.sum(labels) + eps
    return numerator / denominator

# Initialize model and optimizer
model = MedCNN()
dummy_input = jnp.ones([1, num_slices, height, width, channels])
variables = model.init(random.PRNGKey(0), dummy_input)

optimizer = optax.adam(learning_rate=0.01)
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=optimizer
)

@jit
def train_step(state, ct_images, segmentation_masks):
    def loss_fn(params):
        predictions = model.apply({'params': params}, ct_images)
        return compute_dice_loss(predictions, segmentation_masks)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

epochs = 5
for epoch in range(epochs):
    state, loss = train_step(state, ct_images, segmentation_masks)
    print(f"Loss at epoch {epoch}: {loss}")
