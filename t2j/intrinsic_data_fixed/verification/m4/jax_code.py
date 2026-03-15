import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
# Set random seed for reproducibility (equivalent to torch.manual_seed(42))
rng = jax.random.PRNGKey(42)

# Generate synthetic CT-scan data (batches, slices, channels, width, height)
batch = 5
num_slices = 10
channels = 3
width = 256
height = 256

def generate_synthetic_data(rng, batch, num_slices, channels, width, height):
    rng_data, rng_masks = jax.random.split(rng)
    ct_images = jax.random.normal(rng_data, (batch, num_slices, channels, width, height))
    segmentation_masks = (jax.random.normal(rng_masks, (batch, num_slices, 1, width, height)) > 0).astype(jnp.float32)
    return ct_images, segmentation_masks

ct_images, segmentation_masks = generate_synthetic_data(rng, batch, num_slices, channels, width, height)
print(f"CT images (train examples) shape: {ct_images.shape}")
print(f"Segmentation binary masks (labels) shape: {segmentation_masks.shape}")

# Define the MedCNN class in Flax
class MedCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        b, d, c, w, h = x.shape  # Input size: [B, D, C, W, H]
        print(f"Input shape [B, D, C, W, H]: {(b, d, c, w, h)}")
        x = x.reshape(b * d, c, w, h)  # [B*D, C, W, H]
        x = jnp.transpose(x, (0, 2, 3, 1))  # [B*D, W, H, C] = (1000, 256, 256, 3) for Flax NHWC
        x = nn.Conv(features=512, kernel_size=(3, 3), padding="SAME")(x)  # Simplified ResNet-like layer
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(32, 32), strides=(32, 32), padding="VALID")  # Downsample to 8x8
        x = jnp.transpose(x, (0, 3, 1, 2))  # [B*D, 512, 8, 8] to match PyTorch NCHW
        print(f"ResNet-like output shape [B*D, C, W, H]: {x.shape}")
        _, new_c, new_w, new_h = x.shape
        x = x.reshape(b, d, new_c, new_w, new_h)  # [B, D, C, W, H]
        x = jnp.transpose(x, (0, 2, 1, 3, 4))  # [B, C, D, W, H]
        print(f"Reshape ResNet output for 3DConv #1 [B, C, D, W, H]: {x.shape}")
        x = nn.Conv(features=64, kernel_size=(3, 3, 3), padding="SAME")(x)
        x = nn.relu(x)
        print(f"Output shape 3D Conv #1: {x.shape}")
        x = nn.Conv(features=64, kernel_size=(3, 3, 3), padding="SAME")(x)
        x = nn.relu(x)
        print(f"Output shape 3D Conv #2: {x.shape}")
        x = nn.ConvTranspose(features=32, kernel_size=(1, 4, 4), strides=(1, 4, 4), padding="VALID")(x)
        x = nn.relu(x)
        print(f"Output shape 3D Transposed Conv #1: {x.shape}")
        x = nn.ConvTranspose(features=16, kernel_size=(1, 8, 8), strides=(1, 8, 8), padding="VALID")(x)
        x = nn.relu(x)
        print(f"Output shape 3D Transposed Conv #2: {x.shape}")
        x = nn.Conv(features=1, kernel_size=(1, 1, 1))(x)
        x = jax.nn.sigmoid(x)
        print(f"Final shape: {x.shape}")
        return x

# Dice loss function
def compute_dice_loss(pred, labels, eps=1e-8):
    numerator = 2 * jnp.sum(pred * labels)
    denominator = jnp.sum(pred) + jnp.sum(labels) + eps
    print(f"Dice numerator: {numerator}")
    print(f"Dice denominator: {denominator}")
    return numerator / denominator

# Training step with JIT
@jax.jit
def train_step(params, state, ct_images, segmentation_masks):
    def loss_fn(params):
        pred = model.apply({'params': params}, ct_images)
        dice = compute_dice_loss(pred, segmentation_masks)
        return 1 - dice
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, state = optimizer.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, loss

model = MedCNN()
rng_init, rng_train = jax.random.split(rng)
dummy_input = jnp.ones((batch, num_slices, channels, width, height))
params = model.init(rng_init, dummy_input)['params']
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

epochs = 5
for epoch in range(epochs):
    params, opt_state, loss = train_step(params, opt_state, ct_images, segmentation_masks)
    print(f"Loss at epoch {epoch}: {loss}")

print("Training completed successfully.")