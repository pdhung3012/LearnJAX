"""
M4 Medical CNN (3D CT Segmentation) - JAX Implementation

Fixed to match PyTorch behavior (including the Dice loss bug for equivalence)
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np

# FIXED: Match PyTorch random seed
rng = jax.random.PRNGKey(42)

# FIXED: Match PyTorch batch size
batch = 100  # Was 5, now 100 to match PyTorch
num_slices = 10
channels = 3
width = 256
height = 256

def generate_synthetic_data(rng, batch, num_slices, channels, width, height):
    """Generate synthetic CT data matching PyTorch"""
    rng_data, rng_masks = jax.random.split(rng)
    ct_images = jax.random.normal(rng_data, (batch, num_slices, channels, width, height))
    segmentation_masks = (jax.random.normal(rng_masks, (batch, num_slices, 1, width, height)) > 0).astype(jnp.float32)
    return ct_images, segmentation_masks

ct_images, segmentation_masks = generate_synthetic_data(rng, batch, num_slices, channels, width, height)
print(f"CT images (train examples) shape: {ct_images.shape}")
print(f"Segmentation binary masks (labels) shape: {segmentation_masks.shape}")


class MedCNN(nn.Module):
    """
    Medical CNN for 3D CT segmentation
    
    Note: Uses simplified backbone instead of pre-trained ResNet18
    For full equivalence, would need to load PyTorch ResNet18 weights
    """
    out_channels: int = 1
    
    @nn.compact
    def __call__(self, x, verbose=False):
        b, d, c, w, h = x.shape
        if verbose:
            print(f"Input shape [B, D, C, W, H]: {(b, d, c, w, h)}")
        
        # Reshape for 2D conv backbone: [B, D, C, W, H] → [B*D, C, W, H]
        x = x.reshape(b * d, c, w, h)
        
        # Convert to NHWC for Flax
        x = jnp.transpose(x, (0, 2, 3, 1))  # [B*D, W, H, C]
        
        # SIMPLIFIED BACKBONE (not pre-trained ResNet18!)
        # PyTorch uses pre-trained ResNet18 with 11.7M params
        # This is a 1-layer approximation matching output shape only
        # Note: This is the main non-equivalence - missing transfer learning
        x = nn.Conv(features=512, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(32, 32), strides=(32, 32), padding='VALID')
        
        # Convert back to NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))  # [B*D, 512, 8, 8]
        
        if verbose:
            print(f"ResNet-like output shape [B*D, C, W, H]: {x.shape}")
        
        # Reshape back to 3D: [B*D, C, W, H] → [B, D, C, W, H]
        _, new_c, new_w, new_h = x.shape
        x = x.reshape(b, d, new_c, new_w, new_h)
        
        # Permute for 3D conv: [B, D, C, W, H] → [B, C, D, W, H]
        x = jnp.transpose(x, (0, 2, 1, 3, 4))
        
        if verbose:
            print(f"Reshape for 3DConv [B, C, D, W, H]: {x.shape}")
        
        # Downsampling 3D convolutions
        x = nn.Conv(features=64, kernel_size=(3, 3, 3), padding='SAME')(x)
        x = nn.relu(x)
        if verbose:
            print(f"Output shape 3D Conv #1: {x.shape}")
        
        x = nn.Conv(features=64, kernel_size=(3, 3, 3), padding='SAME')(x)
        x = nn.relu(x)
        if verbose:
            print(f"Output shape 3D Conv #2: {x.shape}")
        
        # Upsampling with transposed convolutions
        x = nn.ConvTranspose(features=32, kernel_size=(1, 4, 4), strides=(1, 4, 4), padding='VALID')(x)
        x = nn.relu(x)
        if verbose:
            print(f"Output shape 3D Transposed Conv #1: {x.shape}")
        
        x = nn.ConvTranspose(features=16, kernel_size=(1, 8, 8), strides=(1, 8, 8), padding='VALID')(x)
        x = nn.relu(x)
        if verbose:
            print(f"Output shape 3D Transposed Conv #2: {x.shape}")
        
        # Final segmentation layer
        x = nn.Conv(features=self.out_channels, kernel_size=(1, 1, 1))(x)
        x = jax.nn.sigmoid(x)
        
        if verbose:
            print(f"Final shape: {x.shape}")
        
        return x


def compute_dice_coefficient(pred, labels, eps=1e-8):
    """
    Compute Dice coefficient (similarity measure)
    Returns value in [0, 1] where 1 = perfect overlap
    
    Note: This is NOT a loss! Higher is better.
    """
    numerator = 2 * jnp.sum(pred * labels)
    denominator = jnp.sum(pred) + jnp.sum(labels) + eps
    return numerator / denominator


# Training step (not JIT for debugging, can add @jax.jit for speed)
def train_step(params, opt_state, optimizer, ct_images, segmentation_masks, match_pytorch_bug=False):
    """
    Training step
    
    Args:
        match_pytorch_bug: If True, minimizes Dice coefficient (wrong!)
                          If False, maximizes Dice coefficient (correct)
    """
    def loss_fn(params):
        pred = model.apply({'params': params}, ct_images, verbose=False)
        dice = compute_dice_coefficient(pred, segmentation_masks)
        
        if match_pytorch_bug:
            # WRONG: Minimize Dice coefficient (PyTorch bug)
            # This trains model to make worse predictions!
            return dice
        else:
            # CORRECT: Minimize Dice loss = 1 - Dice coefficient
            return 1 - dice
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss


# Initialize model
model = MedCNN(out_channels=1)
rng_init, rng_train = jax.random.split(rng)

# Forward pass to see shapes (verbose mode)
print("\n" + "="*60)
print("Model architecture (first forward pass):")
print("="*60)
dummy_input = jnp.ones((1, num_slices, channels, width, height))
init_vars = model.init(rng_init, dummy_input, verbose=True)
params = init_vars['params']

# Initialize optimizer
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

print("\n" + "="*60)
print("Training:")
print("="*60)

# CHOICE: Match PyTorch bug or use correct loss?
# Set to True to replicate PyTorch's backwards behavior
# Set to False for correct training
MATCH_PYTORCH_BUG = False  # Change to True to replicate PyTorch exactly

if MATCH_PYTORCH_BUG:
    print("WARNING: Matching PyTorch bug - minimizing Dice coefficient (wrong direction!)")
else:
    print("Using correct Dice loss (1 - Dice coefficient)")

epochs = 5
for epoch in range(epochs):
    params, opt_state, loss = train_step(
        params, opt_state, optimizer, ct_images, segmentation_masks,
        match_pytorch_bug=MATCH_PYTORCH_BUG
    )
    
    if MATCH_PYTORCH_BUG:
        print(f"Loss at epoch {epoch}: {loss} (minimizing Dice - WRONG!)")
    else:
        print(f"Loss at epoch {epoch}: {loss} (Dice loss - correct)")

print("\nTraining completed successfully.")
print("\nNOTES:")
print("1. Using batch=100 to match PyTorch (was 5)")
print("2. Using simplified backbone (not pre-trained ResNet18)")
print("3. PyTorch has bug: minimizes Dice coefficient instead of loss")
print("4. Set MATCH_PYTORCH_BUG=True to replicate PyTorch's wrong behavior")