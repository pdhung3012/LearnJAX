import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state
import optax
import numpy as np


# Generate synthetic CT-scan data (batches, slices, RGB) and associated segmentation masks
# This mirrors the original PyTorch tensor shapes and is memory intensive.
key = random.PRNGKey(42)
key, image_key, mask_key = random.split(key, 3)
batch = 100
num_slices = 10
channels = 3
width = 256
height = 256

ct_images = random.normal(image_key, (batch, num_slices, channels, width, height))
segmentation_masks = (random.normal(mask_key, (batch, num_slices, 1, width, height)) > 0).astype(
    jnp.float32
)

print(f"CT images (train examples) shape: {ct_images.shape}")
print(f"Segmentation binary masks (labels) shape: {segmentation_masks.shape}")


class BasicBlock(nn.Module):
    features: int
    stride: int = 1
    downsample: bool = False

    @nn.compact
    def __call__(self, x, train=True):
        residual = x

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            name="conv1",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            name="conv2",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn2")(x)

        if self.downsample:
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding="VALID",
                use_bias=False,
                name="downsample_conv",
            )(residual)
            residual = nn.BatchNorm(use_running_average=not train, name="downsample_bn")(residual)

        x = x + residual
        return nn.relu(x)


class ResNet18Backbone(nn.Module):
    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            use_bias=False,
            name="conv1",
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        # Layer1
        x = BasicBlock(features=64, stride=1, downsample=False, name="layer1_block1")(x, train=train)
        x = BasicBlock(features=64, stride=1, downsample=False, name="layer1_block2")(x, train=train)
        # Layer2
        x = BasicBlock(features=128, stride=2, downsample=True, name="layer2_block1")(x, train=train)
        x = BasicBlock(features=128, stride=1, downsample=False, name="layer2_block2")(x, train=train)
        # Layer3
        x = BasicBlock(features=256, stride=2, downsample=True, name="layer3_block1")(x, train=train)
        x = BasicBlock(features=256, stride=1, downsample=False, name="layer3_block2")(x, train=train)
        # Layer4
        x = BasicBlock(features=512, stride=2, downsample=True, name="layer4_block1")(x, train=train)
        x = BasicBlock(features=512, stride=1, downsample=False, name="layer4_block2")(x, train=train)
        return x


# Define the MedCNN class and its forward method
class MedCNN(nn.Module):
    @nn.compact
    def __call__(self, x, train=True):
        b, d, c, w, h = x.shape  # Input size: [B, D, C, W, H]
        print(f"Input shape [B, D, C, W, H]: {b, d, c, w, h}")

        # PyTorch path: [B, D, C, W, H] -> [B*D, C, W, H]
        # Flax backbone uses NHWC, so convert to [B*D, W, H, C] before applying it.
        x = jnp.transpose(x, (0, 1, 3, 4, 2)).reshape((b * d, w, h, c))
        features = ResNet18Backbone(name="backbone")(x, train=train)  # [B*D, W', H', C']
        features_cf = jnp.transpose(features, (0, 3, 1, 2))  # [B*D, C', W', H']
        print(f"ResNet output shape[B*D, C, W, H]: {features_cf.shape}")

        _, new_c, new_w, new_h = features_cf.shape
        x = features_cf.reshape((b, d, new_c, new_w, new_h))  # [B, D, C, W, H]
        x = jnp.transpose(x, (0, 2, 1, 3, 4))  # [B, C, D, W, H]
        print(f"Reshape Resnet output for 3DConv #1 [B, C, D, W, H]: {x.shape}")

        # Flax 3D conv uses NDHWC; convert [B, C, D, W, H] -> [B, D, W, H, C]
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        # Downsampling
        x = nn.Conv(features=64, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)), name="conv1")(x)
        x = nn.relu(x)
        print(f"Output shape 3D Conv #1: {jnp.transpose(x, (0, 4, 1, 2, 3)).shape}")

        x = nn.Conv(features=64, kernel_size=(3, 3, 3), padding=((1, 1), (1, 1), (1, 1)), name="conv2")(x)
        x = nn.relu(x)
        print(f"Output shape 3D Conv #2: {jnp.transpose(x, (0, 4, 1, 2, 3)).shape}")

        # Upsampling (PyTorch ConvTranspose3d has padding=0 by default -> VALID in Flax)
        x = nn.ConvTranspose(
            features=32,
            kernel_size=(1, 4, 4),
            strides=(1, 4, 4),
            padding="VALID",
            name="conv_transpose1",
        )(x)
        x = nn.relu(x)
        print(f"Output shape 3D Transposed Conv #1: {jnp.transpose(x, (0, 4, 1, 2, 3)).shape}")

        x = nn.ConvTranspose(
            features=16,
            kernel_size=(1, 8, 8),
            strides=(1, 8, 8),
            padding="VALID",
            name="conv_transpose2",
        )(x)
        x = nn.relu(x)
        print(f"Output shape 3D Transposed Conv #2: {jnp.transpose(x, (0, 4, 1, 2, 3)).shape}")

        # Final segmentation -> return as [B, 1, D, W, H] to mirror PyTorch output layout
        x = nn.Conv(features=1, kernel_size=(1, 1, 1), name="final_conv")(x)
        x = jax.nn.sigmoid(x)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        print(f"Final shape: {x.shape}")

        return x


def compute_dice_loss(pred, labels, eps=1e-8):
    """
    Args
    pred: [B, D, 1, W, H]
    labels: [B, D, 1, W, H]

    Returns
    dice_loss: [B, D, 1, W, H]
    """
    numerator = 2 * jnp.sum(pred * labels)
    denominator = jnp.sum(pred) + jnp.sum(labels) + eps
    return numerator / denominator


class TrainState(train_state.TrainState):
    batch_stats: dict


def _to_hwio(weight_oi_hw):
    return np.transpose(weight_oi_hw, (2, 3, 1, 0))


def _assign_conv2d(bb_params, torch_state, flax_name, torch_name):
    kernel = _to_hwio(torch_state[f"{torch_name}.weight"].detach().cpu().numpy())
    bb_params[flax_name]["kernel"] = jnp.asarray(kernel)


def _assign_bn(bb_params, bb_stats, torch_state, flax_name, torch_name):
    bb_params[flax_name]["scale"] = jnp.asarray(torch_state[f"{torch_name}.weight"].detach().cpu().numpy())
    bb_params[flax_name]["bias"] = jnp.asarray(torch_state[f"{torch_name}.bias"].detach().cpu().numpy())
    bb_stats[flax_name]["mean"] = jnp.asarray(
        torch_state[f"{torch_name}.running_mean"].detach().cpu().numpy()
    )
    bb_stats[flax_name]["var"] = jnp.asarray(torch_state[f"{torch_name}.running_var"].detach().cpu().numpy())


def load_pretrained_torchvision_backbone(state):
    """Load torchvision ResNet18 pretrained weights into the Flax backbone."""
    import torchvision

    try:
        from torchvision.models import ResNet18_Weights, resnet18

        torch_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        # Fallback for older torchvision API.
        torch_model = torchvision.models.resnet18(pretrained=True)

    torch_state = torch_model.state_dict()

    params = unfreeze(state.params)
    batch_stats = unfreeze(state.batch_stats)
    bb_params = params["backbone"]
    bb_stats = batch_stats["backbone"]

    _assign_conv2d(bb_params, torch_state, "conv1", "conv1")
    _assign_bn(bb_params, bb_stats, torch_state, "bn1", "bn1")

    for layer_idx in range(1, 5):
        for block_idx in range(1, 3):
            flax_block = f"layer{layer_idx}_block{block_idx}"
            torch_block = f"layer{layer_idx}.{block_idx - 1}"

            _assign_conv2d(bb_params[flax_block], torch_state, "conv1", f"{torch_block}.conv1")
            _assign_bn(bb_params[flax_block], bb_stats[flax_block], torch_state, "bn1", f"{torch_block}.bn1")
            _assign_conv2d(bb_params[flax_block], torch_state, "conv2", f"{torch_block}.conv2")
            _assign_bn(bb_params[flax_block], bb_stats[flax_block], torch_state, "bn2", f"{torch_block}.bn2")

            if layer_idx > 1 and block_idx == 1:
                _assign_conv2d(
                    bb_params[flax_block],
                    torch_state,
                    "downsample_conv",
                    f"{torch_block}.downsample.0",
                )
                _assign_bn(
                    bb_params[flax_block],
                    bb_stats[flax_block],
                    torch_state,
                    "downsample_bn",
                    f"{torch_block}.downsample.1",
                )

    return state.replace(params=freeze(params), batch_stats=freeze(batch_stats))


model = MedCNN()
variables = model.init(random.PRNGKey(0), ct_images, train=True)
state = TrainState.create(
    apply_fn=model.apply,
    params=variables["params"],
    batch_stats=variables["batch_stats"],
    tx=optax.adam(learning_rate=0.01),
)

try:
    state = load_pretrained_torchvision_backbone(state)
except Exception as exc:
    raise RuntimeError(
        "Could not load pretrained torchvision ResNet18 weights; this translation requires pretrained=True parity."
    ) from exc
print("Loaded torchvision ResNet18 pretrained weights into Flax backbone.")


@jax.jit
def train_step(state, images, labels):
    def loss_fn(params):
        predictions, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            images,
            train=True,
            mutable=["batch_stats"],
        )
        loss = compute_dice_loss(predictions, labels)
        return loss, updates["batch_stats"]

    (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)
    return state, loss


epochs = 5
for epoch in range(epochs):
    state, loss = train_step(state, ct_images, segmentation_masks)
    print(f"Loss at epoch {epoch}: {loss}")
