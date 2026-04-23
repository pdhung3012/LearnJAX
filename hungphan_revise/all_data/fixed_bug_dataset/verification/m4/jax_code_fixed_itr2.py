import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import optax
import numpy as np
import torch
import torchvision.models as models


BATCH = 100
NUM_SLICES = 10
CHANNELS = 3
WIDTH = 256
HEIGHT = 256
LEARNING_RATE = 0.01
EPOCHS = 5

RESNET_BLOCK_NAME_MAP = (
    ("layer1.0", "layer1_block0"),
    ("layer1.1", "layer1_block1"),
    ("layer2.0", "layer2_block0"),
    ("layer2.1", "layer2_block1"),
    ("layer3.0", "layer3_block0"),
    ("layer3.1", "layer3_block1"),
    ("layer4.0", "layer4_block0"),
    ("layer4.1", "layer4_block1"),
)


def generate_synthetic_data(batch, num_slices, channels, width, height):
    """Match the source PyTorch data generation exactly."""
    torch.manual_seed(42)
    ct_images = torch.randn(size=(batch, num_slices, channels, width, height))
    segmentation_masks = (
        torch.randn(size=(batch, num_slices, 1, width, height)) > 0
    ).float()
    return jnp.asarray(ct_images.numpy()), jnp.asarray(segmentation_masks.numpy())


def get_torchvision_resnet18():
    """Load Torchvision's pretrained ResNet18 across old/new torchvision APIs."""
    if hasattr(models, "ResNet18_Weights"):
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    return models.resnet18(pretrained=True)


def torch_tensor_to_numpy(tensor):
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)


def convert_conv2d_kernel(torch_kernel):
    """PyTorch OIHW -> Flax HWIO."""
    return jnp.asarray(np.transpose(torch_tensor_to_numpy(torch_kernel), (2, 3, 1, 0)))


def assign_batch_norm(flax_params, flax_batch_stats, torch_state, prefix):
    flax_params["scale"] = jnp.asarray(
        torch_tensor_to_numpy(torch_state[f"{prefix}.weight"]))
    flax_params["bias"] = jnp.asarray(
        torch_tensor_to_numpy(torch_state[f"{prefix}.bias"]))
    flax_batch_stats["mean"] = jnp.asarray(
        torch_tensor_to_numpy(torch_state[f"{prefix}.running_mean"])
    )
    flax_batch_stats["var"] = jnp.asarray(
        torch_tensor_to_numpy(torch_state[f"{prefix}.running_var"])
    )


class ResNet18BasicBlock(nn.Module):
    features: int
    strides: tuple[int, int] = (1, 1)
    use_projection: bool = False

    @nn.compact
    def __call__(self, x, train=True):
        residual = x

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=self.strides,
            padding=((1, 1), (1, 1)),
            use_bias=False,
            name="conv1",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            name="bn1",
        )(x)
        x = nn.relu(x)

        x = nn.Conv(
            features=self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            name="conv2",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            name="bn2",
        )(x)

        if self.use_projection:
            residual = nn.Conv(
                features=self.features,
                kernel_size=(1, 1),
                strides=self.strides,
                padding=((0, 0), (0, 0)),
                use_bias=False,
                name="downsample_conv",
            )(residual)
            residual = nn.BatchNorm(
                use_running_average=not train,
                momentum=0.9,
                epsilon=1e-5,
                name="downsample_bn",
            )(residual)

        return nn.relu(x + residual)


class TruncatedResNet18Backbone(nn.Module):
    """Matches nn.Sequential(*list(resnet18.children())[:-2])."""

    @nn.compact
    def __call__(self, x, train=True):
        x = jnp.transpose(x, (0, 2, 3, 1))

        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            use_bias=False,
            name="conv1",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            name="bn1",
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(
            x,
            window_shape=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
        )

        x = ResNet18BasicBlock(64, name="layer1_block0")(x, train=train)
        x = ResNet18BasicBlock(64, name="layer1_block1")(x, train=train)
        x = ResNet18BasicBlock(
            128, strides=(2, 2), use_projection=True, name="layer2_block0"
        )(x, train=train)
        x = ResNet18BasicBlock(128, name="layer2_block1")(x, train=train)
        x = ResNet18BasicBlock(
            256, strides=(2, 2), use_projection=True, name="layer3_block0"
        )(x, train=train)
        x = ResNet18BasicBlock(256, name="layer3_block1")(x, train=train)
        x = ResNet18BasicBlock(
            512, strides=(2, 2), use_projection=True, name="layer4_block0"
        )(x, train=train)
        x = ResNet18BasicBlock(512, name="layer4_block1")(x, train=train)

        return jnp.transpose(x, (0, 3, 1, 2))


class MedCNN(nn.Module):
    backbone: nn.Module
    out_channel: int = 1

    @nn.compact
    def __call__(self, x, train=True, verbose=False):
        b, d, c, w, h = x.shape
        if verbose:
            print(f"Input shape [B, D, C, W, H]: {(b, d, c, w, h)}")

        x = x.reshape(b * d, c, w, h)
        features = self.backbone(x, train=train)
        if verbose:
            print(f"ResNet output shape[B*D, C, W, H]: {features.shape}")

        _, new_c, new_w, new_h = features.shape
        x = features.reshape(b, d, new_c, new_w, new_h)
        x = jnp.transpose(x, (0, 2, 1, 3, 4))
        if verbose:
            print(
                f"Reshape Resnet output for 3DConv #1 [B, C, D, W, H]: {x.shape}")

        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3, 3),
            padding=((1, 1), (1, 1), (1, 1)),
            name="conv1",
        )(x)
        x = nn.relu(x)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        if verbose:
            print(f"Output shape 3D Conv #1: {x.shape}")

        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3, 3),
            padding=((1, 1), (1, 1), (1, 1)),
            name="conv2",
        )(x)
        x = nn.relu(x)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        if verbose:
            print(f"Output shape 3D Conv #2: {x.shape}")

        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = nn.ConvTranspose(
            features=32,
            kernel_size=(1, 4, 4),
            strides=(1, 4, 4),
            padding="VALID",
            name="conv_transpose1",
        )(x)
        x = nn.relu(x)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        if verbose:
            print(f"Output shape 3D Transposed Conv #1: {x.shape}")

        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = nn.ConvTranspose(
            features=16,
            kernel_size=(1, 8, 8),
            strides=(1, 8, 8),
            padding="VALID",
            name="conv_transpose2",
        )(x)
        x = nn.relu(x)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        if verbose:
            print(f"Output shape 3D Transposed Conv #2: {x.shape}")

        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = nn.Conv(
            features=self.out_channel,
            kernel_size=(1, 1, 1),
            padding="VALID",
            name="final_conv",
        )(x)
        x = jax.nn.sigmoid(x)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        if verbose:
            print(f"Final shape: {x.shape}")

        return x


def load_pretrained_resnet18_backbone(variables, backbone_key=None):
    """Copy Torchvision ResNet18 weights into a matching Flax backbone."""
    torch_model = get_torchvision_resnet18()
    torch_state = torch_model.state_dict()

    variables = unfreeze(variables)
    if backbone_key is None:
        backbone_params = variables["params"]
        backbone_batch_stats = variables["batch_stats"]
    else:
        backbone_params = variables["params"][backbone_key]
        backbone_batch_stats = variables["batch_stats"][backbone_key]

    backbone_params["conv1"]["kernel"] = convert_conv2d_kernel(
        torch_state["conv1.weight"])
    assign_batch_norm(
        backbone_params["bn1"], backbone_batch_stats["bn1"], torch_state, "bn1")

    for torch_prefix, flax_name in RESNET_BLOCK_NAME_MAP:
        block_params = backbone_params[flax_name]
        block_batch_stats = backbone_batch_stats[flax_name]

        block_params["conv1"]["kernel"] = convert_conv2d_kernel(
            torch_state[f"{torch_prefix}.conv1.weight"]
        )
        assign_batch_norm(
            block_params["bn1"], block_batch_stats[
                "bn1"], torch_state, f"{torch_prefix}.bn1"
        )

        block_params["conv2"]["kernel"] = convert_conv2d_kernel(
            torch_state[f"{torch_prefix}.conv2.weight"]
        )
        assign_batch_norm(
            block_params["bn2"], block_batch_stats[
                "bn2"], torch_state, f"{torch_prefix}.bn2"
        )

        if "downsample_conv" in block_params:
            block_params["downsample_conv"]["kernel"] = convert_conv2d_kernel(
                torch_state[f"{torch_prefix}.downsample.0.weight"]
            )
            assign_batch_norm(
                block_params["downsample_bn"],
                block_batch_stats["downsample_bn"],
                torch_state,
                f"{torch_prefix}.downsample.1",
            )

    return freeze(variables)


def initialize_model_variables(model, rng_init):
    dummy_input = jnp.ones(
        (1, NUM_SLICES, CHANNELS, WIDTH, HEIGHT), dtype=jnp.float32)
    variables = model.init(rng_init, dummy_input, train=True, verbose=False)
    variables = load_pretrained_resnet18_backbone(
        variables, backbone_key="backbone")
    return variables


def compute_dice_loss(pred, labels, eps=1e-8):
    """
    Matches the PyTorch implementation exactly.

    Note: despite the name, this returns the Dice coefficient and the training
    loop minimizes it, which mirrors the source file's behavior.
    """
    numerator = 2 * jnp.sum(pred * labels)
    denominator = jnp.sum(pred) + jnp.sum(labels) + eps
    return numerator / denominator


def train_step(model, optimizer, params, batch_stats, opt_state, ct_images, segmentation_masks):
    def loss_fn(current_params):
        variables = {"params": current_params, "batch_stats": batch_stats}
        pred, updated_state = model.apply(
            variables,
            ct_images,
            train=True,
            verbose=False,
            mutable=["batch_stats"],
        )
        loss = compute_dice_loss(pred, segmentation_masks)
        return loss, updated_state["batch_stats"]

    (loss, new_batch_stats), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, new_batch_stats, opt_state, loss


def main():
    ct_images, segmentation_masks = generate_synthetic_data(
        BATCH, NUM_SLICES, CHANNELS, WIDTH, HEIGHT
    )
    print(f"CT images (train examples) shape: {ct_images.shape}")
    print(
        f"Segmentation binary masks (labels) shape: {segmentation_masks.shape}")

    model = MedCNN(backbone=TruncatedResNet18Backbone(), out_channel=1)
    rng_init = jax.random.PRNGKey(42)
    variables = initialize_model_variables(model, rng_init)
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    print("Loaded pretrained ResNet18 weights into the JAX backbone.")

    print("\n" + "=" * 60)
    print("Model architecture (first forward pass):")
    print("=" * 60)
    dummy_input = jnp.ones(
        (1, NUM_SLICES, CHANNELS, WIDTH, HEIGHT), dtype=jnp.float32)
    _, updated_state = model.apply(
        {"params": params, "batch_stats": batch_stats},
        dummy_input,
        train=True,
        verbose=True,
        mutable=["batch_stats"],
    )
    batch_stats = updated_state["batch_stats"]

    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params)

    print("\n" + "=" * 60)
    print("Training:")
    print("=" * 60)

    for epoch in range(EPOCHS):
        params, batch_stats, opt_state, loss = train_step(
            model,
            optimizer,
            params,
            batch_stats,
            opt_state,
            ct_images,
            segmentation_masks,
        )
        print(f"Loss at epoch {epoch}: {loss}")


if __name__ == "__main__":
    main()
