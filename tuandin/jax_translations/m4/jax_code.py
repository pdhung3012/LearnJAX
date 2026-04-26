"""JAX translation of m4: medical-image segmentation with ResNet18 backbone + 3D head.

Faithful to PyTorch:
- Synthetic CT data of shape (100, 10, 3, 256, 256), random binary masks of shape
  (100, 10, 1, 256, 256). batch=100, num_slices=10.
- Backbone: torchvision.models.resnet18(pretrained=True) with the avgpool+fc removed
  (so the backbone outputs 512-channel feature maps at 8x8 spatial resolution for a
  256x256 input). We port the pretrained weights from torchvision into Flax.
- Head: Conv3D 512->64 (k=3, pad=1), Conv3D 64->64 (k=3, pad=1), ConvTranspose3D
  64->32 (k=(1,4,4), s=(1,4,4)), ConvTranspose3D 32->16 (k=(1,8,8), s=(1,8,8)),
  Conv3D 16->1 (k=1) -> sigmoid. ReLU between conv stages.
- Loss: dice = 2*sum(p*y) / (sum(p) + sum(y) + eps). Adam(lr=0.01), 5 epochs.

Speed notes: this is heavy on conv ops; on CPU PyTorch's MKL-DNN paths are usually
faster than XLA for 3D transposed conv. JAX likely SLOWER on CPU here. On GPU
the gap closes substantially.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import torch
import torchvision


# ---- Flax ResNet18 backbone (matches torchvision arch up to layer4) ----------

class BasicBlock(nn.Module):
    out_channels: int
    stride: int = 1
    downsample: bool = False

    @nn.compact
    def __call__(self, x):
        identity = x
        y = nn.Conv(self.out_channels, (3, 3), strides=self.stride, padding="SAME",
                    use_bias=False, name="conv1")(x)
        y = nn.BatchNorm(use_running_average=True, name="bn1")(y)
        y = nn.relu(y)
        y = nn.Conv(self.out_channels, (3, 3), strides=1, padding="SAME",
                    use_bias=False, name="conv2")(y)
        y = nn.BatchNorm(use_running_average=True, name="bn2")(y)

        if self.downsample:
            identity = nn.Conv(self.out_channels, (1, 1), strides=self.stride,
                               padding="VALID", use_bias=False, name="downsample_conv")(x)
            identity = nn.BatchNorm(use_running_average=True, name="downsample_bn")(identity)
        return nn.relu(y + identity)


class ResNet18Backbone(nn.Module):
    """Returns the layer4 output (512 channels) — equivalent to
    nn.Sequential(*list(resnet18.children())[:-2])."""

    @nn.compact
    def __call__(self, x):
        # Stem: conv 7x7 stride 2 -> bn -> relu -> maxpool 3x3 stride 2.
        x = nn.Conv(64, (7, 7), strides=2, padding=[(3, 3), (3, 3)],
                    use_bias=False, name="conv1")(x)
        x = nn.BatchNorm(use_running_average=True, name="bn1")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        # 4 layers, each with 2 BasicBlocks.
        for layer_idx, (channels, first_stride) in enumerate(
            [(64, 1), (128, 2), (256, 2), (512, 2)]
        ):
            for block_idx in range(2):
                stride = first_stride if block_idx == 0 else 1
                downsample = (block_idx == 0) and (layer_idx > 0 or False)
                # Layer 1 block 0 has stride=1 and same channels, so no downsample.
                if layer_idx == 0 and block_idx == 0:
                    downsample = False
                x = BasicBlock(channels, stride=stride, downsample=downsample,
                               name=f"layer{layer_idx + 1}_block{block_idx}")(x)
        return x  # (B, H/32, W/32, 512)


# ---- Port pretrained torchvision weights into the Flax backbone --------------

def port_resnet18_pretrained(flax_params):
    """Copy weights from torchvision.models.resnet18(pretrained=True) into the
    Flax param tree. Layout differences:
    - Conv kernels: torch (out, in, kh, kw) -> flax (kh, kw, in, out).
    - BatchNorm: torch has weight, bias, running_mean, running_var, num_batches_tracked.
      Flax BatchNorm in eval mode uses scale, bias, mean, var.
    """
    tv_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    tv_state = tv_model.state_dict()

    def conv_t2f(t):
        return jnp.asarray(t.numpy()).transpose(2, 3, 1, 0)

    def bn_t2f(prefix):
        return {
            "scale": jnp.asarray(tv_state[f"{prefix}.weight"].numpy()),
            "bias":  jnp.asarray(tv_state[f"{prefix}.bias"].numpy()),
            "mean":  jnp.asarray(tv_state[f"{prefix}.running_mean"].numpy()),
            "var":   jnp.asarray(tv_state[f"{prefix}.running_var"].numpy()),
        }

    p = {"params": dict(flax_params["params"]), "batch_stats": dict(flax_params["batch_stats"])}

    p["params"]["conv1"] = {"kernel": conv_t2f(tv_state["conv1.weight"])}
    bn1 = bn_t2f("bn1")
    p["params"]["bn1"] = {"scale": bn1["scale"], "bias": bn1["bias"]}
    p["batch_stats"]["bn1"] = {"mean": bn1["mean"], "var": bn1["var"]}

    layer_chs = [(64, 1), (128, 2), (256, 2), (512, 2)]
    for li, (_, first_stride) in enumerate(layer_chs):
        for bi in range(2):
            block_name = f"layer{li + 1}_block{bi}"
            tv_block = f"layer{li + 1}.{bi}"

            p["params"][block_name] = {
                "conv1": {"kernel": conv_t2f(tv_state[f"{tv_block}.conv1.weight"])},
                "conv2": {"kernel": conv_t2f(tv_state[f"{tv_block}.conv2.weight"])},
            }
            for bn in ("bn1", "bn2"):
                bnp = bn_t2f(f"{tv_block}.{bn}")
                p["params"][block_name][bn] = {"scale": bnp["scale"], "bias": bnp["bias"]}
                p["batch_stats"].setdefault(block_name, {})[bn] = {
                    "mean": bnp["mean"], "var": bnp["var"]
                }

            has_downsample = bi == 0 and li > 0
            if has_downsample:
                p["params"][block_name]["downsample_conv"] = {
                    "kernel": conv_t2f(tv_state[f"{tv_block}.downsample.0.weight"])
                }
                bnp = bn_t2f(f"{tv_block}.downsample.1")
                p["params"][block_name]["downsample_bn"] = {
                    "scale": bnp["scale"], "bias": bnp["bias"]
                }
                p["batch_stats"].setdefault(block_name, {})["downsample_bn"] = {
                    "mean": bnp["mean"], "var": bnp["var"]
                }

    return p


# ---- 3D segmentation head ----------------------------------------------------

class MedCNN(nn.Module):
    backbone: nn.Module

    @nn.compact
    def __call__(self, x):
        # x: (B, D, H, W, C=3) channels-last; after flatten (B*D, H, W, 3).
        b, d, h, w, c = x.shape
        print(f"Input shape [B, D, H, W, C]: {(b, d, h, w, c)}")

        x_flat = x.reshape(b * d, h, w, c)
        feats = self.backbone(x_flat)  # (B*D, h', w', 512)
        print(f"ResNet output shape[B*D, H, W, C]: {feats.shape}")

        _, fh, fw, fc = feats.shape
        feats = feats.reshape(b, d, fh, fw, fc)
        # Rearrange for 3D convs: Flax 3D conv expects (B, D, H, W, C) channels-last,
        # which matches what we have already.
        print(f"Reshape Resnet output for 3DConv #1 [B, D, H, W, C]: {feats.shape}")

        x = nn.relu(nn.Conv(64, (3, 3, 3), padding="SAME", name="conv1_3d")(feats))
        print(f"Output shape 3D Conv #1: {x.shape}")
        x = nn.relu(nn.Conv(64, (3, 3, 3), padding="SAME", name="conv2_3d")(x))
        print(f"Output shape 3D Conv #2: {x.shape}")

        x = nn.relu(nn.ConvTranspose(32, (1, 4, 4), strides=(1, 4, 4),
                                     padding="VALID", name="convT1")(x))
        print(f"Output shape 3D Transposed Conv #1: {x.shape}")
        x = nn.relu(nn.ConvTranspose(16, (1, 8, 8), strides=(1, 8, 8),
                                     padding="VALID", name="convT2")(x))
        print(f"Output shape 3D Transposed Conv #2: {x.shape}")

        x = jax.nn.sigmoid(nn.Conv(1, (1, 1, 1), padding="SAME", name="final_conv")(x))
        print(f"Final shape: {x.shape}")
        return x


def dice_loss(pred, labels, eps=1e-8):
    num = 2.0 * jnp.sum(pred * labels)
    den = jnp.sum(pred) + jnp.sum(labels) + eps
    return num / den


def main():
    key = jax.random.PRNGKey(42)

    batch = 100
    num_slices = 10
    channels = 3
    width = height = 256

    # PyTorch shape: (B, D, C, H, W). We use channels-last (B, D, H, W, C).
    key, kx, ky = jax.random.split(key, 3)
    ct_images = jax.random.normal(kx, (batch, num_slices, height, width, channels))
    segmentation_masks = (jax.random.normal(ky, (batch, num_slices, height, width, 1)) > 0
                          ).astype(jnp.float32)

    print(f"CT images (train examples) shape: {ct_images.shape}")
    print(f"Segmentation binary masks (labels) shape: {segmentation_masks.shape}")

    backbone = ResNet18Backbone()
    model = MedCNN(backbone=backbone)

    key, k_init = jax.random.split(key)
    sample = jnp.zeros((1, 1, height, width, channels))
    init_vars = model.init(k_init, sample)

    # Port pretrained backbone weights.
    backbone_only = {
        "params": init_vars["params"]["backbone"],
        "batch_stats": init_vars["batch_stats"]["backbone"],
    }
    ported = port_resnet18_pretrained(backbone_only)
    init_vars = {
        "params": {**init_vars["params"], "backbone": ported["params"]},
        "batch_stats": {**init_vars["batch_stats"], "backbone": ported["batch_stats"]},
    }
    params = init_vars["params"]
    batch_stats = init_vars["batch_stats"]

    opt = optax.adam(0.01)
    opt_state = opt.init(params)

    def loss_fn(params, batch_stats, x, y):
        pred = model.apply({"params": params, "batch_stats": batch_stats}, x)
        return dice_loss(pred, y)

    @jax.jit
    def train_step(params, batch_stats, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_stats, x, y)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    epochs = 5
    for epoch in range(epochs):
        params, opt_state, loss = train_step(
            params, batch_stats, opt_state, ct_images, segmentation_masks
        )
        print(f"Loss at epoch {epoch}: {loss}")


if __name__ == "__main__":
    main()
