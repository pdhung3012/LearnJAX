"""From-scratch ResNet18-small forward in JAX/jax.numpy.

Adds these patterns over simple_bn_block:
- BasicBlock with optional downsample (Conv1x1 + BN on the residual path).
- Stride-2 conv at the start of layer2.
- 7x7 stride-2 stem conv with explicit padding (3 on each side).
- Max-pool with kernel 3, stride 2, padding 1.
"""
import sys
from pathlib import Path

import jax
import jax.lax as lax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _weight_loader import load_pt_config, load_pt_safetensors


def _conv2d(x_nchw, weight, bias=None, stride=1, padding="SAME"):
    """PT NCHW + (out, in, kH, kW) -> NHWC + HWIO -> back to NCHW."""
    x = jnp.transpose(x_nchw, (0, 2, 3, 1))
    k = jnp.transpose(weight, (2, 3, 1, 0))
    out = lax.conv_general_dilated(
        x, k, (stride, stride), padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    if bias is not None:
        out = out + bias
    return jnp.transpose(out, (0, 3, 1, 2))


def _bn_eval(x, w_prefix, w, eps):
    rm = w[f"{w_prefix}.running_mean"][None, :, None, None]
    rv = w[f"{w_prefix}.running_var"][None, :, None, None]
    weight = w[f"{w_prefix}.weight"][None, :, None, None]
    bias = w[f"{w_prefix}.bias"][None, :, None, None]
    return (x - rm) / jnp.sqrt(rv + eps) * weight + bias


def _basic_block(x, prefix, w, eps, stride, has_downsample):
    """PT BasicBlock: Conv-BN-ReLU-Conv-BN -> + identity -> ReLU."""
    identity = x
    if has_downsample:
        # downsample[0] = Conv1x1 stride=stride, downsample[1] = BN
        identity = _conv2d(x, w[f"{prefix}.downsample.0.weight"],
                           stride=stride, padding="VALID")
        identity = _bn_eval(identity, f"{prefix}.downsample.1", w, eps)

    # PT's nn.Conv2d(padding=1) is symmetric; JAX "SAME" with stride=2 picks
    # asymmetric padding that DOES NOT match PT. Use explicit symmetric pads.
    out = _conv2d(x, w[f"{prefix}.conv1.weight"], stride=stride,
                  padding=((1, 1), (1, 1)))
    out = _bn_eval(out, f"{prefix}.bn1", w, eps)
    out = jax.nn.relu(out)
    out = _conv2d(out, w[f"{prefix}.conv2.weight"], stride=1,
                  padding=((1, 1), (1, 1)))
    out = _bn_eval(out, f"{prefix}.bn2", w, eps)
    return jax.nn.relu(out + identity)


def compute(inputs):
    config = load_pt_config(HERE / "pt_weights")
    eps = config["bn_eps"]
    w = {k: jnp.asarray(v) for k, v in load_pt_safetensors(HERE / "pt_weights").items()}

    x = jnp.asarray(inputs["pixel_values"])

    # Stem: Conv7x7 stride 2 (padding 3), BN, ReLU, MaxPool 3x3 stride 2.
    x = _conv2d(x, w["conv1.weight"], stride=2, padding=((3, 3), (3, 3)))
    x = _bn_eval(x, "bn1", w, eps)
    x = jax.nn.relu(x)
    # Max-pool with NCHW layout -> use Flax helper after transpose.
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    x_nhwc = nn.max_pool(x_nhwc, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
    x = jnp.transpose(x_nhwc, (0, 3, 1, 2))

    # layer1: 2 BasicBlocks (no downsample, stride 1).
    x = _basic_block(x, "layer1_0", w, eps, stride=1, has_downsample=False)
    x = _basic_block(x, "layer1_1", w, eps, stride=1, has_downsample=False)

    # layer2: first block has stride 2 + 16->32 channel change, so downsample.
    x = _basic_block(x, "layer2_0", w, eps, stride=2, has_downsample=True)
    x = _basic_block(x, "layer2_1", w, eps, stride=1, has_downsample=False)

    x = jnp.mean(x, axis=(2, 3))
    logits = x @ w["fc.weight"].T + w["fc.bias"]
    return {"logits": np.asarray(logits)}


if __name__ == "__main__":
    inputs = dict(np.load(HERE / "inputs.npz"))
    out = compute(inputs)
    print("logits shape:", out["logits"].shape)
    print("checksum:", float(out["logits"].sum()))
