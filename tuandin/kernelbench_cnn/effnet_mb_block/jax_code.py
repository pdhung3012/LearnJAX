"""From-scratch EfficientNet-style MBConv-with-SE block in JAX/jax.numpy."""
import sys
from pathlib import Path

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _weight_loader import load_pt_config, load_pt_safetensors


def _conv2d(x_nchw, weight, bias=None, stride=1, padding="VALID", groups=1):
    x = jnp.transpose(x_nchw, (0, 2, 3, 1))
    k = jnp.transpose(weight, (2, 3, 1, 0))
    out = lax.conv_general_dilated(
        x, k, (stride, stride), padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=groups,
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


def _silu(x):
    return x * jax.nn.sigmoid(x)


def _squeeze_excite(x, w_prefix, w):
    """Global avg pool -> Conv1x1 SiLU -> Conv1x1 Sigmoid -> channel-wise scale."""
    squeeze = jnp.mean(x, axis=(2, 3), keepdims=True)              # (B, C, 1, 1)
    gate = _conv2d(squeeze, w[f"{w_prefix}.fc1.weight"], bias=w[f"{w_prefix}.fc1.bias"])
    gate = _silu(gate)
    gate = _conv2d(gate, w[f"{w_prefix}.fc2.weight"], bias=w[f"{w_prefix}.fc2.bias"])
    gate = jax.nn.sigmoid(gate)
    return x * gate


def compute(inputs):
    config = load_pt_config(HERE / "pt_weights")
    eps = config["bn_eps"]
    c_mid = config["in_channels"] * config["expansion_factor"]
    w = {k: jnp.asarray(v) for k, v in load_pt_safetensors(HERE / "pt_weights").items()}

    x = jnp.asarray(inputs["pixel_values"])
    identity = x

    out = _conv2d(x, w["block.expand_conv.weight"], padding="VALID")
    out = _bn_eval(out, "block.expand_bn", w, eps)
    out = _silu(out)

    out = _conv2d(out, w["block.dw_conv.weight"],
                  stride=1, padding=((1, 1), (1, 1)), groups=c_mid)
    out = _bn_eval(out, "block.dw_bn", w, eps)
    out = _silu(out)

    out = _squeeze_excite(out, "block.se", w)

    out = _conv2d(out, w["block.project_conv.weight"], padding="VALID")
    out = _bn_eval(out, "block.project_bn", w, eps)
    if config["in_channels"] == config["out_channels"]:
        out = out + identity
    return {"output": np.asarray(out)}


if __name__ == "__main__":
    inputs = dict(np.load(HERE / "inputs.npz"))
    out = compute(inputs)
    print("output shape:", out["output"].shape)
    print("checksum:", float(out["output"].sum()))
