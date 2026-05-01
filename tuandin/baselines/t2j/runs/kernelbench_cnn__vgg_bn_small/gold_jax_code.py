"""From-scratch VGG-BN-small forward in JAX/jax.numpy."""
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


def _conv2d(x_nchw, weight, bias=None, stride=1, padding="VALID"):
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


def _max_pool_2x2(x_nchw):
    x = jnp.transpose(x_nchw, (0, 2, 3, 1))
    x = nn.max_pool(x, (2, 2), strides=(2, 2))
    return jnp.transpose(x, (0, 3, 1, 2))


def _conv_bn_relu(x, conv_w, bn_prefix, w, eps):
    x = _conv2d(x, conv_w, padding=((1, 1), (1, 1)))
    x = _bn_eval(x, bn_prefix, w, eps)
    return jax.nn.relu(x)


def compute(inputs):
    config = load_pt_config(HERE / "pt_weights")
    eps = config["bn_eps"]
    w = {k: jnp.asarray(v) for k, v in load_pt_safetensors(HERE / "pt_weights").items()}

    x = jnp.asarray(inputs["pixel_values"])

    x = _conv_bn_relu(x, w["conv1.weight"], "bn1", w, eps)
    x = _conv_bn_relu(x, w["conv2.weight"], "bn2", w, eps)
    x = _max_pool_2x2(x)
    x = _conv_bn_relu(x, w["conv3.weight"], "bn3", w, eps)
    x = _conv_bn_relu(x, w["conv4.weight"], "bn4", w, eps)
    x = _max_pool_2x2(x)

    x = x.reshape(x.shape[0], -1)               # NCHW flatten
    logits = x @ w["fc.weight"].T + w["fc.bias"]
    return {"logits": np.asarray(logits)}


if __name__ == "__main__":
    inputs = dict(np.load(HERE / "inputs.npz"))
    out = compute(inputs)
    print("logits shape:", out["logits"].shape)
    print("checksum:", float(out["logits"].sum()))
