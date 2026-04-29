"""From-scratch SimpleBNBlock forward in JAX/jax.numpy.

The eval-mode BatchNorm formula:
    y = (x - running_mean) / sqrt(running_var + eps) * weight + bias

Common cheap-LLM bugs this case targets:
- Using `flax.linen.BatchNorm` without `use_running_average=True` ->
  computes batch statistics, wrong output.
- Forgetting to load `running_mean` / `running_var` from the safetensors
  (loading only `weight` and `bias`) -> uses ones/zeros, wrong output.
- NCHW (PT) vs NHWC (JAX) layout for conv inputs.
- PT Conv2d weight (out, in, kH, kW) -> JAX HWIO (kH, kW, in, out).
"""
import sys
from pathlib import Path

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _weight_loader import load_pt_config, load_pt_safetensors


def _conv2d(x, weight, bias):
    """PT NCHW input, PT (out, in, kH, kW) kernel -> JAX NHWC + HWIO."""
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    kernel_hwio = jnp.transpose(weight, (2, 3, 1, 0))
    out = lax.conv_general_dilated(
        x_nhwc, kernel_hwio, (1, 1), "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    ) + bias
    return jnp.transpose(out, (0, 3, 1, 2))   # back to NCHW for the rest of the chain.


def _bn_eval(x, running_mean, running_var, weight, bias, eps):
    # x is NCHW; per-channel BN scales/biases broadcast along channel dim 1.
    rm = running_mean[None, :, None, None]
    rv = running_var[None, :, None, None]
    w = weight[None, :, None, None]
    b = bias[None, :, None, None]
    return (x - rm) / jnp.sqrt(rv + eps) * w + b


def compute(inputs):
    config = load_pt_config(HERE / "pt_weights")
    eps = config["bn_eps"]
    w = {k: jnp.asarray(v) for k, v in load_pt_safetensors(HERE / "pt_weights").items()}

    x = jnp.asarray(inputs["pixel_values"])

    x = _conv2d(x, w["conv1.weight"], w["conv1.bias"])
    x = _bn_eval(x, w["bn1.running_mean"], w["bn1.running_var"],
                 w["bn1.weight"], w["bn1.bias"], eps)
    x = jax.nn.relu(x)

    x = _conv2d(x, w["conv2.weight"], w["conv2.bias"])
    x = _bn_eval(x, w["bn2.running_mean"], w["bn2.running_var"],
                 w["bn2.weight"], w["bn2.bias"], eps)
    x = jax.nn.relu(x)

    x = jnp.mean(x, axis=(2, 3))                # global avg pool: (B, C)
    logits = x @ w["fc.weight"].T + w["fc.bias"]
    return {"logits": np.asarray(logits)}


if __name__ == "__main__":
    inputs = dict(np.load(HERE / "inputs.npz"))
    out = compute(inputs)
    print("logits shape:", out["logits"].shape)
    print("checksum:", float(out["logits"].sum()))
