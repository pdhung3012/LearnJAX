"""m3 equivalence test: VanillaCNN forward pass with shared weights.

Tests the algorithmic equivalence of Conv -> ReLU -> Conv -> ReLU -> MaxPool ->
Linear -> ReLU -> Linear with the *same* weights. We don't run the full
training because (a) it downloads CIFAR-10 and (b) training paths diverge by
RNG.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import jax.numpy as jnp
import flax.linen as fnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close


def main():
    rng = np.random.default_rng(0)
    # PyTorch conv weights: (out_c, in_c, k, k); flax expects (k, k, in, out).
    c1_w = rng.standard_normal((32, 3, 3, 3)).astype(np.float32) * 0.1
    c1_b = rng.standard_normal((32,)).astype(np.float32) * 0.05
    c2_w = rng.standard_normal((64, 32, 3, 3)).astype(np.float32) * 0.1
    c2_b = rng.standard_normal((64,)).astype(np.float32) * 0.05
    f1_w = rng.standard_normal((128, 64 * 16 * 16)).astype(np.float32) * 0.01
    f1_b = rng.standard_normal((128,)).astype(np.float32) * 0.05
    f2_w = rng.standard_normal((10, 128)).astype(np.float32) * 0.05
    f2_b = rng.standard_normal((10,)).astype(np.float32) * 0.05
    x = rng.standard_normal((2, 3, 32, 32)).astype(np.float32)

    # PyTorch.
    pt_c1 = nn.Conv2d(3, 32, 3, padding=1); pt_c2 = nn.Conv2d(32, 64, 3, padding=1)
    pt_pool = nn.MaxPool2d(2, 2)
    pt_f1 = nn.Linear(64*16*16, 128); pt_f2 = nn.Linear(128, 10)
    with torch.no_grad():
        pt_c1.weight.copy_(torch.from_numpy(c1_w)); pt_c1.bias.copy_(torch.from_numpy(c1_b))
        pt_c2.weight.copy_(torch.from_numpy(c2_w)); pt_c2.bias.copy_(torch.from_numpy(c2_b))
        pt_f1.weight.copy_(torch.from_numpy(f1_w)); pt_f1.bias.copy_(torch.from_numpy(f1_b))
        pt_f2.weight.copy_(torch.from_numpy(f2_w)); pt_f2.bias.copy_(torch.from_numpy(f2_b))
    h = torch.relu(pt_c1(torch.from_numpy(x)))
    h = pt_pool(torch.relu(pt_c2(h)))
    h = h.view(h.size(0), -1)
    h = torch.relu(pt_f1(h))
    out_pt = pt_f2(h).detach().numpy()

    # JAX (channels-last).
    x_jx = jnp.asarray(np.transpose(x, (0, 2, 3, 1)))
    c1_k = jnp.asarray(np.transpose(c1_w, (2, 3, 1, 0)))
    c2_k = jnp.asarray(np.transpose(c2_w, (2, 3, 1, 0)))
    h = fnn.relu(jax_conv(x_jx, c1_k, c1_b))
    h = fnn.relu(jax_conv(h, c2_k, c2_b))
    h = fnn.max_pool(h, (2, 2), strides=(2, 2))
    # Flatten in the same order as PyTorch's view: (B, C, H, W) flattened. We
    # need to convert NHWC back to NCHW before flatten to match PyTorch's
    # raster order.
    h_nchw = jnp.transpose(h, (0, 3, 1, 2))
    h = h_nchw.reshape(h_nchw.shape[0], -1)
    h = fnn.relu(h @ jnp.asarray(f1_w.T) + jnp.asarray(f1_b))
    out_jx = np.asarray(h @ jnp.asarray(f2_w.T) + jnp.asarray(f2_b))

    assert_close(out_pt, out_jx, atol=2e-4, name="cnn_forward")
    print("[m3] PASS")


def jax_conv(x, kernel, bias):
    import jax.lax as lax
    out = lax.conv_general_dilated(
        x, kernel, window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    return out + bias


if __name__ == "__main__":
    main()
