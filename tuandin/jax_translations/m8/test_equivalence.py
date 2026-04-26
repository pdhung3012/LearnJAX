"""m8 equivalence test: encoder forward (Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool)
with shared weights produces identical features.

We restrict to the encoder because the decoder uses ConvTranspose2d, where
PyTorch's `output_padding` parameter has no direct match in Flax (Flax
ConvTranspose with padding='SAME' lines up only when input size is even).
The encoder by itself is a sound test of the conv path equivalence.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import jax.lax as lax
import jax.numpy as jnp
import flax.linen as fnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close


def main():
    rng = np.random.default_rng(0)
    c1_w = rng.standard_normal((32, 1, 3, 3)).astype(np.float32) * 0.1
    c1_b = rng.standard_normal((32,)).astype(np.float32) * 0.05
    c2_w = rng.standard_normal((64, 32, 3, 3)).astype(np.float32) * 0.1
    c2_b = rng.standard_normal((64,)).astype(np.float32) * 0.05
    x = rng.standard_normal((2, 1, 28, 28)).astype(np.float32)

    pt_c1 = nn.Conv2d(1, 32, 3, padding=1); pt_c2 = nn.Conv2d(32, 64, 3, padding=1)
    pt_pool = nn.MaxPool2d(2, 2)
    with torch.no_grad():
        pt_c1.weight.copy_(torch.from_numpy(c1_w)); pt_c1.bias.copy_(torch.from_numpy(c1_b))
        pt_c2.weight.copy_(torch.from_numpy(c2_w)); pt_c2.bias.copy_(torch.from_numpy(c2_b))
    h = pt_pool(torch.relu(pt_c1(torch.from_numpy(x))))
    enc_pt = pt_pool(torch.relu(pt_c2(h))).detach().numpy()

    x_jx = jnp.asarray(np.transpose(x, (0, 2, 3, 1)))
    c1_k = jnp.asarray(np.transpose(c1_w, (2, 3, 1, 0)))
    c2_k = jnp.asarray(np.transpose(c2_w, (2, 3, 1, 0)))
    def conv(x, k, b):
        return lax.conv_general_dilated(
            x, k, (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        ) + b
    h = fnn.max_pool(fnn.relu(conv(x_jx, c1_k, c1_b)), (2, 2), strides=(2, 2))
    enc_jx = fnn.max_pool(fnn.relu(conv(h, c2_k, c2_b)), (2, 2), strides=(2, 2))
    enc_jx = np.transpose(np.asarray(enc_jx), (0, 3, 1, 2))

    assert_close(enc_pt, enc_jx, atol=1e-4, name="encoder_forward")
    print("[m8] PASS")


if __name__ == "__main__":
    main()
