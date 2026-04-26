"""m7 equivalence test: SimpleNN forward (28*28 -> 128 -> 10) with shared weights."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close, torch_linear_to_jax


def main():
    rng = np.random.default_rng(0)
    fc1_w = rng.standard_normal((128, 28 * 28)).astype(np.float32) * 0.05
    fc1_b = rng.standard_normal((128,)).astype(np.float32) * 0.05
    fc2_w = rng.standard_normal((10, 128)).astype(np.float32) * 0.05
    fc2_b = rng.standard_normal((10,)).astype(np.float32) * 0.05
    x = rng.standard_normal((4, 1, 28, 28)).astype(np.float32)

    fc1 = nn.Linear(28 * 28, 128); fc2 = nn.Linear(128, 10)
    with torch.no_grad():
        fc1.weight.copy_(torch.from_numpy(fc1_w)); fc1.bias.copy_(torch.from_numpy(fc1_b))
        fc2.weight.copy_(torch.from_numpy(fc2_w)); fc2.bias.copy_(torch.from_numpy(fc2_b))
    flat_pt = torch.from_numpy(x).view(-1, 28 * 28)
    out_pt = fc2(torch.relu(fc1(flat_pt))).detach().numpy()

    W1, b1 = torch_linear_to_jax(fc1.weight, fc1.bias)
    W2, b2 = torch_linear_to_jax(fc2.weight, fc2.bias)
    flat_jx = jnp.asarray(x).reshape(-1, 28 * 28)
    h = jnp.maximum(flat_jx @ W1 + b1, 0.0)
    out_jx = np.asarray(h @ W2 + b2)
    assert_close(out_pt, out_jx, atol=1e-4, name="simple_nn_forward")
    print("[m7] PASS")


if __name__ == "__main__":
    main()
