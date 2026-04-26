"""e5 equivalence test: 2->10->1 MLP forward pass with shared weights."""
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
    fc1_W = rng.standard_normal((10, 2)).astype(np.float32) * 0.3
    fc1_b = rng.standard_normal((10,)).astype(np.float32) * 0.1
    fc2_W = rng.standard_normal((1, 10)).astype(np.float32) * 0.3
    fc2_b = rng.standard_normal((1,)).astype(np.float32) * 0.1
    X = rng.uniform(0, 10, (100, 2)).astype(np.float32)

    fc1 = nn.Linear(2, 10); fc2 = nn.Linear(10, 1)
    with torch.no_grad():
        fc1.weight.copy_(torch.from_numpy(fc1_W)); fc1.bias.copy_(torch.from_numpy(fc1_b))
        fc2.weight.copy_(torch.from_numpy(fc2_W)); fc2.bias.copy_(torch.from_numpy(fc2_b))
    out_pt = fc2(torch.relu(fc1(torch.from_numpy(X)))).detach().numpy()

    W1j, b1j = torch_linear_to_jax(fc1.weight, fc1.bias)
    W2j, b2j = torch_linear_to_jax(fc2.weight, fc2.bias)
    h = jnp.maximum(jnp.asarray(X) @ W1j + b1j, 0.0)
    out_jax = np.asarray(h @ W2j + b2j)
    assert_close(out_pt, out_jax, atol=1e-5, name="mlp_forward")
    print("[e5] PASS")


if __name__ == "__main__":
    main()
