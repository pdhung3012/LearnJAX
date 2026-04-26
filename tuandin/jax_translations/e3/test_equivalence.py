"""e3 equivalence test: linear + custom activation tanh(x) + x."""
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
    W = rng.uniform(-1, 1, (1, 1)).astype(np.float32)
    b = rng.uniform(-1, 1, (1,)).astype(np.float32)
    X = rng.uniform(0, 10, (100, 1)).astype(np.float32)

    pt = nn.Linear(1, 1)
    with torch.no_grad():
        pt.weight.copy_(torch.from_numpy(W))
        pt.bias.copy_(torch.from_numpy(b))
    z_pt = pt(torch.from_numpy(X))
    out_pt = (torch.tanh(z_pt) + z_pt).detach().numpy()

    Wj, bj = torch_linear_to_jax(pt.weight, pt.bias)
    z_jax = jnp.asarray(X) @ Wj + bj
    out_jax = np.asarray(jnp.tanh(z_jax) + z_jax)
    assert_close(out_pt, out_jax, atol=1e-6, name="custom_activation_forward")
    print("[e3] PASS")


if __name__ == "__main__":
    main()
