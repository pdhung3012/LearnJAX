"""e6 equivalence test: linear regression + TensorBoard logging.

The script's main effect is writing a TensorBoard event file. We test:
- forward equivalence with shared weights
- loss equivalence
"""
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
    y = (3 * X + 5 + rng.standard_normal((100, 1))).astype(np.float32)

    pt = nn.Linear(1, 1)
    with torch.no_grad():
        pt.weight.copy_(torch.from_numpy(W)); pt.bias.copy_(torch.from_numpy(b))
    out_pt = pt(torch.from_numpy(X)).detach().numpy()
    loss_pt = nn.functional.mse_loss(pt(torch.from_numpy(X)), torch.from_numpy(y)).item()

    Wj, bj = torch_linear_to_jax(pt.weight, pt.bias)
    out_jax = np.asarray(jnp.asarray(X) @ Wj + bj)
    loss_jax = float(jnp.mean((jnp.asarray(X) @ Wj + bj - jnp.asarray(y)) ** 2))

    assert_close(out_pt, out_jax, atol=1e-6, name="forward")
    assert_close(loss_pt, loss_jax, atol=1e-5, name="mse_loss")
    print("[e6] PASS")


if __name__ == "__main__":
    main()
