"""e1 equivalence test: nn.Linear(1, 1) forward pass with shared weights.

We can't compare end-to-end training because PyTorch and JAX use different
RNGs, so the trained weights differ. But we can verify that *given the same
weights and inputs*, both frameworks produce the same output (the algorithmic
core). MSELoss + SGD update step is also numerically checked.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close, torch_linear_to_jax


def main():
    rng = np.random.default_rng(0)
    W = rng.uniform(-1, 1, (1, 1)).astype(np.float32)
    b = rng.uniform(-1, 1, (1,)).astype(np.float32)
    X = rng.uniform(0, 10, (100, 1)).astype(np.float32)
    y = (2 * X + 3 + rng.standard_normal((100, 1)).astype(np.float32))

    # ----- Forward equivalence -----
    pt = nn.Linear(1, 1)
    with torch.no_grad():
        pt.weight.copy_(torch.from_numpy(W))
        pt.bias.copy_(torch.from_numpy(b))
    out_pt = pt(torch.from_numpy(X)).detach().numpy()

    Wj, bj = torch_linear_to_jax(pt.weight, pt.bias)
    out_jax = np.asarray(jnp.asarray(X) @ Wj + bj)
    assert_close(out_pt, out_jax, atol=1e-6, name="forward")

    # ----- MSE equivalence -----
    loss_pt = nn.functional.mse_loss(pt(torch.from_numpy(X)), torch.from_numpy(y)).item()
    loss_jax = float(jnp.mean((jnp.asarray(X) @ Wj + bj - jnp.asarray(y)) ** 2))
    assert_close(loss_pt, loss_jax, atol=1e-5, name="mse_loss")

    # ----- One SGD step equivalence -----
    pt2 = nn.Linear(1, 1)
    with torch.no_grad():
        pt2.weight.copy_(torch.from_numpy(W))
        pt2.bias.copy_(torch.from_numpy(b))
    opt = torch.optim.SGD(pt2.parameters(), lr=0.01)
    pred = pt2(torch.from_numpy(X))
    loss = nn.functional.mse_loss(pred, torch.from_numpy(y))
    loss.backward()
    opt.step()
    pt_W_after = pt2.weight.detach().numpy().copy()
    pt_b_after = pt2.bias.detach().numpy().copy()

    def loss_fn(params, X, y):
        return jnp.mean((X @ params["W"] + params["b"] - y) ** 2)

    params = {"W": Wj, "b": bj}
    grads = jax.grad(loss_fn)(params, jnp.asarray(X), jnp.asarray(y))
    new_W = params["W"] - 0.01 * grads["W"]
    new_b = params["b"] - 0.01 * grads["b"]
    # Compare: PyTorch's W is (out, in), JAX is (in, out).
    assert_close(pt_W_after, np.asarray(new_W).T, atol=1e-5, name="W_after_sgd")
    assert_close(pt_b_after, np.asarray(new_b),     atol=1e-5, name="b_after_sgd")
    print("[e1] PASS")


if __name__ == "__main__":
    main()
