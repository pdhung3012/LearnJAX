"""e4 equivalence test: Huber loss formula matches PyTorch's hand-rolled version."""
import sys
from pathlib import Path
import numpy as np
import torch
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close


def huber_pt(y_pred, y_true, delta=1.0):
    err = torch.abs(y_pred - y_true)
    return torch.where(err <= delta, 0.5 * err ** 2,
                       delta * (err - 0.5 * delta)).mean()


def huber_jax(y_pred, y_true, delta=1.0):
    err = jnp.abs(y_pred - y_true)
    return jnp.mean(jnp.where(err <= delta, 0.5 * err ** 2,
                              delta * (err - 0.5 * delta)))


def main():
    rng = np.random.default_rng(0)
    pred = rng.standard_normal((100, 1)).astype(np.float32) * 3
    target = rng.standard_normal((100, 1)).astype(np.float32) * 3

    for delta in [0.5, 1.0, 2.0]:
        l_pt = huber_pt(torch.from_numpy(pred), torch.from_numpy(target), delta).item()
        l_jax = float(huber_jax(jnp.asarray(pred), jnp.asarray(target), delta))
        assert_close(l_pt, l_jax, atol=1e-5, name=f"huber_loss(delta={delta})")
    print("[e4] PASS")


if __name__ == "__main__":
    main()
