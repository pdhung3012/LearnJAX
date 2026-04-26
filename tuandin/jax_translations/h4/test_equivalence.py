"""h4 equivalence test: BCE loss formula equivalence.

The full GAN training loop is RNG-sensitive (different latents per step), so
we test the BCE loss formula and the LeakyReLU/Tanh activations used in the
discriminator/generator.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close


def bce_pt(p, t, eps=1e-7):
    p = torch.clamp(p, eps, 1 - eps)
    return -(t * torch.log(p) + (1 - t) * torch.log(1 - p)).mean()


def bce_jx(p, t, eps=1e-7):
    p = jnp.clip(p, eps, 1 - eps)
    return -jnp.mean(t * jnp.log(p) + (1 - t) * jnp.log(1 - p))


def main():
    rng = np.random.default_rng(0)
    p = np.clip(rng.uniform(0.0, 1.0, (32, 1)).astype(np.float32), 0.01, 0.99)
    t = rng.integers(0, 2, (32, 1)).astype(np.float32)
    l_pt = bce_pt(torch.from_numpy(p), torch.from_numpy(t)).item()
    l_jx = float(bce_jx(jnp.asarray(p), jnp.asarray(t)))
    assert_close(l_pt, l_jx, atol=1e-5, name="bce_loss")

    # LeakyReLU(0.2).
    z = rng.standard_normal((10,)).astype(np.float32)
    a_pt = torch.nn.functional.leaky_relu(torch.from_numpy(z), 0.2).numpy()
    import jax
    a_jx = np.asarray(jax.nn.leaky_relu(jnp.asarray(z), negative_slope=0.2))
    assert_close(a_pt, a_jx, atol=1e-7, name="leaky_relu_0.2")

    # Tanh.
    a_pt = torch.tanh(torch.from_numpy(z)).numpy()
    a_jx = np.asarray(jnp.tanh(jnp.asarray(z)))
    assert_close(a_pt, a_jx, atol=1e-7, name="tanh")
    print("[h4] PASS")


if __name__ == "__main__":
    main()
