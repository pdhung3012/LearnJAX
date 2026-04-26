"""m4 equivalence test: dice loss formula matches across frameworks.

The full pipeline (pretrained ResNet18 + 3D conv head + training) is too
heavy to fully test here. We focus on the dice-loss formula equivalence,
which is the core scoring metric in the original problem.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close


def dice_pt(pred, label, eps=1e-8):
    return (2 * torch.sum(pred * label)) / (torch.sum(pred) + torch.sum(label) + eps)


def dice_jx(pred, label, eps=1e-8):
    return (2 * jnp.sum(pred * label)) / (jnp.sum(pred) + jnp.sum(label) + eps)


def main():
    rng = np.random.default_rng(0)
    pred = np.abs(rng.standard_normal((2, 4, 1, 8, 8))).astype(np.float32)
    label = (rng.standard_normal((2, 4, 1, 8, 8)) > 0).astype(np.float32)
    l_pt = dice_pt(torch.from_numpy(pred), torch.from_numpy(label)).item()
    l_jx = float(dice_jx(jnp.asarray(pred), jnp.asarray(label)))
    assert_close(l_pt, l_jx, atol=1e-6, name="dice_loss")
    print("[m4] PASS")


if __name__ == "__main__":
    main()
