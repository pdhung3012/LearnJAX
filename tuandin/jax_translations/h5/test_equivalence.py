"""h5 equivalence test: per-step CE loss accumulation matches across
frameworks for a single decoder timestep.

Tests the cross-entropy formula used in the seq2seq teacher-forcing loop.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import jax.numpy as jnp
import optax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close


def main():
    rng = np.random.default_rng(0)
    B, V = 8, 20
    logits = rng.standard_normal((B, V)).astype(np.float32) * 2
    labels = rng.integers(0, V, (B,)).astype(np.int64)

    # PyTorch: CrossEntropyLoss expects raw logits.
    ce_pt = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels)).item()
    # In the seq2seq script the per-step losses are *summed* via `+=`, so the
    # mean reduction itself is what the optimizer differentiates. Here we test
    # the mean-reduction equivalence.
    ce_jx = float(optax.softmax_cross_entropy_with_integer_labels(
        jnp.asarray(logits), jnp.asarray(labels.astype(np.int32))).mean())
    assert_close(ce_pt, ce_jx, atol=1e-5, name="cross_entropy")
    print("[h5] PASS")


if __name__ == "__main__":
    main()
