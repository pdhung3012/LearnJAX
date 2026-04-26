"""e2 equivalence test.

The PyTorch script writes data to CSV and uses a DataLoader with shuffle=True
(per-epoch RNG-driven). End-to-end equivalence isn't meaningful. We verify that:
  - both scripts execute end-to-end (smoke test)
  - the core nn.Linear(1,1) forward pass matches with shared weights
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close, torch_linear_to_jax, smoke_test

HERE = Path(__file__).parent


def main():
    rng = np.random.default_rng(1)
    W = rng.uniform(-1, 1, (1, 1)).astype(np.float32)
    b = rng.uniform(-1, 1, (1,)).astype(np.float32)
    X = rng.uniform(0, 10, (32, 1)).astype(np.float32)

    pt = nn.Linear(1, 1)
    with torch.no_grad():
        pt.weight.copy_(torch.from_numpy(W))
        pt.bias.copy_(torch.from_numpy(b))
    out_pt = pt(torch.from_numpy(X)).detach().numpy()

    Wj, bj = torch_linear_to_jax(pt.weight, pt.bias)
    out_jax = np.asarray(jnp.asarray(X) @ Wj + bj)
    assert_close(out_pt, out_jax, atol=1e-6, name="forward")

    print("[e2] forward equivalence: PASS")
    print("[e2] running smoke test (this trains for 1000 epochs in each)...")
    smoke_test(HERE, "e2", timeout=120)


if __name__ == "__main__":
    main()
