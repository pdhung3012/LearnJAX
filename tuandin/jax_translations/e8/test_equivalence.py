"""e8 equivalence test: RMSNorm forward pass with shared scale parameter."""
import sys
from pathlib import Path
import numpy as np
import torch
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close

# Import RMSNorm classes from the case files.
sys.path.insert(0, str(Path(__file__).parent))
from pytorch_code import RMSNorm as RMSNorm_pt
from jax_code import RMSNorm as RMSNorm_jax
import flax.linen as nn


def main():
    rng = np.random.default_rng(0)
    dim = 8
    x_np = rng.standard_normal((4, dim)).astype(np.float32) * 2
    scale_np = rng.uniform(0.5, 1.5, (dim,)).astype(np.float32)

    pt = RMSNorm_pt(dim=dim, eps=1e-8)
    with torch.no_grad():
        pt.scale.copy_(torch.from_numpy(scale_np))
    out_pt = pt(torch.from_numpy(x_np)).detach().numpy()

    jax_model = RMSNorm_jax(dim=dim, eps=1e-8)
    params = {"params": {"scale": jnp.asarray(scale_np)}}
    out_jax = np.asarray(jax_model.apply(params, jnp.asarray(x_np)))

    assert_close(out_pt, out_jax, atol=1e-6, name="rmsnorm_forward")
    print("[e8] PASS")


if __name__ == "__main__":
    main()
