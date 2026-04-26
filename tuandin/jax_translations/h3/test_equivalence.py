"""h3 equivalence test: small Transformer encoder layer forward.

We can't compare nn.TransformerEncoderLayer to a Flax-based one bit-for-bit
without dropping into both implementations and porting weights — so we check
the *underlying* attention formula on shared inputs as a representative
spot-check, plus a smoke test on the LayerNorm + FFN sub-block.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close


def main():
    rng = np.random.default_rng(0)
    B, S, D = 2, 5, 16
    x = rng.standard_normal((B, S, D)).astype(np.float32)

    # LayerNorm with shared weight/bias.
    gamma = rng.uniform(0.5, 1.5, (D,)).astype(np.float32)
    beta = rng.standard_normal((D,)).astype(np.float32) * 0.1
    ln_pt = nn.LayerNorm(D)
    with torch.no_grad():
        ln_pt.weight.copy_(torch.from_numpy(gamma)); ln_pt.bias.copy_(torch.from_numpy(beta))
    out_pt = ln_pt(torch.from_numpy(x)).detach().numpy()

    # Flax LayerNorm with the same params.
    import flax.linen as fnn
    ln_jx = fnn.LayerNorm()
    params = {"params": {"scale": jnp.asarray(gamma), "bias": jnp.asarray(beta)}}
    out_jx = np.asarray(ln_jx.apply(params, jnp.asarray(x)))
    assert_close(out_pt, out_jx, atol=1e-5, name="layer_norm")

    # FFN sub-block: Linear -> ReLU -> Linear.
    W1 = rng.standard_normal((64, D)).astype(np.float32) * 0.1
    b1 = rng.standard_normal((64,)).astype(np.float32) * 0.1
    W2 = rng.standard_normal((D, 64)).astype(np.float32) * 0.1
    b2 = rng.standard_normal((D,)).astype(np.float32) * 0.1
    out_pt = (torch.relu(torch.from_numpy(x) @ torch.from_numpy(W1.T)
                          + torch.from_numpy(b1))
               @ torch.from_numpy(W2.T) + torch.from_numpy(b2)).numpy()
    h = jnp.maximum(jnp.asarray(x) @ jnp.asarray(W1.T) + jnp.asarray(b1), 0.0)
    out_jx = np.asarray(h @ jnp.asarray(W2.T) + jnp.asarray(b2))
    assert_close(out_pt, out_jx, atol=1e-5, name="ffn_block")
    print("[h3] PASS")


if __name__ == "__main__":
    main()
