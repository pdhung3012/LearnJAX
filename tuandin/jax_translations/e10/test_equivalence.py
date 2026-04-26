"""e10 equivalence test: scaled-dot-product attention with shared Q, K, V."""
import sys
from pathlib import Path
import numpy as np
import torch
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close

sys.path.insert(0, str(Path(__file__).parent))
from pytorch_code import scaled_dot_product_attention as sdpa_pt
from jax_code import scaled_dot_product_attention as sdpa_jax


def main():
    rng = np.random.default_rng(0)
    B, S, D = 2, 5, 16
    q = rng.standard_normal((B, S, D)).astype(np.float32)
    k = rng.standard_normal((B, S, D)).astype(np.float32)
    v = rng.standard_normal((B, S, D)).astype(np.float32)

    out_pt, attn_pt = sdpa_pt(torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v))
    out_jx, attn_jx = sdpa_jax(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v))
    assert_close(out_pt.numpy(),  np.asarray(out_jx),  atol=1e-5, name="output")
    assert_close(attn_pt.numpy(), np.asarray(attn_jx), atol=1e-5, name="attention_weights")

    # With a causal mask.
    mask = np.tril(np.ones((S, S), dtype=np.float32))
    mask_pt = torch.from_numpy(mask)
    mask_jx = jnp.asarray(mask)
    out_pt2, _ = sdpa_pt(torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v), mask=mask_pt)
    out_jx2, _ = sdpa_jax(jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), mask=mask_jx)
    assert_close(out_pt2.numpy(), np.asarray(out_jx2), atol=1e-5, name="output_with_causal_mask")
    print("[e10] PASS")


if __name__ == "__main__":
    main()
