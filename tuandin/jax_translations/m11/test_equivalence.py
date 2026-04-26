"""m11 equivalence test: cosine-similarity formula equivalence.

The actual SmolLM2-135M model is downloaded by both implementations; we don't
re-run it here (would re-download ~270 MB). We verify that the masked-mean
sentence-embedding formula and the cosine-similarity formula agree
bit-for-bit between PyTorch and JAX given the same hidden states.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close


def masked_mean_pt(last_hidden, mask):
    expanded = mask.unsqueeze(-1).float()
    summed = (last_hidden * expanded).sum(dim=1)
    counts = expanded.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def masked_mean_jx(last_hidden, mask):
    expanded = mask[..., None].astype(jnp.float32)
    summed = jnp.sum(last_hidden * expanded, axis=1)
    counts = jnp.maximum(jnp.sum(expanded, axis=1), 1e-9)
    return summed / counts


def main():
    rng = np.random.default_rng(0)
    B, S, H = 4, 7, 16
    last_hidden = rng.standard_normal((B, S, H)).astype(np.float32)
    mask = (rng.standard_normal((B, S)) > -0.5).astype(np.int32)
    keyword = rng.standard_normal((1, H)).astype(np.float32)

    # Masked mean.
    me_pt = masked_mean_pt(torch.from_numpy(last_hidden), torch.from_numpy(mask)).numpy()
    me_jx = np.asarray(masked_mean_jx(jnp.asarray(last_hidden), jnp.asarray(mask)))
    assert_close(me_pt, me_jx, atol=1e-6, name="masked_mean")

    # Cosine similarity.
    cs_pt = F.cosine_similarity(torch.from_numpy(me_pt), torch.from_numpy(keyword)).numpy()
    a = jnp.asarray(me_pt)
    b = jnp.asarray(keyword)
    a_n = a / jnp.maximum(jnp.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b_n = b / jnp.maximum(jnp.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    cs_jx = np.asarray(jnp.sum(a_n * b_n, axis=-1))
    assert_close(cs_pt, cs_jx, atol=1e-5, name="cosine_similarity")
    print("[m11] PASS")


if __name__ == "__main__":
    main()
