"""JAX translation of e10: Scaled Dot-Product Attention from scratch.

Faithful to PyTorch:
- Same formula: softmax(Q K^T / sqrt(d_k)) V, with optional mask.
- Compares against `jax.nn.dot_product_attention` (the JAX equivalent of
  torch.nn.functional.scaled_dot_product_attention).

Speed: jit'd version on small inputs is microsecond-level; ≈ on par with
PyTorch.
"""
import jax
import jax.numpy as jnp


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Args:
        q: (..., seq_len_q, d_k)
        k: (..., seq_len_k, d_k)
        v: (..., seq_len_k, d_v)
        mask: (..., seq_len_q, seq_len_k) — 0 means "mask out".
    Returns:
        output:       (..., seq_len_q, d_v)
        attn_weights: (..., seq_len_q, seq_len_k)
    """
    d_k = q.shape[-1]
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / jnp.sqrt(jnp.float32(d_k))
    if mask is not None:
        scores = jnp.where(mask == 0, -jnp.inf, scores)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(attn_weights, v)
    return output, attn_weights


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (1, 3, 3))
    k = jax.random.normal(kk, (1, 3, 3))
    v = jax.random.normal(kv, (1, 3, 3))

    out_custom, _ = scaled_dot_product_attention(q, k, v)

    # Compare against JAX's first-party SDPA (note: JAX expects q/k/v shaped
    # (B, T, H, D); we add a singleton head axis).
    q4 = q[:, :, None, :]
    k4 = k[:, :, None, :]
    v4 = v[:, :, None, :]
    out_ref = jax.nn.dot_product_attention(q4, k4, v4)
    out_ref = out_ref.squeeze(2)

    print("custom:", out_custom)
    print("jax   :", out_ref)
    assert jnp.allclose(out_custom, out_ref, atol=1e-5, rtol=1e-5)
    print("passed")
