"""JAX translation of m10: Rotary Positional Embeddings (RoPE).

Faithful to PyTorch:
- inv_freq = 1/base^(2i/dim) for i in 0,1,...,dim/2-1.
- For each position p, freqs_p = p * inv_freq, then emb = concat([freqs, freqs])
  along the last axis, and cos_cached/sin_cached are emb.cos()/emb.sin()
  reshaped to (seq_len, 1, 1, dim) for broadcasting against (B, S, H, D).
- rotate_half splits the last axis into two halves and produces [-x2, x1].
- apply_rotary_pos_emb returns (q * cos + rotate_half(q) * sin,
                                k * cos + rotate_half(k) * sin).

Speed: pure elementwise + matmul-free. JAX jit gives a tight kernel; ≈ on par
with PyTorch eager on small inputs, faster on larger sequences when fused into
the surrounding attention computation.
"""
import jax
import jax.numpy as jnp


def make_rotary(dim, seq_len, base=10000.0):
    """Precompute (cos, sin) tables shaped (1, seq_len, 1, dim) so they
    broadcast against tensors of shape (batch, seq_len, n_heads, dim)."""
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.einsum("i,j->ij", t, inv_freq)               # (seq_len, dim/2)
    emb = jnp.concatenate([freqs, freqs], axis=-1)           # (seq_len, dim)
    cos = jnp.cos(emb)[None, :, None, :]                     # (1, seq_len, 1, dim)
    sin = jnp.sin(emb)[None, :, None, :]
    return cos, sin


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    batch, seq_len, n_heads, head_dim = 2, 8, 4, 16

    kq, kk = jax.random.split(key)
    q = jax.random.normal(kq, (batch, seq_len, n_heads, head_dim))
    k = jax.random.normal(kk, (batch, seq_len, n_heads, head_dim))

    cos, sin = make_rotary(head_dim, seq_len)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print("q shape:", q_rot.shape, "k shape:", k_rot.shape)
    assert q_rot.shape == q.shape and k_rot.shape == k.shape
