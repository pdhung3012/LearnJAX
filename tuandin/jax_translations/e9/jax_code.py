"""JAX translation of e9: Sinusoidal Positional Embeddings.

Faithful to PyTorch:
- Same formula: even indices get sin(pos * w), odd indices get cos(pos * w),
  with w_k = exp(-k * log(10000) / d_model) for k in 0,2,4,...,d_model-2.
- The `pe` buffer is precomputed once and reused (in JAX we just precompute
  it as a regular array — there's no notion of a non-trainable buffer; it's
  baked in as a constant inside the module).

Speed: pure data construction; trivial.
"""
import math
import jax
import jax.numpy as jnp


def make_sinusoidal_pe(max_seq_len: int, d_model: int):
    position = jnp.arange(max_seq_len, dtype=jnp.float32)[:, None]      # (L, 1)
    div_term = jnp.exp(jnp.arange(0, d_model, 2, dtype=jnp.float32)
                       * (-math.log(10000.0) / d_model))                # (d_model/2,)
    pe = jnp.zeros((max_seq_len, d_model), dtype=jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe


def sinusoidal_pe_forward(pe, x):
    # x: (batch_size, seq_len, ...) — we slice pe to seq_len.
    return pe[:x.shape[1], :][None, ...]  # (1, seq_len, d_model)


if __name__ == "__main__":
    max_seq_len, d_model = 100, 64
    pe = make_sinusoidal_pe(max_seq_len, d_model)
    seq_len = 50
    dummy = jnp.zeros((1, seq_len, d_model))
    out = sinusoidal_pe_forward(pe, dummy)
    print("shape:", out.shape)  # (1, 50, 64)
    assert out.shape == (1, 50, 64)
