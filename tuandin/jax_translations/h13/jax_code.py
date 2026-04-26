"""JAX translation of h13: Flash Attention v2 forward.

The PyTorch reference is a Triton kernel (GPU only). JAX has two equivalents:

1. **Vanilla blocked / streaming-softmax implementation** (this file's
   `flash_attention_forward`). Runs on any backend (CPU/GPU/TPU). It walks the
   K/V dimension in tiles and uses the same online-softmax trick, just
   expressed in regular JAX ops via `jax.lax.scan`. This produces *the same
   numerical algorithm* as Flash-Attention but doesn't fuse into a single
   kernel — XLA may still fuse it, especially on TPU.

2. **Pallas kernel** (commented stub at the bottom). Pallas is JAX's analogue
   of Triton: a low-level kernel DSL that lets you tile, load blocks, and emit
   a single fused kernel. On GPU this produces a kernel structurally
   equivalent to the Triton one. We include the stub but don't run it here
   because Pallas requires GPU/TPU.

Speed:
- The blocked implementation in JAX outperforms naive O(S^2) attention for
  long sequences by reducing memory traffic, but on CPU it won't beat XLA's
  fused naive softmax+matmul for short sequences. Real Flash-Attention wins
  show up on GPU with sequence lengths in the thousands.
- A Pallas/Triton-level kernel on GPU is what gives the canonical
  Flash-Attention speedup; we don't run it on this CPU.
"""
import math
import jax
import jax.numpy as jnp


def flash_attention_forward(Q, K, V, block_size_k: int = 16):
    """Streaming-softmax attention. Q/K/V: (B, N, D).
    Returns (O, L) where L is the row logsumexp of the pre-softmax scores."""
    B, N_q, D = Q.shape
    N_k = K.shape[1]
    scale = 1.0 / math.sqrt(D)
    assert N_k % block_size_k == 0, "N_k must be divisible by block_size_k"

    def per_batch(q, k, v):
        # q, k, v: (N, D) for a single batch element. Walk K/V in tiles.
        n_tiles = N_k // block_size_k

        def step(carry, j):
            out, l, prev_max = carry
            k_tile = jax.lax.dynamic_slice_in_dim(k, j * block_size_k, block_size_k, axis=0)  # (Bk, D)
            v_tile = jax.lax.dynamic_slice_in_dim(v, j * block_size_k, block_size_k, axis=0)
            s = q @ k_tile.T * scale                          # (N, Bk)
            curr_max = jnp.maximum(prev_max, jnp.max(s, axis=1))
            p = jnp.exp(s - curr_max[:, None])                # (N, Bk)
            alpha = jnp.exp(prev_max - curr_max)              # (N,)
            out = out * alpha[:, None] + p @ v_tile           # (N, D)
            l = l * alpha + jnp.sum(p, axis=1)                # (N,)
            return (out, l, curr_max), None

        init_out = jnp.zeros((N_q, D), dtype=jnp.float32)
        init_l = jnp.ones((N_q,), dtype=jnp.float32)
        init_max = jnp.full((N_q,), -jnp.inf, dtype=jnp.float32)
        (out, l, prev_max), _ = jax.lax.scan(step, (init_out, init_l, init_max), jnp.arange(n_tiles))
        out = out / l[:, None]
        log_l = prev_max + jnp.log(l)
        return out, log_l

    return jax.vmap(per_batch)(Q, K, V)


# --- Optional: Pallas kernel (GPU/TPU only). Uncomment to use. ---------------
#
# from jax.experimental import pallas as pl
#
# def flash_attention_pallas(Q, K, V, block_q=16, block_k=16):
#     """Pallas kernel mirroring the Triton reference. Requires GPU/TPU."""
#     B, N_q, D = Q.shape
#     N_k = K.shape[1]
#     scale = 1.0 / math.sqrt(D)
#
#     def kernel(q_ref, k_ref, v_ref, o_ref, l_ref):
#         q = q_ref[...]  # (block_q, D)
#         out = jnp.zeros((block_q, D), dtype=jnp.float32)
#         l = jnp.ones((block_q,), dtype=jnp.float32)
#         prev_max = jnp.full((block_q,), -jnp.inf, dtype=jnp.float32)
#         for j in range(N_k // block_k):
#             k = pl.load(k_ref, (pl.ds(j * block_k, block_k), slice(None)))
#             v = pl.load(v_ref, (pl.ds(j * block_k, block_k), slice(None)))
#             s = jnp.dot(q, k.T) * scale
#             curr_max = jnp.maximum(prev_max, jnp.max(s, axis=1))
#             p = jnp.exp(s - curr_max[:, None])
#             alpha = jnp.exp(prev_max - curr_max)
#             out = out * alpha[:, None] + jnp.dot(p, v)
#             l = l * alpha + jnp.sum(p, axis=1)
#             prev_max = curr_max
#         o_ref[...] = out / l[:, None]
#         l_ref[...] = prev_max + jnp.log(l)
#
#     return pl.pallas_call(
#         kernel,
#         grid=(N_q // block_q, B),
#         out_shape=[
#             jax.ShapeDtypeStruct(Q.shape, Q.dtype),
#             jax.ShapeDtypeStruct((B, N_q), jnp.float32),
#         ],
#     )(Q, K, V)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    B, N_q, N_k, D = 1, 64, 128, 256
    kQ, kK, kV = jax.random.split(key, 3)
    Q = jax.random.normal(kQ, (B, N_q, D), dtype=jnp.float32)
    K = jax.random.normal(kK, (B, N_k, D), dtype=jnp.float32)
    V = jax.random.normal(kV, (B, N_k, D), dtype=jnp.float32)

    O, L = jax.jit(flash_attention_forward)(Q, K, V)

    # Reference using vanilla attention.
    scale = 1.0 / math.sqrt(D)
    scores = (Q @ jnp.swapaxes(K, -2, -1)) * scale
    O_ref = jax.nn.softmax(scores, axis=-1) @ V
    L_ref = jax.scipy.special.logsumexp(scores, axis=-1)

    print("O max abs diff:", float(jnp.max(jnp.abs(O - O_ref))))
    print("L max abs diff:", float(jnp.max(jnp.abs(L - L_ref))))
    assert jnp.allclose(O, O_ref, atol=1e-3, rtol=1e-3)
    assert jnp.allclose(L, L_ref, atol=1e-3, rtol=1e-3)
    print("passed")
