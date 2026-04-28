"""JAX translation of h11: Grouped-Query Attention.

Faithful to PyTorch:
- Same algorithm: Q has H heads, K/V have G groups (G | H). After projecting
  K and V, broadcast each group to H/G query heads (jnp.repeat axis=1) so that
  the final attention is computed per-query-head.

Speed: jit'd JAX is competitive with PyTorch. The repeat is a *cheap* expand
(no real copy if you broadcast at the matmul level), but for clarity we
materialize the repeat with `jnp.repeat`.
"""
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn


class GroupedQueryAttention(nn.Module):
    d_model: int
    num_query_heads: int
    num_query_groups: int  # must divide num_query_heads

    @nn.compact
    def __call__(self, q, k, v, mask=None):
        assert self.d_model % self.num_query_heads == 0
        assert self.num_query_heads % self.num_query_groups == 0
        d_head = self.d_model // self.num_query_heads
        B, S, _ = q.shape

        Q = nn.Dense(self.num_query_heads * d_head,  use_bias=False, name="Q_w")(q)
        K = nn.Dense(self.num_query_groups * d_head, use_bias=False, name="K_w")(k)
        V = nn.Dense(self.num_query_groups * d_head, use_bias=False, name="V_w")(v)

        Q = Q.reshape(B, S, self.num_query_heads,  d_head).transpose(0, 2, 1, 3)
        K = K.reshape(B, S, self.num_query_groups, d_head).transpose(0, 2, 1, 3)
        V = V.reshape(B, S, self.num_query_groups, d_head).transpose(0, 2, 1, 3)

        repeat_factor = self.num_query_heads // self.num_query_groups
        K = jnp.repeat(K, repeat_factor, axis=1)  # (B, H, S, d_head)
        V = jnp.repeat(V, repeat_factor, axis=1)

        scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / jnp.sqrt(jnp.float32(d_head))
        if mask is not None:
            scores = jnp.where(mask == 0, -jnp.inf, scores)
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.matmul(attn, V)  # (B, H, S, d_head)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, self.num_query_heads * d_head)
        return nn.Dense(self.d_model, use_bias=False, name="W_out")(out)


# ---- Contract API used by test_equivalence.py ------------------------------
def compute(inputs):
    """Run h11's deterministic core: GQA forward pass with caller-provided weights.

    Args:
      inputs: dict with keys
        - "q", "k", "v" each shape (B, S, d_model)
        - "Q_w" shape (num_query_heads * d_head, d_model)  (PyTorch nn.Linear convention: out, in)
        - "K_w" shape (num_query_groups * d_head, d_model)
        - "V_w" shape (num_query_groups * d_head, d_model)
        - "W_out" shape (d_model, d_model)
        - "d_model", "num_query_heads", "num_query_groups" as 0-d int arrays.
    Returns:
      dict with key "output" shape (B, S, d_model).
    """
    d_model          = int(inputs["d_model"])
    num_query_heads  = int(inputs["num_query_heads"])
    num_query_groups = int(inputs["num_query_groups"])
    model = GroupedQueryAttention(
        d_model=d_model,
        num_query_heads=num_query_heads,
        num_query_groups=num_query_groups,
    )
    q = jnp.asarray(inputs["q"])
    k = jnp.asarray(inputs["k"])
    v = jnp.asarray(inputs["v"])
    # Initialize the module so we get the param-tree structure, then overwrite
    # the kernels with the caller-supplied weights (Flax kernel convention is
    # the transpose of PyTorch's nn.Linear weight).
    _ = model.init(jax.random.PRNGKey(0), q, k, v)  # discard random init
    params = {"params": {
        "Q_w":   {"kernel": jnp.asarray(inputs["Q_w"].T)},
        "K_w":   {"kernel": jnp.asarray(inputs["K_w"].T)},
        "V_w":   {"kernel": jnp.asarray(inputs["V_w"].T)},
        "W_out": {"kernel": jnp.asarray(inputs["W_out"].T)},
    }}
    out = model.apply(params, q, k, v)
    return {"output": np.asarray(out)}


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    batch_size, seq_len, d_model = 3, 4, 64
    num_query_heads, num_query_groups = 8, 2

    kq, kk, kv, ki = jax.random.split(key, 4)
    q = jax.random.uniform(kq, (batch_size, seq_len, d_model))
    k = jax.random.uniform(kk, (batch_size, seq_len, d_model))
    v = jax.random.uniform(kv, (batch_size, seq_len, d_model))

    model = GroupedQueryAttention(
        d_model=d_model,
        num_query_heads=num_query_heads,
        num_query_groups=num_query_groups,
    )
    params = model.init(ki, q, k, v)
    out = model.apply(params, q, k, v)
    print("output shape:", out.shape)  # (3, 4, 64)
    assert out.shape == (batch_size, seq_len, d_model)
