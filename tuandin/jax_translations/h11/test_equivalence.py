"""h11 equivalence test: Grouped-Query Attention forward with shared weights."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close

sys.path.insert(0, str(Path(__file__).parent))
from jax_code import GroupedQueryAttention


def gqa_forward_pt(q, k, v, Q_w, K_w, V_w, W_out,
                    num_query_heads, num_query_groups, d_model):
    d_head = d_model // num_query_heads
    B, S, _ = q.shape
    Q = (q @ Q_w.T).view(B, S, num_query_heads, d_head).transpose(1, 2)
    K = (k @ K_w.T).view(B, S, num_query_groups, d_head).transpose(1, 2)
    V = (v @ V_w.T).view(B, S, num_query_groups, d_head).transpose(1, 2)
    n_rep = num_query_heads // num_query_groups
    K = K.repeat_interleave(n_rep, dim=1)
    V = V.repeat_interleave(n_rep, dim=1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, d_model)
    return out @ W_out.T


def main():
    rng = np.random.default_rng(0)
    B, S, d_model = 2, 4, 64
    num_query_heads, num_query_groups = 8, 2
    d_head = d_model // num_query_heads

    q = rng.standard_normal((B, S, d_model)).astype(np.float32)
    k = rng.standard_normal((B, S, d_model)).astype(np.float32)
    v = rng.standard_normal((B, S, d_model)).astype(np.float32)
    Q_w = rng.standard_normal((num_query_heads * d_head, d_model)).astype(np.float32) * 0.1
    K_w = rng.standard_normal((num_query_groups * d_head, d_model)).astype(np.float32) * 0.1
    V_w = rng.standard_normal((num_query_groups * d_head, d_model)).astype(np.float32) * 0.1
    W_out = rng.standard_normal((d_model, d_model)).astype(np.float32) * 0.1

    out_pt = gqa_forward_pt(
        torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v),
        torch.from_numpy(Q_w), torch.from_numpy(K_w), torch.from_numpy(V_w),
        torch.from_numpy(W_out),
        num_query_heads, num_query_groups, d_model,
    ).numpy()

    model = GroupedQueryAttention(d_model=d_model,
                                  num_query_heads=num_query_heads,
                                  num_query_groups=num_query_groups)
    rng_key = jax.random.PRNGKey(0)
    _ = model.init(rng_key, jnp.asarray(q), jnp.asarray(k), jnp.asarray(v))
    new_params = {"params": {
        "Q_w":   {"kernel": jnp.asarray(Q_w.T)},
        "K_w":   {"kernel": jnp.asarray(K_w.T)},
        "V_w":   {"kernel": jnp.asarray(V_w.T)},
        "W_out": {"kernel": jnp.asarray(W_out.T)},
    }}
    out_jx = np.asarray(model.apply(new_params,
                                     jnp.asarray(q), jnp.asarray(k), jnp.asarray(v)))
    assert_close(out_pt, out_jx, atol=1e-5, name="gqa_forward")
    print("[h11] PASS")


if __name__ == "__main__":
    main()
