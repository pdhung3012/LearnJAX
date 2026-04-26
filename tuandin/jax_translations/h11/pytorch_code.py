"""Grouped-Query Attention (GQA) — used in Llama-2/3, Mistral, etc. Lets you
have more query heads than KV heads, reducing KV-cache size without changing
the query-head capacity.

Source: TorchLeet llm/Grouped-Query-Attention/grouped-query-attention.ipynb.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def grouped_query_attention(q, k, v, num_query_heads, num_query_groups, d_model,
                             mask=None, device="cpu"):
    """
    Args:
        q, k, v: (B, S, d_model)
        num_query_heads: number of Q heads.
        num_query_groups: number of K/V heads (must divide num_query_heads).
    """
    assert d_model % num_query_heads == 0
    assert num_query_heads % num_query_groups == 0
    d_head = d_model // num_query_heads
    B, S, _ = q.shape

    Q_w = nn.Linear(d_model, num_query_heads * d_head,  bias=False).to(device)
    K_w = nn.Linear(d_model, num_query_groups * d_head, bias=False).to(device)
    V_w = nn.Linear(d_model, num_query_groups * d_head, bias=False).to(device)
    W_out = nn.Linear(d_model, d_model, bias=False).to(device)

    Q = Q_w(q).view(B, S, num_query_heads,  d_head).transpose(1, 2)  # (B, H, S, d_head)
    K = K_w(k).view(B, S, num_query_groups, d_head).transpose(1, 2)  # (B, G, S, d_head)
    V = V_w(v).view(B, S, num_query_groups, d_head).transpose(1, 2)

    repeat_factor = num_query_heads // num_query_groups
    K = K.repeat_interleave(repeat_factor, dim=1)  # (B, H, S, d_head)
    V = V.repeat_interleave(repeat_factor, dim=1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)  # (B, H, S, d_head)
    out = out.transpose(1, 2).contiguous().view(B, S, num_query_heads * d_head)
    return W_out(out)


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size, seq_len, d_model = 3, 4, 64
    num_query_heads, num_query_groups = 8, 2  # 8 query heads, 2 KV heads

    q = torch.rand(batch_size, seq_len, d_model)
    k = torch.rand(batch_size, seq_len, d_model)
    v = torch.rand(batch_size, seq_len, d_model)
    out = grouped_query_attention(q, k, v, num_query_heads, num_query_groups, d_model)
    print("output shape:", out.shape)  # (3, 4, 64)
    assert out.shape == (batch_size, seq_len, d_model)
