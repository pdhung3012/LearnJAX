"""Multi-Head Attention from scratch — must match nn.MultiheadAttention output.

Source: TorchLeet llm/Multi-Head-Attention/multi-head-attention-q5.ipynb.

Note: the notebook's reference function builds fresh nn.Linear layers inside
each call, so its weights are random and the assertion vs. nn.MultiheadAttention
fails. We preserve that bug here so the file matches the source. The proper
behavior — keeping projections in a stateful module — is what the JAX
translation demonstrates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def multi_head_attention(q, k, v, num_heads, d_model, mask=None, device="cpu"):
    assert d_model % num_heads == 0
    d_head = d_model // num_heads
    batch_size, seq_len, _ = q.shape

    # NOTE: random projections — the notebook's bug, preserved verbatim.
    Q_w = nn.Linear(d_model, d_model, bias=False).to(device)
    K_w = nn.Linear(d_model, d_model, bias=False).to(device)
    V_w = nn.Linear(d_model, d_model, bias=False).to(device)
    W_out = nn.Linear(d_model, d_model, bias=False).to(device)

    Q = Q_w(q).view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)
    K = K_w(k).view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)
    V = V_w(v).view(batch_size, seq_len, num_heads, d_head).transpose(1, 2)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_head ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, V)  # (B, H, S, d_head)
    out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    return W_out(out)


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size, seq_len, d_model, num_heads = 3, 4, 8, 2
    q = torch.rand(batch_size, seq_len, d_model)
    k = torch.rand(batch_size, seq_len, d_model)
    v = torch.rand(batch_size, seq_len, d_model)
    output_custom = multi_head_attention(q, k, v, num_heads, d_model)
    print("custom shape:", output_custom.shape)

    multihead_attn = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=num_heads, bias=False, batch_first=True
    )
    output_ref, _ = multihead_attn(q, k, v)
    print("torch shape :", output_ref.shape)
