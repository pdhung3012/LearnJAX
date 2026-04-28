"""Scaled Dot-Product Attention from scratch — must match
torch.nn.functional.scaled_dot_product_attention.

Source: TorchLeet llm/Implement-Attention-from-Scratch/attention-q4.ipynb.
"""
import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Args:
        q: (..., seq_len_q, d_k)
        k: (..., seq_len_k, d_k)
        v: (..., seq_len_k, d_v)
        mask: (..., seq_len_q, seq_len_k) — 0 means "mask out".
    Returns:
        output: (..., seq_len_q, d_v)
        attn_weights: (..., seq_len_q, seq_len_k)
    """
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 3
    dim = 3
    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)
    v = torch.randn(batch_size, seq_len, dim)

    output_custom, _ = scaled_dot_product_attention(q, k, v)
    output_ref = F.scaled_dot_product_attention(q, k, v)
    print("custom:", output_custom)
    print("torch :", output_ref)
    assert torch.allclose(output_custom, output_ref, atol=1e-6, rtol=1e-5)
    print("passed")
