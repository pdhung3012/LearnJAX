"""Rotary Positional Embeddings (RoPE) — used in Llama, GPT-NeoX, and most
modern LLMs.

Source: TorchLeet llm/Rotary-Positional-Embedding/rope-q8.ipynb. The notebook's
test harness has a typo (`apply_rotary_pos_emb(positions)` with one argument);
we replace it with a proper end-to-end demo.
"""
import torch
import torch.nn as nn


class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


if __name__ == "__main__":
    torch.manual_seed(42)
    # Notebook's Rotary caches cos/sin as (S, 1, 1, D), which only broadcasts
    # cleanly against tensors shaped (S, B, H, D) — the GPT-NeoX layout. Use
    # that layout here so the test runs.
    seq_len, batch, n_heads, head_dim = 8, 2, 4, 16
    q = torch.randn(seq_len, batch, n_heads, head_dim)
    k = torch.randn(seq_len, batch, n_heads, head_dim)

    rotary = Rotary(dim=head_dim)
    cos, sin = rotary(q, seq_dim=0)  # cos/sin: (seq_len, 1, 1, head_dim)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print("q shape:", q_rot.shape, "k shape:", k_rot.shape)
    assert q_rot.shape == q.shape and k_rot.shape == k.shape
