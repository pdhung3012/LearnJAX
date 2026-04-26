"""Sinusoidal Positional Embeddings (Vaswani et al. 2017).

Source: TorchLeet llm/Sinusoidal-Positional-Embedding/sinusoidal-q7.ipynb.
"""
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Buffer (not trainable, but moves with the module).
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch_size, seq_len, ...) — we only use seq_len.
        return self.pe[:x.shape[1], :].unsqueeze(0)  # (1, seq_len, d_model)


if __name__ == "__main__":
    torch.manual_seed(42)
    max_seq_len, d_model = 100, 64
    pos_emb = SinusoidalPositionalEmbedding(max_seq_len=max_seq_len, d_model=d_model)
    seq_len = 50
    # Dummy input shaped (1, 50, d_model) so shape[1] == 50.
    dummy = torch.zeros(1, seq_len, d_model)
    out = pos_emb(dummy)
    print("shape:", out.shape)  # (1, 50, 64)
    assert out.shape == (1, 50, 64)
