"""Hand-written PyTorch DistilBERT, architecturally identical to transformers.DistilBertModel.

DistilBERT vs BERT:
- No token_type_embeddings (only position + word).
- Linear names use the *_lin suffix (q_lin, k_lin, v_lin, out_lin).
- Per-layer LayerNorms are sa_layer_norm (after attention) and output_layer_norm
  (after FFN), both post-norm.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistilBertConfig:
    vocab_size: int = 100
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    hidden_dim: int = 128
    max_position_embeddings: int = 32


class Embeddings(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)

    def forward(self, input_ids):
        S = input_ids.shape[1]
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        return self.LayerNorm(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.num_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.q_lin = nn.Linear(config.dim, config.dim)
        self.k_lin = nn.Linear(config.dim, config.dim)
        self.v_lin = nn.Linear(config.dim, config.dim)
        self.out_lin = nn.Linear(config.dim, config.dim)

    def forward(self, x, attention_mask):
        B, S, H = x.shape
        Q = self.q_lin(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_lin(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_lin(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            m = attention_mask[:, None, None, :].float()
            scores = scores + (1.0 - m) * torch.finfo(scores.dtype).min
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, H)
        return self.out_lin(out)


class FFN(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.lin1 = nn.Linear(config.dim, config.hidden_dim)
        self.lin2 = nn.Linear(config.hidden_dim, config.dim)

    def forward(self, x):
        return self.lin2(F.gelu(self.lin1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)

    def forward(self, x, attention_mask):
        x = self.sa_layer_norm(x + self.attention(x, attention_mask))
        return self.output_layer_norm(x + self.ffn(x))


class Transformer(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

    def forward(self, x, attention_mask):
        for layer in self.layer:
            x = layer(x, attention_mask)
        return x


class DistilBertModel(nn.Module):
    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.transformer = Transformer(config)

    def forward(self, input_ids, attention_mask):
        return self.transformer(self.embeddings(input_ids), attention_mask)


def build_pt_model(seed: int = 0) -> DistilBertModel:
    config = DistilBertConfig()
    torch.manual_seed(seed)
    model = DistilBertModel(config)
    model.eval()
    return model


def main():
    model = build_pt_model()
    input_ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids, attention_mask)
    print("last_hidden_state shape:", tuple(out.shape))
    print("checksum:", float(out.sum()))


if __name__ == "__main__":
    main()
