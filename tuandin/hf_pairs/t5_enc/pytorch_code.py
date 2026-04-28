"""Hand-written PyTorch T5 encoder, architecturally identical to transformers.T5EncoderModel.

T5-specific patterns to translate:
1. T5LayerNorm = RMSNorm (no mean centering, no bias, eps=1e-6).
2. No QK scaling — T5 does NOT divide attention scores by sqrt(d_head).
3. No biases on any of q, k, v, o, wi, wo.
4. Relative position bias instead of absolute position embeddings: a learned
   `relative_attention_bias` embedding indexed by a non-trivial bucketing
   function, ADDED to attention scores. Computed in the FIRST encoder layer
   only and reused across all layers.
5. ReLU activation in the FFN (T5 base; T5-v1.1 uses gated GELU).
6. Final encoder layer norm.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class T5Config:
    vocab_size: int = 100
    d_model: int = 64
    d_ff: int = 128
    num_layers: int = 2
    num_heads: int = 4
    d_kv: int = 16
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_epsilon: float = 1e-6


class T5LayerNorm(nn.Module):
    """RMSNorm: rescale by RMS, no mean centering, no bias."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.variance_epsilon)


def _relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
    """T5 relative-position bucketing — bidirectional for the encoder."""
    ret = torch.zeros_like(relative_position)
    n = relative_position
    if bidirectional:
        num_buckets //= 2
        ret = ret + (n > 0).long() * num_buckets
        n = torch.abs(n)
    else:
        n = -torch.minimum(n, torch.zeros_like(n))
    max_exact = num_buckets // 2
    is_small = n < max_exact
    n_clipped = torch.maximum(n, torch.tensor(max_exact, device=n.device)).float()
    val_if_large = max_exact + (
        torch.log(n_clipped / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    val_if_large = torch.minimum(
        val_if_large, torch.tensor(num_buckets - 1, device=val_if_large.device)
    )
    return ret + torch.where(is_small, n, val_if_large)


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_kv
        self.has_relative_attention_bias = has_relative_attention_bias
        self.q = nn.Linear(config.d_model, config.num_heads * config.d_kv, bias=False)
        self.k = nn.Linear(config.d_model, config.num_heads * config.d_kv, bias=False)
        self.v = nn.Linear(config.d_model, config.num_heads * config.d_kv, bias=False)
        self.o = nn.Linear(config.num_heads * config.d_kv, config.d_model, bias=False)
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                config.relative_attention_num_buckets, config.num_heads
            )
        self.config = config

    def compute_bias(self, qlen: int, klen: int, device):
        ctx = torch.arange(qlen, device=device).unsqueeze(1)
        mem = torch.arange(klen, device=device).unsqueeze(0)
        rel = mem - ctx
        bucket = _relative_position_bucket(
            rel, bidirectional=True,
            num_buckets=self.config.relative_attention_num_buckets,
            max_distance=self.config.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(bucket)        # (qlen, klen, num_heads)
        return values.permute(2, 0, 1).unsqueeze(0)          # (1, num_heads, qlen, klen)

    def forward(self, x, attention_mask, position_bias=None):
        B, S, _ = x.shape
        Q = self.q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1))   # NOTE: T5 does NOT scale.
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(S, S, x.device)
        if position_bias is not None:
            scores = scores + position_bias
        if attention_mask is not None:
            m = attention_mask[:, None, None, :].float()
            scores = scores + (1.0 - m) * torch.finfo(scores.dtype).min
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o(out), position_bias


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, x, attention_mask, position_bias=None):
        normed = self.layer_norm(x)
        attn_out, position_bias = self.SelfAttention(normed, attention_mask, position_bias)
        return x + attn_out, position_bias


class T5DenseReluDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x):
        return self.wo(F.relu(self.wi(x)))


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.DenseReluDense = T5DenseReluDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, x):
        return x + self.DenseReluDense(self.layer_norm(x))


class T5Block(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList([
            T5LayerSelfAttention(config, has_relative_attention_bias),
            T5LayerFF(config),
        ])

    def forward(self, x, attention_mask, position_bias=None):
        x, position_bias = self.layer[0](x, attention_mask, position_bias)
        x = self.layer[1](x)
        return x, position_bias


class T5Stack(nn.Module):
    def __init__(self, config: T5Config, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens
        # Only the first block carries the relative_attention_bias table;
        # subsequent blocks reuse the position_bias produced by block 0.
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=(i == 0))
             for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, attention_mask):
        x = self.embed_tokens(input_ids)
        position_bias = None
        for block in self.block:
            x, position_bias = block(x, attention_mask, position_bias)
        return self.final_layer_norm(x)


class T5EncoderModel(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, self.shared)

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids, attention_mask)


def build_pt_model(seed: int = 0) -> T5EncoderModel:
    config = T5Config()
    torch.manual_seed(seed)
    model = T5EncoderModel(config)
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
