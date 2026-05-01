"""Hand-written PyTorch GPT-2, architecturally identical to transformers.GPT2Model.

We avoid `from transformers import GPT2Model` so the cheap LLM has the full
architecture in front of it. State_dict layout matches HF GPT-2's exactly so
weights are interchangeable; freeze_fixtures.py verifies bit-for-bit
equivalence with the library version.

Architecture (decoder-only, pre-norm):
- wte (token embedding) + wpe (position embedding).
- N x GPT2Block (pre-norm):
    ln_1 -> Attention -> +residual
    ln_2 -> MLP       -> +residual
- ln_f (final layer norm).

Quirks worth knowing:
- GPT-2 uses `Conv1D` (not nn.Linear) for c_attn / c_proj / c_fc / c_proj.
  Conv1D stores weight as (in, out) — opposite of nn.Linear's (out, in) —
  and computes `x @ weight + bias`. We reimplement Conv1D below.
- c_attn is a fused (Q, K, V) projection of width 3 * hidden_size; we split.
- Activation is `gelu_new` (tanh approximation), not exact GELU.
- Causal mask is mandatory; combined with the (B, S) padding mask.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 100
    n_positions: int = 32
    n_embd: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_inner: int = 128
    layer_norm_epsilon: float = 1e-5


class Conv1D(nn.Module):
    """HuggingFace's GPT-2 Conv1D: weight shape (in, out); forward is x @ W + b."""

    def __init__(self, nx: int, nf: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return x @ self.weight + self.bias


def _gelu_new(x):
    """tanh approximation of GELU (PyTorch's NewGELUActivation)."""
    return 0.5 * x * (
        1.0
        + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
    )


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = Conv1D(config.n_embd, 3 * config.n_embd)
        self.c_proj = Conv1D(config.n_embd, config.n_embd)

    def forward(self, x, attention_mask):
        B, S, H = x.shape
        qkv = self.c_attn(x)                                 # (B, S, 3H)
        Q, K, V = qkv.split(H, dim=-1)
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device))
        scores = torch.where(causal[None, None], scores,
                             torch.tensor(torch.finfo(scores.dtype).min, device=x.device))
        if attention_mask is not None:
            m = attention_mask[:, None, None, :].float()
            scores = scores + (1.0 - m) * torch.finfo(scores.dtype).min
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, H)
        return self.c_proj(out)


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = Conv1D(config.n_embd, config.n_inner)
        self.c_proj = Conv1D(config.n_inner, config.n_embd)

    def forward(self, x):
        return self.c_proj(_gelu_new(self.c_fc(x)))


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, x, attention_mask):
        x = x + self.attn(self.ln_1(x), attention_mask)
        return x + self.mlp(self.ln_2(x))


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, attention_mask):
        S = input_ids.shape[1]
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(position_ids)
        for block in self.h:
            x = block(x, attention_mask)
        return self.ln_f(x)


def build_pt_model(seed: int = 0) -> GPT2Model:
    config = GPT2Config()
    torch.manual_seed(seed)
    model = GPT2Model(config)
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
