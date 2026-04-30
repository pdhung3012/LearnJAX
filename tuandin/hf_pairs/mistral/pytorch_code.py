"""Hand-written PyTorch Mistral, architecturally identical to transformers.MistralModel.

Mistral is the modern-LLM canonical case in this suite — it stresses every
architectural pattern at once:

1. **RMSNorm** instead of LayerNorm (no mean centering, no bias).
2. **Rotary positional embeddings (RoPE)** applied to Q and K (no absolute
   position embedding table).
3. **Grouped-Query Attention (GQA)**: more Q heads than KV heads. KV heads
   are repeated to match Q heads before the dot-product.
4. **SwiGLU MLP**: `down(silu(gate(x)) * up(x))` with NO biases.
5. No biases on the attention projections either.

We keep the model tiny (hidden=64, 2 layers, 4 Q heads, 2 KV heads, ff=128)
so the case runs in seconds on CPU. State_dict layout matches HF Mistral's.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MistralConfig:
    vocab_size: int = 100
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2          # 2:1 GQA (Q heads : KV heads)
    intermediate_size: int = 128
    max_position_embeddings: int = 32
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0


# --- Primitives ---------------------------------------------------------------


class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.variance_epsilon)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(q, k, cos, sin):
    # q, k: (B, num_heads, S, head_dim);  cos/sin: (1, 1, S, head_dim)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _build_rope_cos_sin(seq_len, head_dim, base, device, dtype):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)               # (S, D/2)
    emb = torch.cat([angles, angles], dim=-1)         # (S, D)
    return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]   # (1, 1, S, D)


def _repeat_kv(x, n_rep):
    """(B, H_kv, S, D) -> (B, H_kv * n_rep, S, D)."""
    if n_rep == 1:
        return x
    B, H_kv, S, D = x.shape
    return x.unsqueeze(2).expand(B, H_kv, n_rep, S, D).reshape(B, H_kv * n_rep, S, D)


# --- Attention + MLP + decoder layer -----------------------------------------


class MistralAttention(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.rope_theta = config.rope_theta

    def forward(self, x, attention_mask):
        B, S, _ = x.shape
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = _build_rope_cos_sin(S, self.head_dim, self.rope_theta, x.device, x.dtype)
        Q, K = _apply_rope(Q, K, cos, sin)

        n_rep = self.num_heads // self.num_kv_heads
        K = _repeat_kv(K, n_rep)
        V = _repeat_kv(V, n_rep)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Causal mask (Mistral is decoder-only).
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=x.device))
        scores = torch.where(causal[None, None], scores,
                             torch.tensor(torch.finfo(scores.dtype).min, device=x.device))
        if attention_mask is not None:
            m = attention_mask[:, None, None, :].float()
            scores = scores + (1.0 - m) * torch.finfo(scores.dtype).min
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class MistralMLP(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.self_attn = MistralAttention(config)
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, attention_mask):
        x = x + self.self_attn(self.input_layernorm(x), attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MistralModel(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MistralDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.norm(x)


def build_pt_model(seed: int = 0) -> MistralModel:
    config = MistralConfig()
    torch.manual_seed(seed)
    model = MistralModel(config)
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
