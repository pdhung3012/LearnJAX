"""SmolLM-135M from scratch — full Llama-style decoder-only LM with RoPE,
Grouped-Query Attention, RMSNorm, and SwiGLU MLP.

Source: TorchLeet llm/SmolLM/smollm-q12.ipynb.

Architecture:
- 30 LlamaDecoder layers
- hidden_size=576, intermediate_size=1536, num_heads=9, kv_heads=3
- rope_theta=10000.0, RMSNorm eps=1e-5
- vocab_size=49152, lm_head shares embedding weights
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states, n_rep):
    b, h_kv, s, d = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(b, h_kv, n_rep, s, d).reshape(b, h_kv * n_rep, s, d)


class RotaryEmbedder(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        self.freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    @torch.no_grad()
    def forward(self, x):
        # x: (B, H, S, D). We use shape[-2] = S.
        pos = torch.arange(x.shape[-2], dtype=torch.float32)
        angles = torch.einsum("p,f->pf", pos, self.freq).unsqueeze(0)
        emb = torch.cat((angles, angles), dim=-1)
        return emb.cos(), emb.sin()


class RopeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.kv_heads = config.kv_heads
        self.rope_theta = 10000.0

        self.W_query = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.W_key   = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.W_value = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.W_output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedder(base=self.rope_theta, dim=self.head_dim)

    def forward(self, hidden_states, attention_mask=None):
        b, q, _ = hidden_states.size()
        Q = self.W_query(hidden_states).view(b, q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(hidden_states).view(b, q, self.kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(hidden_states).view(b, q, self.kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(Q)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        n_rep = self.num_heads // self.kv_heads
        K = repeat_kv(K, n_rep)
        V = repeat_kv(V, n_rep)

        attn_weights = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(b, q, -1)
        return self.W_output(attn_output)


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.W_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.W_up   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.W_down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.W_down(self.act_fn(self.W_gate(x)) * self.W_up(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, h):
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * h


class LlamaDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = RopeAttention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.pre_attn_rmsnorm = RMSNorm(config.hidden_size, eps=1e-5)
        self.pre_mlp_rmsnorm  = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states, attention_mask):
        residual = hidden_states
        hidden_states = self.pre_attn_rmsnorm(hidden_states)
        causal_mask = torch.triu(
            torch.full((attention_mask.shape[-1], attention_mask.shape[-1]),
                       fill_value=float("-inf")), diagonal=1
        )
        hidden_states = self.self_attn(hidden_states, attention_mask=causal_mask)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.pre_mlp_rmsnorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class smolModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoder(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=1e-5)

    def forward(self, input_ids, attention_mask):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask)
        return self.norm(h)


class smolLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = smolModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight  # tied

    def forward(self, input_ids, attention_mask):
        h = self.model(input_ids, attention_mask=attention_mask)
        return {"logits": self.lm_head(h).float()}


class smolConfig:
    vocab_size = 49152
    hidden_size = 576
    intermediate_size = 1536
    num_hidden_layers = 30
    num_heads = 9
    kv_heads = 3


if __name__ == "__main__":
    torch.manual_seed(0)
    config = smolConfig()
    model = smolLM(config)
    model.eval()
    # Forward on a 4-token prompt; outputs logits of shape (1, 4, 49152).
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids, attention_mask)
    print("logits shape:", out["logits"].shape)
    assert out["logits"].shape == (1, 4, config.vocab_size)
