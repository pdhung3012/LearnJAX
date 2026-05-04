"""Hand-written PyTorch BART encoder, architecturally identical to
transformers.BartModel's encoder.

BART encoder has a few quirks worth knowing about:
1. Learned absolute position embeddings, but with a +2 offset: position
   ids are `[0, 1, ..., S-1] + 2` and the embedding table has size
   `max_position_embeddings + 2`. This is HF's
   `BartLearnedPositionalEmbedding` (offset=2).
2. Embedding scaling: inputs_embeds are multiplied by `embed_scale` —
   sqrt(d_model) when scale_embedding=True, otherwise 1.0. The HF default
   is False, so we set scale=1.0.
3. Pre-norm-then-attention-then-add (post-norm style with layer norms
   applied AFTER the residual). HF's BartEncoderLayer uses
   self_attn_layer_norm AFTER the self-attn residual and final_layer_norm
   AFTER the FFN residual.
4. layernorm_embedding is applied to embeddings (BART-specific; mBART
   skips it).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BartConfig:
    vocab_size: int = 100
    d_model: int = 64
    encoder_layers: int = 2
    encoder_attention_heads: int = 4
    encoder_ffn_dim: int = 128
    max_position_embeddings: int = 32
    pad_token_id: int = 1
    scale_embedding: bool = False


class BartLearnedPositionalEmbedding(nn.Embedding):
    """Position embeddings indexed by `position_ids + offset`. HF uses
    offset=2 (the first 2 indices are reserved)."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Allocate +offset slots up front (HF convention).
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids):
        S = input_ids.shape[1]
        positions = torch.arange(S, device=input_ids.device, dtype=torch.long)
        return super().forward(positions + self.offset)


class BartAttention(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.num_heads = config.encoder_attention_heads
        self.head_dim = config.d_model // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)

    def forward(self, x, attention_mask):
        B, S, H = x.shape
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2) * self.scaling
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # Note: scaling is folded into Q above (HF style); no /sqrt here.
        scores = torch.matmul(Q, K.transpose(-2, -1))
        if attention_mask is not None:
            m = attention_mask[:, None, None, :].float()
            scores = scores + (1.0 - m) * torch.finfo(scores.dtype).min
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, H)
        return self.out_proj(out)


class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.self_attn = BartAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, x, attention_mask):
        # Self-attention block.
        residual = x
        x = self.self_attn(x, attention_mask)
        x = self.self_attn_layer_norm(residual + x)
        # FFN block.
        residual = x
        x = self.fc2(F.gelu(self.fc1(x)))
        x = self.final_layer_norm(residual + x)
        return x


class BartEncoder(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.config = config
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model
        )
        self.layers = nn.ModuleList(
            [BartEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def forward(self, input_ids, attention_mask):
        x = self.embed_tokens(input_ids) * self.embed_scale
        x = x + self.embed_positions(input_ids)
        x = self.layernorm_embedding(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


def build_pt_model(seed: int = 0) -> BartEncoder:
    config = BartConfig()
    torch.manual_seed(seed)
    model = BartEncoder(config)
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
