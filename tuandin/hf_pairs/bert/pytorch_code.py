"""Hand-written PyTorch BERT, architecturally identical to transformers.BertModel.

This file is the *source* the cheap LLM is asked to translate. We deliberately
avoid `from transformers import BertModel` so the LLM has the full architecture
in front of it (no need to recall what BertModel does from memory). The
state_dict layout matches HF's exactly so that weights are interchangeable;
freeze_fixtures.py verifies bit-for-bit equivalence with the HF library version.

Architecture:
- BertEmbeddings: word + position + token_type, LayerNorm.
- N x BertLayer (post-norm):
    BertSelfAttention -> BertSelfOutput.dense -> +residual + LayerNorm
    BertIntermediate.dense -> GELU -> BertOutput.dense -> +residual + LayerNorm
- (No pooler — we return last_hidden_state, matching expected.npz.)

Implementation notes:
- LayerNorm eps = 1e-12 (BERT default).
- Activation: exact GELU (erf-based).
- Attention mask: (B, S) is broadcast to (B, 1, 1, S); 0 -> -inf, 1 -> 0.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BertConfig:
    vocab_size: int = 100
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 128
    max_position_embeddings: int = 32
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12


class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids, token_type_ids):
        S = input_ids.shape[1]
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.LayerNorm(x)


class BertSelfAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key   = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, attention_mask):
        B, S, H = x.shape
        Q = self.query(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            m = attention_mask[:, None, None, :].float()
            scores = scores + (1.0 - m) * torch.finfo(scores.dtype).min
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)            # (B, h, S, d_h)
        return out.transpose(1, 2).contiguous().view(B, S, H)


class BertSelfOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        return self.LayerNorm(input_tensor + self.dense(hidden_states))


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, x, attention_mask):
        return self.output(self.self(x, attention_mask), x)


class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, x):
        return F.gelu(self.dense(x))   # exact GELU (erf-based), not gelu_new


class BertOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        return self.LayerNorm(input_tensor + self.dense(hidden_states))


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, x, attention_mask):
        x = self.attention(x, attention_mask)
        return self.output(self.intermediate(x), x)


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, attention_mask):
        for layer in self.layer:
            x = layer(x, attention_mask)
        return x


class BertModel(nn.Module):
    """Hand-written equivalent of transformers.BertModel (without pooler)."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = self.embeddings(input_ids, token_type_ids)
        return self.encoder(x, attention_mask)


def build_pt_model(seed: int = 0) -> BertModel:
    config = BertConfig()
    torch.manual_seed(seed)
    model = BertModel(config)
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
