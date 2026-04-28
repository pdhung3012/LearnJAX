"""Hand-written PyTorch RoBERTa, architecturally identical to transformers.RobertaModel.

Differences from BERT:
1. Position IDs start at `pad_token_id + 1` (= 2 by default), not 0. The HF
   convention computes them as:
       position_ids = (cumsum(attention_mask, dim=1) * attention_mask) + pad_id
   so the first non-pad token gets position pad_id+1, the second pad_id+2, etc.
2. token_type_embeddings has only one entry (type_vocab_size=1).
3. Otherwise the encoder layer is the BERT layer verbatim.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RobertaConfig:
    vocab_size: int = 100
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 128
    max_position_embeddings: int = 32
    type_vocab_size: int = 1
    pad_token_id: int = 1
    layer_norm_eps: float = 1e-12


def _make_position_ids(input_ids, attention_mask, pad_id):
    """RoBERTa-specific: positions for non-pad tokens start at pad_id + 1."""
    mask = attention_mask.long()
    incremental = torch.cumsum(mask, dim=1) * mask
    return incremental + pad_id


class RobertaEmbeddings(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pad_id = config.pad_token_id

    def forward(self, input_ids, attention_mask, token_type_ids):
        position_ids = _make_position_ids(input_ids, attention_mask, self.pad_id)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.LayerNorm(x)


class RobertaSelfAttention(nn.Module):
    def __init__(self, config: RobertaConfig):
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
        out = torch.matmul(attn, V)
        return out.transpose(1, 2).contiguous().view(B, S, H)


class RobertaSelfOutput(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        return self.LayerNorm(input_tensor + self.dense(hidden_states))


class RobertaAttention(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.self = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(self, x, attention_mask):
        return self.output(self.self(x, attention_mask), x)


class RobertaIntermediate(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, x):
        return F.gelu(self.dense(x))


class RobertaOutput(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        return self.LayerNorm(input_tensor + self.dense(hidden_states))


class RobertaLayer(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(self, x, attention_mask):
        x = self.attention(x, attention_mask)
        return self.output(self.intermediate(x), x)


class RobertaEncoder(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x, attention_mask):
        for layer in self.layer:
            x = layer(x, attention_mask)
        return x


class RobertaModel(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = self.embeddings(input_ids, attention_mask, token_type_ids)
        return self.encoder(x, attention_mask)


def build_pt_model(seed: int = 0) -> RobertaModel:
    config = RobertaConfig()
    torch.manual_seed(seed)
    model = RobertaModel(config)
    model.eval()
    return model


def main():
    model = build_pt_model()
    input_ids = torch.tensor([[0, 5, 9, 13, 17, 21, 25, 2]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids, attention_mask)
    print("last_hidden_state shape:", tuple(out.shape))
    print("checksum:", float(out.sum()))


if __name__ == "__main__":
    main()
