"""Hand-written PyTorch ALBERT, architecturally identical to transformers.AlbertModel.

ALBERT-specific patterns to translate:
1. Factorized embeddings: word/position/type embeddings live at embedding_size,
   then `encoder.embedding_hidden_mapping_in` projects up to hidden_size.
2. Cross-layer parameter sharing: a single block's weights are reused across
   all `num_hidden_layers` (state_dict has only one set of layer weights at
   `encoder.albert_layer_groups.0.albert_layers.0.*`).
3. Activation defaults to gelu_new (tanh approximation), unlike BERT's
   exact GELU.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AlbertConfig:
    vocab_size: int = 100
    embedding_size: int = 48
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 128
    max_position_embeddings: int = 32
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12


def _gelu_new(x):
    return 0.5 * x * (
        1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))
    )


class AlbertEmbeddings(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

    def forward(self, input_ids, token_type_ids):
        S = input_ids.shape[1]
        position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.LayerNorm(x)


class AlbertAttention(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key   = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, H)
        return self.LayerNorm(x + self.dense(out))


class AlbertLayer(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, attention_mask):
        x = self.attention(x, attention_mask)
        h = self.ffn_output(_gelu_new(self.ffn(x)))
        return self.full_layer_layer_norm(x + h)


class AlbertLayerGroup(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()
        # inner_group_num=1 by default; just one layer per group.
        self.albert_layers = nn.ModuleList([AlbertLayer(config)])

    def forward(self, x, attention_mask):
        for layer in self.albert_layers:
            x = layer(x, attention_mask)
        return x


class AlbertTransformer(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        # num_hidden_groups=1 by default; single group reused N times.
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config)])
        self.num_hidden_layers = config.num_hidden_layers

    def forward(self, x, attention_mask):
        x = self.embedding_hidden_mapping_in(x)
        for _ in range(self.num_hidden_layers):
            x = self.albert_layer_groups[0](x, attention_mask)
        return x


class AlbertModel(nn.Module):
    def __init__(self, config: AlbertConfig):
        super().__init__()
        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = self.embeddings(input_ids, token_type_ids)
        return self.encoder(x, attention_mask)


def build_pt_model(seed: int = 0) -> AlbertModel:
    config = AlbertConfig()
    torch.manual_seed(seed)
    model = AlbertModel(config)
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
