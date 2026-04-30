"""Hand-written PyTorch ViT, architecturally identical to transformers.ViTModel.

Architecture:
- Patch embed: Conv2d(3, hidden, kH=patch, kW=patch, stride=patch).
- Prepend a learned [CLS] token; add learned absolute position embeddings.
- N x ViTLayer (pre-norm):
    layernorm_before -> SelfAttention -> +residual
    layernorm_after  -> FFN(GELU)     -> +residual
- Final LayerNorm on the full sequence.

Quirks:
- LayerNorm eps = 1e-12 (ViTConfig default).
- Activation: exact GELU (erf-based).
- The submodule names match HuggingFace's ViTModel exactly so the saved
  state_dict can be loaded into transformers.ViTModel verbatim.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ViTConfig:
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 128
    image_size: int = 32
    patch_size: int = 8
    num_channels: int = 3
    layer_norm_eps: float = 1e-12


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels, config.hidden_size,
            kernel_size=config.patch_size, stride=config.patch_size,
        )

    def forward(self, pixel_values):
        # pixel_values: (B, C, H, W) -> (B, hidden, gh, gw) -> (B, num_patches, hidden)
        x = self.projection(pixel_values)
        return x.flatten(2).transpose(1, 2)


class ViTEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.hidden_size)
        )

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        patches = self.patch_embeddings(pixel_values)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1)
        return x + self.position_embeddings


class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key   = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        B, S, H = x.shape
        Q = self.query(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, H)
        return out


class ViTSelfOutput(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        return self.dense(x)


class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, x):
        return self.output(self.attention(x))


class ViTIntermediate(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, x):
        return F.gelu(self.dense(x))


class ViTOutput(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.dense(x)


class ViTLayer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after  = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        # Pre-norm self-attention.
        x = x + self.attention(self.layernorm_before(x))
        # Pre-norm FFN.
        x = x + self.output(self.intermediate(self.layernorm_after(x)))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x


class ViTModel(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        x = self.embeddings(pixel_values)
        x = self.encoder(x)
        return self.layernorm(x)


def build_pt_model(seed: int = 0) -> ViTModel:
    config = ViTConfig()
    torch.manual_seed(seed)
    model = ViTModel(config)
    model.eval()
    return model


def main():
    model = build_pt_model()
    rng = torch.Generator().manual_seed(42)
    pixel_values = torch.rand((1, 3, 32, 32), generator=rng) * 2 - 1
    with torch.no_grad():
        out = model(pixel_values)
    print("last_hidden_state shape:", tuple(out.shape))
    print("checksum:", float(out.sum()))


if __name__ == "__main__":
    main()
