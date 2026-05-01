"""Hand-written PyTorch Wav2Vec2, architecturally identical to transformers.Wav2Vec2Model.

Wav2Vec2 is the most distinctive case in the suite — it's a SPEECH model
with a 1-D convolutional feature extractor in front of a transformer
encoder. Patterns to translate:

1. **1-D conv feature extractor** — N stacked Conv1d layers (here 3) with
   strides that progressively downsample the raw audio waveform (input
   shape (B, T_audio); output shape (B, T_features, hidden) after a
   feature_projection LayerNorm + Linear).
2. **Per-layer LayerNorm on conv outputs** — applied along the channel
   dim (so we transpose, LayerNorm, transpose back).
3. **Positional convolutional embedding** — a single grouped Conv1d
   parametrized via `torch.nn.utils.parametrizations.weight_norm`. State
   dict stores this as `parametrizations.weight.original0` (magnitude)
   and `original1` (direction). Output is GELU-activated and trimmed to
   the input length when the kernel is even (kernel=16 here -> trim by 1).
4. **Standard transformer encoder** (post-norm) atop the positional
   embedding sum. Same as BERT structurally.

We use `apply_spec_augment=False` (no SpecAugment masking at inference)
and `feat_extract_norm='layer'` (avoids the Wav2Vec2-specific GroupNorm
that lives only on the first conv layer of the 'group' variant).
"""
import math
from dataclasses import dataclass
from dataclasses import field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Wav2Vec2Config:
    vocab_size: int = 100
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    intermediate_size: int = 128
    conv_dim: tuple = (32, 32, 32)
    conv_kernel: tuple = (5, 3, 3)
    conv_stride: tuple = (2, 2, 2)
    conv_bias: bool = False
    feat_extract_norm: str = "layer"
    num_conv_pos_embeddings: int = 16
    num_conv_pos_embedding_groups: int = 4
    layer_norm_eps: float = 1e-5
    feature_proj_layer_norm_eps: float = 1e-5
    apply_spec_augment: bool = False


# --- Feature extractor -------------------------------------------------------


class Wav2Vec2LayerNormConvLayer(nn.Module):
    def __init__(self, config: Wav2Vec2Config, layer_id: int):
        super().__init__()
        in_dim = 1 if layer_id == 0 else config.conv_dim[layer_id - 1]
        out_dim = config.conv_dim[layer_id]
        self.conv = nn.Conv1d(
            in_dim, out_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(-2, -1)
        x = self.layer_norm(x)
        x = x.transpose(-2, -1)
        return F.gelu(x)


class Wav2Vec2FeatureEncoder(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [Wav2Vec2LayerNormConvLayer(config, i) for i in range(len(config.conv_dim))]
        )

    def forward(self, x):
        # x: (B, T_audio) -> add channel dim -> (B, 1, T_audio).
        x = x.unsqueeze(1)
        for layer in self.conv_layers:
            x = layer(x)
        return x   # (B, conv_dim[-1], T_features)


class Wav2Vec2FeatureProjection(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.feature_proj_layer_norm_eps)
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)

    def forward(self, x):
        # x: (B, T_features, conv_dim[-1]).
        x = self.layer_norm(x)
        return self.projection(x)


# --- Positional conv embedding (weight-normed Conv1d + GELU) -----------------


class Wav2Vec2PositionalConvEmbedding(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        conv = nn.Conv1d(
            config.hidden_size, config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )
        self.conv = nn.utils.parametrizations.weight_norm(conv, name="weight", dim=2)
        self.num_conv_pos_embeddings = config.num_conv_pos_embeddings

    def forward(self, x):
        # x: (B, T, hidden) -> conv on (B, hidden, T) -> back.
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Trim 1 element on the right when kernel is even (padding produces +1).
        if self.num_conv_pos_embeddings % 2 == 0:
            x = x[:, :, :-1]
        x = F.gelu(x)
        return x.transpose(1, 2)


# --- Transformer encoder layer ----------------------------------------------


class Wav2Vec2Attention(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def forward(self, x):
        B, S, H = x.shape
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2) * self.scaling
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1))   # scaling folded into Q.
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, S, H)
        return self.out_proj(out)


class Wav2Vec2FeedForward(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.output_dense(F.gelu(self.intermediate_dense(x)))


class Wav2Vec2EncoderLayer(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.attention = Wav2Vec2Attention(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        x = self.layer_norm(x + self.attention(x))
        return self.final_layer_norm(x + self.feed_forward(x))


class Wav2Vec2Encoder(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x):
        x = x + self.pos_conv_embed(x)
        x = self.layer_norm(x)
        for layer in self.layers:
            x = layer(x)
        return x


class Wav2Vec2Model(nn.Module):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.config = config
        # Learnable spec-augment embedding — included in state_dict so the
        # layout matches HF, but unused at eval (apply_spec_augment=False).
        self.masked_spec_embed = nn.Parameter(torch.zeros(config.hidden_size))
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)
        self.encoder = Wav2Vec2Encoder(config)

    def forward(self, input_values):
        x = self.feature_extractor(input_values)            # (B, conv_dim[-1], T)
        x = x.transpose(1, 2)                               # (B, T, conv_dim[-1])
        x = self.feature_projection(x)                      # (B, T, hidden)
        return self.encoder(x)


def build_pt_model(seed: int = 0) -> Wav2Vec2Model:
    config = Wav2Vec2Config()
    torch.manual_seed(seed)
    model = Wav2Vec2Model(config)
    nn.init.uniform_(model.masked_spec_embed, a=0.0, b=0.0)  # zero (deterministic)
    model.eval()
    return model


def main():
    model = build_pt_model()
    rng = torch.Generator().manual_seed(42)
    input_values = torch.rand((1, 256), generator=rng) * 2 - 1
    with torch.no_grad():
        out = model(input_values)
    print("last_hidden_state shape:", tuple(out.shape))
    print("checksum:", float(out.sum()))


if __name__ == "__main__":
    main()
