"""From-scratch Wav2Vec2 forward in JAX/jax.numpy.

Architecture (matches transformers Wav2Vec2Model with feat_extract_norm='layer'):
- 1-D conv feature extractor (3 layers): each is Conv1d -> LayerNorm (on the
  channel dim) -> GELU.
- Feature projection: LayerNorm + Linear(conv_dim[-1] -> hidden_size).
- Encoder:
    - Positional conv embedding: weight-normed grouped Conv1d + GELU.
      In state_dict the weight is stored as (`original0` magnitude,
      `original1` direction) and we reconstruct it on load.
    - Add to features, then LayerNorm.
    - N x EncoderLayer (post-norm BERT-style with bias-having attention).

Implementation notes that often trip translators:
- Conv1d weights are PyTorch shape (out, in, kernel); for
  `jax.lax.conv_general_dilated` with dimension_numbers ('NCH', 'OIH', 'NCH')
  the kernel uses the same layout — no transpose.
- weight_norm reconstruction: weight = direction * (magnitude / ||direction||),
  with the norm computed over all dims EXCEPT the parametrization dim
  (here dim=2, the kernel dim). Result shape = same as direction.
- For an even-kernel pos_conv with padding=kernel//2, the output has length
  T+1; HF trims the last position. We do the same.
- HF Wav2Vec2 attention folds the 1/sqrt(d_head) scale into Q before the
  matmul rather than dividing scores afterwards.
"""
import math
import sys
from pathlib import Path

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from _weight_loader import load_pt_config, load_pt_safetensors


def _layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * gamma + beta


def _gelu_exact(x):
    return jax.nn.gelu(x, approximate=False)


# ---- Feature extractor ------------------------------------------------------


def _conv1d(x, weight, bias=None, stride=1, padding="VALID", groups=1):
    """1-D convolution. Input x: (B, C, T). Kernel: (out, in_per_group, k)."""
    out = lax.conv_general_dilated(
        x, weight, window_strides=(stride,), padding=padding,
        dimension_numbers=("NCH", "OIH", "NCH"),
        feature_group_count=groups,
    )
    if bias is not None:
        out = out + bias[None, :, None]
    return out


def _feature_extractor(x, w, conv_kernels, conv_strides):
    # x: (B, T_audio) -> (B, 1, T_audio).
    x = x[:, None, :]
    for i, (k, s) in enumerate(zip(conv_kernels, conv_strides)):
        weight = w[f"feature_extractor.conv_layers.{i}.conv.weight"]
        x = _conv1d(x, weight, stride=s, padding="VALID")
        # LayerNorm along channel dim: transpose to (B, T, C), norm, transpose.
        x = jnp.transpose(x, (0, 2, 1))
        x = _layer_norm(
            x,
            w[f"feature_extractor.conv_layers.{i}.layer_norm.weight"],
            w[f"feature_extractor.conv_layers.{i}.layer_norm.bias"],
        )
        x = jnp.transpose(x, (0, 2, 1))
        x = _gelu_exact(x)
    return x   # (B, conv_dim[-1], T_features)


def _feature_projection(x, w, eps=1e-5):
    # x: (B, T, conv_dim[-1]).
    x = _layer_norm(x,
                    w["feature_projection.layer_norm.weight"],
                    w["feature_projection.layer_norm.bias"], eps=eps)
    return x @ w["feature_projection.projection.weight"].T \
             + w["feature_projection.projection.bias"]


# ---- Positional conv embedding ---------------------------------------------


def _build_pos_conv_weight(w):
    """Reconstruct weight from weight_norm parametrization."""
    g = w["encoder.pos_conv_embed.conv.parametrizations.weight.original0"]  # (1, 1, K)
    v = w["encoder.pos_conv_embed.conv.parametrizations.weight.original1"]  # (out, in_g, K)
    norm = jnp.linalg.norm(v, axis=(0, 1), keepdims=True)                   # (1, 1, K)
    return v * (g / norm)


def _pos_conv_embed(x, w, num_conv_pos_embeddings, num_groups):
    # x: (B, T, hidden) -> (B, hidden, T).
    x = jnp.transpose(x, (0, 2, 1))
    weight = _build_pos_conv_weight(w)
    bias = w["encoder.pos_conv_embed.conv.bias"]
    pad = num_conv_pos_embeddings // 2
    x = _conv1d(x, weight, bias=bias, stride=1,
                padding=((pad, pad),), groups=num_groups)
    if num_conv_pos_embeddings % 2 == 0:
        x = x[:, :, :-1]
    x = _gelu_exact(x)
    return jnp.transpose(x, (0, 2, 1))


# ---- Transformer encoder layer ---------------------------------------------


def _self_attn(x, w, layer_idx, num_heads):
    p = f"encoder.layers.{layer_idx}.attention"
    B, S, H = x.shape
    d_head = H // num_heads
    scaling = d_head ** -0.5
    Q = (x @ w[f"{p}.q_proj.weight"].T + w[f"{p}.q_proj.bias"]) * scaling
    K = x @ w[f"{p}.k_proj.weight"].T + w[f"{p}.k_proj.bias"]
    V = x @ w[f"{p}.v_proj.weight"].T + w[f"{p}.v_proj.bias"]
    Q = Q.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1))   # scaling already in Q.
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, S, H)
    return out @ w[f"{p}.out_proj.weight"].T + w[f"{p}.out_proj.bias"]


def _layer(x, w, layer_idx, num_heads, eps):
    p = f"encoder.layers.{layer_idx}"
    a = _self_attn(x, w, layer_idx, num_heads)
    x = _layer_norm(x + a, w[f"{p}.layer_norm.weight"], w[f"{p}.layer_norm.bias"], eps=eps)
    h = x @ w[f"{p}.feed_forward.intermediate_dense.weight"].T \
          + w[f"{p}.feed_forward.intermediate_dense.bias"]
    h = _gelu_exact(h)
    h = h @ w[f"{p}.feed_forward.output_dense.weight"].T \
          + w[f"{p}.feed_forward.output_dense.bias"]
    return _layer_norm(x + h, w[f"{p}.final_layer_norm.weight"],
                       w[f"{p}.final_layer_norm.bias"], eps=eps)


def compute(inputs):
    config = load_pt_config(HERE / "pt_weights")
    num_heads = config["num_attention_heads"]
    num_layers = config["num_hidden_layers"]
    eps = config["layer_norm_eps"]
    conv_kernels = tuple(config["conv_kernel"])
    conv_strides = tuple(config["conv_stride"])
    num_pos = config["num_conv_pos_embeddings"]
    num_pos_groups = config["num_conv_pos_embedding_groups"]

    w = {k: jnp.asarray(v) for k, v in load_pt_safetensors(HERE / "pt_weights").items()}

    input_values = jnp.asarray(inputs["input_values"])

    x = _feature_extractor(input_values, w, conv_kernels, conv_strides)
    x = jnp.transpose(x, (0, 2, 1))                       # (B, T, conv_dim[-1])
    x = _feature_projection(x, w, eps=eps)
    pos = _pos_conv_embed(x, w, num_pos, num_pos_groups)
    x = x + pos
    x = _layer_norm(x, w["encoder.layer_norm.weight"],
                    w["encoder.layer_norm.bias"], eps=eps)

    for layer_idx in range(num_layers):
        x = _layer(x, w, layer_idx, num_heads, eps)
    return {"last_hidden_state": np.asarray(x)}


if __name__ == "__main__":
    inputs = dict(np.load(HERE / "inputs.npz"))
    out = compute(inputs)
    print("last_hidden_state shape:", out["last_hidden_state"].shape)
    print("checksum:", float(out["last_hidden_state"].sum()))
