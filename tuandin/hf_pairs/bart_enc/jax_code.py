"""From-scratch BART encoder forward in JAX/jax.numpy.

Architecture (matches transformers BartModel.encoder for the small config):
- Token embeddings + learned-positional embeddings (with +2 offset).
- LayerNorm of the embedding sum.
- N x BartEncoderLayer (post-norm style):
    self-attn -> residual -> LayerNorm
    FFN(GELU) -> residual -> LayerNorm

BART quirks:
- BartLearnedPositionalEmbedding has size (max_pos_emb + 2) and is indexed
  by `arange(S) + 2`. The first 2 slots are reserved (HF convention).
- HF BartAttention folds the 1/sqrt(d_head) scaling into Q (multiplies Q
  *before* the matmul) rather than dividing scores. We do the same so weights
  load identically.
- Activation: exact GELU (erf-based).
- Embedding scaling = sqrt(d_model) when scale_embedding=True; default False.
"""
import math
import sys
from pathlib import Path

import jax
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


def _self_attn(x, attention_mask, w, layer_idx, num_heads):
    p = f"layers.{layer_idx}.self_attn"
    B, S, H = x.shape
    d_head = H // num_heads
    scaling = d_head ** -0.5
    Q = (x @ w[f"{p}.q_proj.weight"].T + w[f"{p}.q_proj.bias"]) * scaling
    K = x @ w[f"{p}.k_proj.weight"].T + w[f"{p}.k_proj.bias"]
    V = x @ w[f"{p}.v_proj.weight"].T + w[f"{p}.v_proj.bias"]
    Q = Q.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    # Scaling already folded into Q; no extra divide here.
    scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1))
    if attention_mask is not None:
        m = attention_mask[:, None, None, :].astype(jnp.float32)
        scores = scores + (1.0 - m) * jnp.finfo(jnp.float32).min
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, S, H)
    return out @ w[f"{p}.out_proj.weight"].T + w[f"{p}.out_proj.bias"]


def _layer(x, attention_mask, w, layer_idx, num_heads):
    p = f"layers.{layer_idx}"
    a = _self_attn(x, attention_mask, w, layer_idx, num_heads)
    x = _layer_norm(x + a,
                    w[f"{p}.self_attn_layer_norm.weight"],
                    w[f"{p}.self_attn_layer_norm.bias"])
    h = x @ w[f"{p}.fc1.weight"].T + w[f"{p}.fc1.bias"]
    h = _gelu_exact(h)
    h = h @ w[f"{p}.fc2.weight"].T + w[f"{p}.fc2.bias"]
    x = _layer_norm(x + h,
                    w[f"{p}.final_layer_norm.weight"],
                    w[f"{p}.final_layer_norm.bias"])
    return x


def compute(inputs):
    config = load_pt_config(HERE / "pt_weights")
    num_heads = config["encoder_attention_heads"]
    num_layers = config["encoder_layers"]
    embed_scale = math.sqrt(config["d_model"]) if config.get("scale_embedding", False) else 1.0

    w = {k: jnp.asarray(v) for k, v in load_pt_safetensors(HERE / "pt_weights").items()}

    input_ids = jnp.asarray(inputs["input_ids"])
    attention_mask = jnp.asarray(inputs["attention_mask"])

    # Embeddings: tokens (* embed_scale) + learned positions (offset by 2).
    S = input_ids.shape[1]
    pos_ids = jnp.arange(S) + 2
    x = w["embed_tokens.weight"][input_ids] * embed_scale
    x = x + w["embed_positions.weight"][pos_ids]
    x = _layer_norm(x,
                    w["layernorm_embedding.weight"],
                    w["layernorm_embedding.bias"])

    for layer_idx in range(num_layers):
        x = _layer(x, attention_mask, w, layer_idx, num_heads)
    return {"last_hidden_state": np.asarray(x)}


if __name__ == "__main__":
    inputs = dict(np.load(HERE / "inputs.npz"))
    out = compute(inputs)
    print("last_hidden_state shape:", out["last_hidden_state"].shape)
    print("checksum:", float(out["last_hidden_state"].sum()))
