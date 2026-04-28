"""From-scratch Mistral forward in JAX/jax.numpy.

Architecture (matches transformers MistralModel for the small config):
- Token embedding (no abs pos embed).
- N x MistralDecoderLayer (pre-norm):
    RMSNorm -> SelfAttention (RoPE + GQA) -> +residual
    RMSNorm -> SwiGLU MLP                 -> +residual
- Final RMSNorm.

Stresses several patterns at once:
- RMSNorm (rescale-by-RMS, no mean centering, no bias).
- RoPE applied to Q and K only (not V).
- GQA: K/V have num_kv_heads heads; we repeat them along axis 1 to match
  num_q_heads before the dot product.
- SwiGLU: down(silu(gate(x)) * up(x)). All linears are bias-free.
- Causal masking + (B, S) padding mask.
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


def _rms_norm(x, gamma, eps=1e-6):
    var = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return gamma * x * jax.lax.rsqrt(var + eps)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _build_rope(seq_len, head_dim, base):
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.einsum("i,j->ij", pos, inv_freq)        # (S, D/2)
    emb = jnp.concatenate([angles, angles], axis=-1)     # (S, D)
    return jnp.cos(emb)[None, None], jnp.sin(emb)[None, None]   # (1, 1, S, D)


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, H_kv, S, D = x.shape
    return jnp.broadcast_to(x[:, :, None, :, :], (B, H_kv, n_rep, S, D)).reshape(
        B, H_kv * n_rep, S, D
    )


def _self_attn(x, attention_mask, w, layer_idx, num_heads, num_kv_heads,
               head_dim, rope_theta):
    p = f"layers.{layer_idx}.self_attn"
    B, S, H = x.shape

    Q = x @ w[f"{p}.q_proj.weight"].T
    K = x @ w[f"{p}.k_proj.weight"].T
    V = x @ w[f"{p}.v_proj.weight"].T
    Q = Q.reshape(B, S, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    cos, sin = _build_rope(S, head_dim, rope_theta)
    Q = Q * cos + _rotate_half(Q) * sin
    K = K * cos + _rotate_half(K) * sin

    n_rep = num_heads // num_kv_heads
    K = _repeat_kv(K, n_rep)
    V = _repeat_kv(V, n_rep)

    scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / math.sqrt(head_dim)
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    scores = jnp.where(causal[None, None], scores, jnp.finfo(jnp.float32).min)
    if attention_mask is not None:
        m = attention_mask[:, None, None, :].astype(jnp.float32)
        scores = scores + (1.0 - m) * jnp.finfo(jnp.float32).min

    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, S, H)
    return out @ w[f"{p}.o_proj.weight"].T


def _mlp(x, w, layer_idx):
    p = f"layers.{layer_idx}.mlp"
    gate = x @ w[f"{p}.gate_proj.weight"].T
    up   = x @ w[f"{p}.up_proj.weight"].T
    return (jax.nn.silu(gate) * up) @ w[f"{p}.down_proj.weight"].T


def _layer(x, attention_mask, w, layer_idx, eps, num_heads, num_kv_heads,
           head_dim, rope_theta):
    p = f"layers.{layer_idx}"
    h = _rms_norm(x, w[f"{p}.input_layernorm.weight"], eps=eps)
    x = x + _self_attn(h, attention_mask, w, layer_idx, num_heads, num_kv_heads,
                       head_dim, rope_theta)
    h = _rms_norm(x, w[f"{p}.post_attention_layernorm.weight"], eps=eps)
    x = x + _mlp(h, w, layer_idx)
    return x


def compute(inputs):
    config = load_pt_config(HERE / "pt_weights")
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    num_layers = config["num_hidden_layers"]
    head_dim = config["hidden_size"] // num_heads
    eps = config["rms_norm_eps"]
    rope_theta = config["rope_theta"]

    w = {k: jnp.asarray(v) for k, v in load_pt_safetensors(HERE / "pt_weights").items()}

    input_ids = jnp.asarray(inputs["input_ids"])
    attention_mask = jnp.asarray(inputs["attention_mask"])

    x = w["embed_tokens.weight"][input_ids]
    for layer_idx in range(num_layers):
        x = _layer(x, attention_mask, w, layer_idx, eps, num_heads, num_kv_heads,
                   head_dim, rope_theta)
    x = _rms_norm(x, w["norm.weight"], eps=eps)
    return {"last_hidden_state": np.asarray(x)}


if __name__ == "__main__":
    inputs = dict(np.load(HERE / "inputs.npz"))
    out = compute(inputs)
    print("last_hidden_state shape:", out["last_hidden_state"].shape)
    print("checksum:", float(out["last_hidden_state"].sum()))
