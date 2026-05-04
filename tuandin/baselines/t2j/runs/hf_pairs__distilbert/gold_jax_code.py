"""From-scratch DistilBERT forward in JAX/jax.numpy.

DistilBERT vs BERT — the differences worth remembering:
- No token_type_embeddings (single token type only).
- Linear layer names use *_lin (q_lin, k_lin, v_lin, out_lin) not query/key/value/dense.
- LayerNorms are named sa_layer_norm and output_layer_norm (post-attention and post-FFN).
- The final encoder block doesn't have a separate "pooler" — we just return last_hidden_state.
- LayerNorm eps is 1e-12 (same as BERT).
- Activation is `gelu` (exact, not the tanh approximation).
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


def _layer_norm(x, gamma, beta, eps=1e-12):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * gamma + beta


def _gelu_exact(x):
    return jax.nn.gelu(x, approximate=False)


def _embeddings(input_ids, w):
    inputs_embeds = w["embeddings.word_embeddings.weight"][input_ids]
    pos_ids = jnp.arange(input_ids.shape[1])
    position_embeds = w["embeddings.position_embeddings.weight"][pos_ids]
    return _layer_norm(
        inputs_embeds + position_embeds,
        w["embeddings.LayerNorm.weight"],
        w["embeddings.LayerNorm.bias"],
    )


def _self_attn(x, attention_mask, p, w, num_heads):
    B, S, H = x.shape
    d_head = H // num_heads
    Q = x @ w[f"{p}.attention.q_lin.weight"].T + w[f"{p}.attention.q_lin.bias"]
    K = x @ w[f"{p}.attention.k_lin.weight"].T + w[f"{p}.attention.k_lin.bias"]
    V = x @ w[f"{p}.attention.v_lin.weight"].T + w[f"{p}.attention.v_lin.bias"]
    Q = Q.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / math.sqrt(d_head)
    if attention_mask is not None:
        m = attention_mask[:, None, None, :].astype(jnp.float32)
        scores = scores + (1.0 - m) * jnp.finfo(jnp.float32).min
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, S, H)
    return out @ w[f"{p}.attention.out_lin.weight"].T + w[f"{p}.attention.out_lin.bias"]


def _ffn(x, p, w):
    h = x @ w[f"{p}.ffn.lin1.weight"].T + w[f"{p}.ffn.lin1.bias"]
    h = _gelu_exact(h)
    return h @ w[f"{p}.ffn.lin2.weight"].T + w[f"{p}.ffn.lin2.bias"]


def _layer(x, attention_mask, w, layer_idx, num_heads):
    p = f"transformer.layer.{layer_idx}"
    a = _self_attn(x, attention_mask, p, w, num_heads)
    x = _layer_norm(x + a,
                    w[f"{p}.sa_layer_norm.weight"],
                    w[f"{p}.sa_layer_norm.bias"])
    h = _ffn(x, p, w)
    x = _layer_norm(x + h,
                    w[f"{p}.output_layer_norm.weight"],
                    w[f"{p}.output_layer_norm.bias"])
    return x


def compute(inputs):
    config = load_pt_config(HERE / "pt_weights")
    num_heads = config["n_heads"]
    num_layers = config["n_layers"]
    w = {k: jnp.asarray(v) for k, v in load_pt_safetensors(HERE / "pt_weights").items()}

    input_ids = jnp.asarray(inputs["input_ids"])
    attention_mask = jnp.asarray(inputs["attention_mask"])

    x = _embeddings(input_ids, w)
    for layer_idx in range(num_layers):
        x = _layer(x, attention_mask, w, layer_idx, num_heads)
    return {"last_hidden_state": np.asarray(x)}


if __name__ == "__main__":
    inputs = dict(np.load(HERE / "inputs.npz"))
    out = compute(inputs)
    print("last_hidden_state shape:", out["last_hidden_state"].shape)
    print("checksum:", float(out["last_hidden_state"].sum()))
