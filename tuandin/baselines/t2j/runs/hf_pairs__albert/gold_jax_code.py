"""From-scratch ALBERT forward in JAX/jax.numpy.

Two architectural twists not present in the other cases:

1. **Factorized embeddings** — word/position/token_type embeddings live at
   `embedding_size` (here 48), then `encoder.embedding_hidden_mapping_in`
   projects up to `hidden_size` (here 64). The embedding LayerNorm is at
   embedding_size, before the projection.

2. **Cross-layer parameter sharing** — the saved state_dict contains exactly
   ONE set of layer weights (`encoder.albert_layer_groups.0.albert_layers.0.*`).
   The forward runs that same block `num_hidden_layers` times. With default
   num_hidden_groups=1 and inner_group_num=1, every iteration reuses the
   single block (the most common ALBERT setup).

Other notes:
- Activation defaults to `gelu_new` (tanh approximation), unlike BERT's
  exact GELU.
- LayerNorm eps = 1e-12.
- The `attention.LayerNorm` applies AFTER attention+dense+residual; the
  `full_layer_layer_norm` applies AFTER the FFN+residual (post-norm).
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


def _gelu_new(x):
    return jax.nn.gelu(x, approximate=True)


def _embeddings_and_project(input_ids, token_type_ids, w):
    word = w["embeddings.word_embeddings.weight"][input_ids]
    pos_ids = jnp.arange(input_ids.shape[1])
    pos = w["embeddings.position_embeddings.weight"][pos_ids]
    type_ = w["embeddings.token_type_embeddings.weight"][token_type_ids]
    e = _layer_norm(
        word + pos + type_,
        w["embeddings.LayerNorm.weight"],
        w["embeddings.LayerNorm.bias"],
    )
    # Project from embedding_size up to hidden_size.
    return e @ w["encoder.embedding_hidden_mapping_in.weight"].T \
             + w["encoder.embedding_hidden_mapping_in.bias"]


def _shared_layer(x, attention_mask, w, num_heads):
    """The single shared layer block — applied repeatedly for each of the
    num_hidden_layers logical layers."""
    p = "encoder.albert_layer_groups.0.albert_layers.0"
    B, S, H = x.shape
    d_head = H // num_heads

    # Attention block.
    Q = x @ w[f"{p}.attention.query.weight"].T + w[f"{p}.attention.query.bias"]
    K = x @ w[f"{p}.attention.key.weight"].T   + w[f"{p}.attention.key.bias"]
    V = x @ w[f"{p}.attention.value.weight"].T + w[f"{p}.attention.value.bias"]
    Q = Q.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / math.sqrt(d_head)
    if attention_mask is not None:
        m = attention_mask[:, None, None, :].astype(jnp.float32)
        scores = scores + (1.0 - m) * jnp.finfo(jnp.float32).min
    attn = jax.nn.softmax(scores, axis=-1)
    a = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, S, H)
    a = a @ w[f"{p}.attention.dense.weight"].T + w[f"{p}.attention.dense.bias"]
    x = _layer_norm(x + a,
                    w[f"{p}.attention.LayerNorm.weight"],
                    w[f"{p}.attention.LayerNorm.bias"])

    # FFN block.
    h = x @ w[f"{p}.ffn.weight"].T + w[f"{p}.ffn.bias"]
    h = _gelu_new(h)
    h = h @ w[f"{p}.ffn_output.weight"].T + w[f"{p}.ffn_output.bias"]
    x = _layer_norm(x + h,
                    w[f"{p}.full_layer_layer_norm.weight"],
                    w[f"{p}.full_layer_layer_norm.bias"])
    return x


def compute(inputs):
    config = load_pt_config(HERE / "pt_weights")
    num_heads = config["num_attention_heads"]
    num_layers = config["num_hidden_layers"]
    w = {k: jnp.asarray(v) for k, v in load_pt_safetensors(HERE / "pt_weights").items()}

    input_ids = jnp.asarray(inputs["input_ids"])
    attention_mask = jnp.asarray(inputs["attention_mask"])
    token_type_ids = jnp.zeros_like(input_ids)

    x = _embeddings_and_project(input_ids, token_type_ids, w)
    for _ in range(num_layers):
        x = _shared_layer(x, attention_mask, w, num_heads)
    return {"last_hidden_state": np.asarray(x)}


if __name__ == "__main__":
    inputs = dict(np.load(HERE / "inputs.npz"))
    out = compute(inputs)
    print("last_hidden_state shape:", out["last_hidden_state"].shape)
    print("checksum:", float(out["last_hidden_state"].sum()))
