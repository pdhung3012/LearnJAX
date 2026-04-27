"""From-scratch RoBERTa forward in JAX/jax.numpy.

RoBERTa is architecturally identical to BERT, with two important quirks:
1. Position IDs start at `pad_token_id + 1` (= 2 for the standard config),
   not at 0. The actual formula HF uses:
       position_ids = (cumsum(attention_mask, axis=1) * attention_mask) + pad_id
   which gives [2, 3, 4, ...] for non-padding tokens and `pad_id` for padding.
2. token_type_embeddings has only one entry (type_vocab_size=1).

Other than that the layer stack matches BERT exactly. Mistakes that look
like RoBERTa-specific translation bugs almost always come from (1) above.
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


def _create_position_ids(input_ids, attention_mask, pad_id):
    """RoBERTa-specific: position_ids = (cumsum(mask) * mask) + pad_id, so
    the first non-pad token gets position pad_id+1."""
    mask = attention_mask.astype(jnp.int64)
    incremental = jnp.cumsum(mask, axis=1) * mask
    return incremental + pad_id


def _embeddings(input_ids, attention_mask, token_type_ids, pad_id, w):
    inputs_embeds = w["embeddings.word_embeddings.weight"][input_ids]
    pos_ids = _create_position_ids(input_ids, attention_mask, pad_id)
    position_embeds = w["embeddings.position_embeddings.weight"][pos_ids]
    type_embeds = w["embeddings.token_type_embeddings.weight"][token_type_ids]
    embeddings = inputs_embeds + position_embeds + type_embeds
    return _layer_norm(
        embeddings,
        w["embeddings.LayerNorm.weight"],
        w["embeddings.LayerNorm.bias"],
    )


def _self_attention(x, attention_mask, w_prefix, w, num_heads):
    B, S, H = x.shape
    d_head = H // num_heads
    Q = x @ w[f"{w_prefix}.query.weight"].T + w[f"{w_prefix}.query.bias"]
    K = x @ w[f"{w_prefix}.key.weight"].T   + w[f"{w_prefix}.key.bias"]
    V = x @ w[f"{w_prefix}.value.weight"].T + w[f"{w_prefix}.value.bias"]
    Q = Q.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / math.sqrt(d_head)
    if attention_mask is not None:
        m = attention_mask[:, None, None, :].astype(jnp.float32)
        scores = scores + (1.0 - m) * jnp.finfo(jnp.float32).min
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(attn, V)
    return out.transpose(0, 2, 1, 3).reshape(B, S, H)


def _layer(x, attention_mask, w, layer_idx, num_heads):
    p = f"encoder.layer.{layer_idx}"
    a = _self_attention(x, attention_mask, f"{p}.attention.self", w, num_heads)
    a = a @ w[f"{p}.attention.output.dense.weight"].T \
          + w[f"{p}.attention.output.dense.bias"]
    x = _layer_norm(x + a,
                    w[f"{p}.attention.output.LayerNorm.weight"],
                    w[f"{p}.attention.output.LayerNorm.bias"])
    h = x @ w[f"{p}.intermediate.dense.weight"].T + w[f"{p}.intermediate.dense.bias"]
    h = _gelu_exact(h)
    h = h @ w[f"{p}.output.dense.weight"].T + w[f"{p}.output.dense.bias"]
    x = _layer_norm(x + h,
                    w[f"{p}.output.LayerNorm.weight"],
                    w[f"{p}.output.LayerNorm.bias"])
    return x


def compute(inputs):
    """RoBERTa forward pass."""
    config = load_pt_config(HERE / "pt_weights")
    num_heads = config["num_attention_heads"]
    num_layers = config["num_hidden_layers"]
    pad_id = config["pad_token_id"]

    w = {k: jnp.asarray(v) for k, v in load_pt_safetensors(HERE / "pt_weights").items()}

    input_ids = jnp.asarray(inputs["input_ids"])
    attention_mask = jnp.asarray(inputs["attention_mask"])
    token_type_ids = jnp.zeros_like(input_ids)

    x = _embeddings(input_ids, attention_mask, token_type_ids, pad_id, w)
    for layer_idx in range(num_layers):
        x = _layer(x, attention_mask, w, layer_idx, num_heads)
    return {"last_hidden_state": np.asarray(x)}


if __name__ == "__main__":
    inputs = dict(np.load(HERE / "inputs.npz"))
    out = compute(inputs)
    print("last_hidden_state shape:", out["last_hidden_state"].shape)
    print("checksum:", float(out["last_hidden_state"].sum()))
