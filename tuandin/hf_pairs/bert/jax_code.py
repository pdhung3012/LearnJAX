"""From-scratch BERT forward in JAX/jax.numpy.

This is the *gold-standard reference translation* the eval harness expects.
We deliberately do NOT import `FlaxBertModel` — that would test API
knowledge, not translation skill. Instead we load raw PT weights via the
shared safetensors helper and re-implement BERT from primitives.

Architecture (matches transformers BertModel exactly for the small config):
- BertEmbeddings: word + position + token_type, then LayerNorm
- N x BertLayer:
    BertSelfAttention -> output Linear -> residual + LayerNorm
    intermediate Linear -> GELU -> output Linear -> residual + LayerNorm
- (No pooler — we return last_hidden_state, matching expected.npz)

Implementation notes that often trip translators:
- PT nn.Linear weight is (out, in); we apply via x @ W.T + b.
- LayerNorm eps = 1e-12 (BertConfig default), NOT 1e-5.
- Activation is `gelu` (exact, erf-based), NOT `gelu_new` (tanh approx).
- Attention mask: HF expands (B, S) to (B, 1, 1, S) and converts 0->masked,
  1->keep into additive (-inf vs 0).
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


# ---- Primitives -------------------------------------------------------------

def _layer_norm(x, gamma, beta, eps=1e-12):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * gamma + beta


def _gelu_exact(x):
    """PyTorch's default GELU (erf-based, not the tanh approximation)."""
    return jax.nn.gelu(x, approximate=False)


# ---- BERT components --------------------------------------------------------

def _embeddings(input_ids, token_type_ids, w):
    inputs_embeds = w["embeddings.word_embeddings.weight"][input_ids]            # (B, S, H)
    pos_ids = jnp.arange(input_ids.shape[1])
    position_embeds = w["embeddings.position_embeddings.weight"][pos_ids]        # (S, H)
    type_embeds = w["embeddings.token_type_embeddings.weight"][token_type_ids]   # (B, S, H)
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

    Q = Q.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)  # (B, h, S, d_h)
    K = K.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)

    scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / math.sqrt(d_head)
    if attention_mask is not None:
        # (B, S) -> (B, 1, 1, S); 1=keep, 0=mask. Convert to additive.
        m = attention_mask[:, None, None, :].astype(jnp.float32)
        scores = scores + (1.0 - m) * jnp.finfo(jnp.float32).min

    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(attn, V)                              # (B, h, S, d_h)
    return out.transpose(0, 2, 1, 3).reshape(B, S, H)      # (B, S, H)


def _bert_layer(x, attention_mask, w, layer_idx, num_heads):
    p = f"encoder.layer.{layer_idx}"

    # --- Self-attention sublayer -------------------------------------------
    a = _self_attention(x, attention_mask, f"{p}.attention.self", w, num_heads)
    a = a @ w[f"{p}.attention.output.dense.weight"].T \
          + w[f"{p}.attention.output.dense.bias"]
    x = _layer_norm(
        x + a,
        w[f"{p}.attention.output.LayerNorm.weight"],
        w[f"{p}.attention.output.LayerNorm.bias"],
    )

    # --- Feed-forward sublayer ---------------------------------------------
    h = x @ w[f"{p}.intermediate.dense.weight"].T \
          + w[f"{p}.intermediate.dense.bias"]
    h = _gelu_exact(h)
    h = h @ w[f"{p}.output.dense.weight"].T \
          + w[f"{p}.output.dense.bias"]
    x = _layer_norm(
        x + h,
        w[f"{p}.output.LayerNorm.weight"],
        w[f"{p}.output.LayerNorm.bias"],
    )
    return x


# ---- Contract API used by test_equivalence.py ------------------------------

def compute(inputs):
    """BERT forward pass.

    Args:
        inputs: dict with input_ids (B, S) int64, attention_mask (B, S) int64.
    Returns:
        dict with last_hidden_state (B, S, hidden_size).
    """
    config = load_pt_config(HERE / "pt_weights")
    num_heads = config["num_attention_heads"]
    num_layers = config["num_hidden_layers"]

    w = {k: jnp.asarray(v) for k, v in load_pt_safetensors(HERE / "pt_weights").items()}

    input_ids = jnp.asarray(inputs["input_ids"])
    attention_mask = jnp.asarray(inputs["attention_mask"])
    token_type_ids = jnp.zeros_like(input_ids)  # BERT default when not provided

    x = _embeddings(input_ids, token_type_ids, w)
    for layer_idx in range(num_layers):
        x = _bert_layer(x, attention_mask, w, layer_idx, num_heads)
    return {"last_hidden_state": np.asarray(x)}


if __name__ == "__main__":
    inputs = dict(np.load(HERE / "inputs.npz"))
    out = compute(inputs)
    print("last_hidden_state shape:", out["last_hidden_state"].shape)
    print("checksum:", float(out["last_hidden_state"].sum()))
