"""From-scratch GPT-2 forward in JAX/jax.numpy.

Pre-norm decoder-only transformer with causal mask.

Implementation notes that often trip translators:
- GPT-2's c_attn / c_proj / c_fc are torch.nn.Conv1D, NOT nn.Linear. The
  Conv1D weight shape is (in, out) — opposite of nn.Linear's (out, in). So
  we apply via x @ W + b directly (NO transpose).
- c_attn produces a fused (Q, K, V) of width 3 * hidden_size; we split.
- Pre-norm: LayerNorm BEFORE attention/FFN, residual added after.
- Activation is `gelu_new` (tanh approximation), not exact GELU.
- Causal mask must combine with the attention_mask (padding mask).
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


def _gelu_new(x):
    """GPT-2's gelu_new: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))."""
    return jax.nn.gelu(x, approximate=True)


def _attn_block(x, attention_mask, w, layer_idx, num_heads):
    p = f"h.{layer_idx}"
    B, S, H = x.shape
    d_head = H // num_heads

    # Pre-norm.
    h = _layer_norm(x, w[f"{p}.ln_1.weight"], w[f"{p}.ln_1.bias"])

    # Conv1D layers: weight shape (in, out), so x @ W (no .T).
    qkv = h @ w[f"{p}.attn.c_attn.weight"] + w[f"{p}.attn.c_attn.bias"]  # (B, S, 3H)
    Q, K, V = jnp.split(qkv, 3, axis=-1)

    Q = Q.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)

    scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / math.sqrt(d_head)
    # Causal mask: (S, S) lower triangular; True = keep.
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    scores = jnp.where(causal[None, None, :, :], scores, jnp.finfo(jnp.float32).min)
    if attention_mask is not None:
        m = attention_mask[:, None, None, :].astype(jnp.float32)
        scores = scores + (1.0 - m) * jnp.finfo(jnp.float32).min

    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(attn, V)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, H)

    out = out @ w[f"{p}.attn.c_proj.weight"] + w[f"{p}.attn.c_proj.bias"]
    return x + out  # residual


def _mlp_block(x, w, layer_idx):
    p = f"h.{layer_idx}"
    h = _layer_norm(x, w[f"{p}.ln_2.weight"], w[f"{p}.ln_2.bias"])
    h = h @ w[f"{p}.mlp.c_fc.weight"] + w[f"{p}.mlp.c_fc.bias"]
    h = _gelu_new(h)
    h = h @ w[f"{p}.mlp.c_proj.weight"] + w[f"{p}.mlp.c_proj.bias"]
    return x + h  # residual


def compute(inputs):
    """GPT-2 forward pass."""
    config = load_pt_config(HERE / "pt_weights")
    num_heads = config["n_head"]
    num_layers = config["n_layer"]

    w = {k: jnp.asarray(v) for k, v in load_pt_safetensors(HERE / "pt_weights").items()}

    input_ids = jnp.asarray(inputs["input_ids"])
    attention_mask = jnp.asarray(inputs["attention_mask"])

    pos_ids = jnp.arange(input_ids.shape[1])
    x = w["wte.weight"][input_ids] + w["wpe.weight"][pos_ids]

    for layer_idx in range(num_layers):
        x = _attn_block(x, attention_mask, w, layer_idx, num_heads)
        x = _mlp_block(x, w, layer_idx)

    x = _layer_norm(x, w["ln_f.weight"], w["ln_f.bias"])
    return {"last_hidden_state": np.asarray(x)}


if __name__ == "__main__":
    inputs = dict(np.load(HERE / "inputs.npz"))
    out = compute(inputs)
    print("last_hidden_state shape:", out["last_hidden_state"].shape)
    print("checksum:", float(out["last_hidden_state"].sum()))
