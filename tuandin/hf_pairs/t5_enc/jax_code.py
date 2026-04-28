"""From-scratch T5 encoder forward in JAX/jax.numpy.

Pre-norm encoder with several T5-specific quirks that make it the most
unforgiving translation target in this tier:

1. **T5LayerNorm** — RMS norm, no mean centering, no bias, eps=1e-6.
2. **No QK scaling** — T5 does NOT divide attention scores by sqrt(d_head).
3. **No biases** on any Linear (q, k, v, o, wi, wo).
4. **Relative position bias** instead of absolute position embeddings:
   computed from a learned `relative_attention_bias` embedding indexed by a
   non-trivial bucketing function, ADDED to attention scores. Computed in
   the first layer only and reused across all layers.
5. **ReLU** activation (T5-base; T5-v1.1 uses gated GELU but the default
   T5Config sets feed_forward_proj='relu').
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


def _t5_layer_norm(x, gamma, eps=1e-6):
    """RMS norm: no mean centering, no bias, just rescale by RMS."""
    var = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + eps) * gamma


def _relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
    """T5 relative-position bucketing. Bidirectional (encoder) splits buckets
    in half by sign; one half handles negative offsets, the other positive.
    Within each half, small distances map exactly; large distances are
    log-spaced up to `max_distance`."""
    ret = jnp.zeros_like(relative_position)
    n = relative_position
    if bidirectional:
        num_buckets //= 2
        ret = ret + (n > 0).astype(jnp.int32) * num_buckets
        n = jnp.abs(n)
    else:
        n = -jnp.minimum(n, jnp.zeros_like(n))

    max_exact = num_buckets // 2
    is_small = n < max_exact

    # Log-spaced bucket index for large distances.
    n_clipped = jnp.maximum(n, max_exact).astype(jnp.float32)
    val_if_large = max_exact + (
        jnp.log(n_clipped / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(jnp.int32)
    val_if_large = jnp.minimum(val_if_large, num_buckets - 1)

    return ret + jnp.where(is_small, n, val_if_large)


def _compute_bias(qlen, klen, rel_attn_table, num_buckets=32, max_distance=128):
    """Build (1, num_heads, qlen, klen) additive position bias from the
    learned relative_attention_bias.weight (num_buckets, num_heads)."""
    context = jnp.arange(qlen, dtype=jnp.int32)[:, None]
    memory = jnp.arange(klen, dtype=jnp.int32)[None, :]
    relative_position = memory - context  # (qlen, klen)
    rp_bucket = _relative_position_bucket(
        relative_position, bidirectional=True,
        num_buckets=num_buckets, max_distance=max_distance,
    )                                    # (qlen, klen)
    values = rel_attn_table[rp_bucket]   # (qlen, klen, num_heads)
    return values.transpose(2, 0, 1)[None, ...]  # (1, num_heads, qlen, klen)


def _self_attn(x, attention_mask, position_bias, w, layer_idx, num_heads):
    p = f"encoder.block.{layer_idx}.layer.0.SelfAttention"
    B, S, H = x.shape
    d_head = H // num_heads

    Q = x @ w[f"{p}.q.weight"].T
    K = x @ w[f"{p}.k.weight"].T
    V = x @ w[f"{p}.v.weight"].T
    Q = Q.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)

    scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1))   # NOTE: T5 does NOT scale.
    scores = scores + position_bias
    if attention_mask is not None:
        m = attention_mask[:, None, None, :].astype(jnp.float32)
        scores = scores + (1.0 - m) * jnp.finfo(jnp.float32).min

    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, S, H)
    return out @ w[f"{p}.o.weight"].T


def _ff(x, w, layer_idx):
    p = f"encoder.block.{layer_idx}.layer.1.DenseReluDense"
    h = x @ w[f"{p}.wi.weight"].T
    h = jax.nn.relu(h)
    return h @ w[f"{p}.wo.weight"].T


def _block(x, attention_mask, position_bias, w, layer_idx, num_heads):
    # Self-attention sublayer (pre-norm + residual).
    h = _t5_layer_norm(x, w[f"encoder.block.{layer_idx}.layer.0.layer_norm.weight"])
    x = x + _self_attn(h, attention_mask, position_bias, w, layer_idx, num_heads)
    # FF sublayer.
    h = _t5_layer_norm(x, w[f"encoder.block.{layer_idx}.layer.1.layer_norm.weight"])
    x = x + _ff(h, w, layer_idx)
    return x


def compute(inputs):
    config = load_pt_config(HERE / "pt_weights")
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    num_buckets = config.get("relative_attention_num_buckets", 32)
    max_distance = config.get("relative_attention_max_distance", 128)

    w = {k: jnp.asarray(v) for k, v in load_pt_safetensors(HERE / "pt_weights").items()}

    input_ids = jnp.asarray(inputs["input_ids"])
    attention_mask = jnp.asarray(inputs["attention_mask"])

    x = w["shared.weight"][input_ids]   # token embeddings (no abs pos embed in T5)

    # Position bias is computed once from layer 0's relative_attention_bias
    # and reused across all layers in HF's implementation.
    rel_table = w["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]
    position_bias = _compute_bias(
        x.shape[1], x.shape[1], rel_table,
        num_buckets=num_buckets, max_distance=max_distance,
    )
    if attention_mask is not None:
        # HF *also* applies the additive padding mask via `position_bias` in
        # T5; we apply it inside _self_attn to keep the function pure.
        pass

    for layer_idx in range(num_layers):
        x = _block(x, attention_mask, position_bias, w, layer_idx, num_heads)
    x = _t5_layer_norm(x, w["encoder.final_layer_norm.weight"])
    return {"last_hidden_state": np.asarray(x)}


if __name__ == "__main__":
    inputs = dict(np.load(HERE / "inputs.npz"))
    out = compute(inputs)
    print("last_hidden_state shape:", out["last_hidden_state"].shape)
    print("checksum:", float(out["last_hidden_state"].sum()))
