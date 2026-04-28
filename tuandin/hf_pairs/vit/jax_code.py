"""From-scratch ViT forward in JAX/jax.numpy.

Architecture:
1. Patch embed: Conv2d(in=3, out=hidden, kH=patch, kW=patch, stride=patch).
   - PT input is NCHW, PT weight is (out, in, kH, kW).
   - JAX/lax.conv expects (NHWC, HWIO, NHWC) by default.
2. Prepend a learned [CLS] token.
3. Add learned absolute position embeddings (1, num_patches+1, hidden).
4. N x ViTLayer (pre-norm):
       layernorm_before -> self-attn -> residual
       layernorm_after  -> FFN(GELU) -> residual
5. Final LayerNorm on the full sequence.

Implementation notes:
- LayerNorm eps = 1e-12 (ViTConfig default).
- Activation: exact GELU (erf-based).
- Attention does scale by sqrt(d_head) (ViT, unlike T5).
- Pre-norm: LayerNorm BEFORE the sub-block, residual added after.
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


def _layer_norm(x, gamma, beta, eps=1e-12):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * gamma + beta


def _gelu_exact(x):
    return jax.nn.gelu(x, approximate=False)


def _patch_embed(pixel_values, w):
    # NCHW -> NHWC for jax.lax.
    x = jnp.transpose(pixel_values, (0, 2, 3, 1))
    # PT conv weight (out, in, kH, kW) -> jax (kH, kW, in, out).
    kernel = jnp.transpose(
        w["embeddings.patch_embeddings.projection.weight"], (2, 3, 1, 0)
    )
    bias = w["embeddings.patch_embeddings.projection.bias"]
    kH = kernel.shape[0]
    out = lax.conv_general_dilated(
        x, kernel,
        window_strides=(kH, kH),  # stride = patch_size for non-overlapping patches
        padding="VALID",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    out = out + bias
    # (B, gh, gw, H) -> (B, num_patches, H).
    B, gh, gw, H = out.shape
    return out.reshape(B, gh * gw, H)


def _self_attn(x, w_prefix, w, num_heads):
    B, S, H = x.shape
    d_head = H // num_heads
    Q = x @ w[f"{w_prefix}.query.weight"].T + w[f"{w_prefix}.query.bias"]
    K = x @ w[f"{w_prefix}.key.weight"].T   + w[f"{w_prefix}.key.bias"]
    V = x @ w[f"{w_prefix}.value.weight"].T + w[f"{w_prefix}.value.bias"]
    Q = Q.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)
    scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / math.sqrt(d_head)
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, S, H)
    return out


def _layer(x, w, layer_idx, num_heads):
    p = f"encoder.layer.{layer_idx}"
    # Self-attention sublayer (pre-norm).
    h = _layer_norm(x,
                    w[f"{p}.layernorm_before.weight"],
                    w[f"{p}.layernorm_before.bias"])
    a = _self_attn(h, f"{p}.attention.attention", w, num_heads)
    a = a @ w[f"{p}.attention.output.dense.weight"].T \
          + w[f"{p}.attention.output.dense.bias"]
    x = x + a

    # FFN sublayer (pre-norm).
    h = _layer_norm(x,
                    w[f"{p}.layernorm_after.weight"],
                    w[f"{p}.layernorm_after.bias"])
    h = h @ w[f"{p}.intermediate.dense.weight"].T + w[f"{p}.intermediate.dense.bias"]
    h = _gelu_exact(h)
    h = h @ w[f"{p}.output.dense.weight"].T + w[f"{p}.output.dense.bias"]
    return x + h


def compute(inputs):
    config = load_pt_config(HERE / "pt_weights")
    num_heads = config["num_attention_heads"]
    num_layers = config["num_hidden_layers"]

    w = {k: jnp.asarray(v) for k, v in load_pt_safetensors(HERE / "pt_weights").items()}

    pixel_values = jnp.asarray(inputs["pixel_values"])
    patches = _patch_embed(pixel_values, w)            # (B, num_patches, H)
    B = patches.shape[0]

    cls = w["embeddings.cls_token"]                    # (1, 1, H)
    cls = jnp.broadcast_to(cls, (B, 1, cls.shape[-1]))
    x = jnp.concatenate([cls, patches], axis=1)        # (B, num_patches+1, H)
    x = x + w["embeddings.position_embeddings"]        # (1, num_patches+1, H)

    for layer_idx in range(num_layers):
        x = _layer(x, w, layer_idx, num_heads)
    x = _layer_norm(x, w["layernorm.weight"], w["layernorm.bias"])

    return {"last_hidden_state": np.asarray(x)}


if __name__ == "__main__":
    inputs = dict(np.load(HERE / "inputs.npz"))
    out = compute(inputs)
    print("last_hidden_state shape:", out["last_hidden_state"].shape)
    print("checksum:", float(out["last_hidden_state"].sum()))
