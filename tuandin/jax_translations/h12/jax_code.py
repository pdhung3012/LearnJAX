"""JAX translation of h12: SmolLM-135M from scratch (Llama-style decoder LM).

Faithful to PyTorch:
- Same architecture: 30x LlamaDecoder, hidden=576, intermediate=1536,
  num_heads=9, kv_heads=3 (so 3:1 GQA), RoPE with base=10000, RMSNorm eps=1e-5,
  SwiGLU MLP, tied embedding/lm_head.

Notes:
- Causal mask is built once and added to attention scores (matching the
  PyTorch implementation).
- For the test we just do a forward on a 4-token prompt; both versions return
  logits of shape (1, 4, 49152). Random init only (no checkpoint loaded).

Speed:
- For a 4-token prompt the runtime is dominated by single-step matmuls; jit
  compile is one-time. JAX is competitive with PyTorch on CPU; on GPU/TPU the
  compiled GQA + RoPE attention typically beats PyTorch eager.
"""
import math
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn


# ---- Contract API used by test_equivalence.py ------------------------------
def compute(inputs):
    """SmolLM building blocks: RMSNorm + SwiGLU MLP + rotate_half.

    Inputs:
        x (B, S, H), weight (H,)         — RMSNorm input + scale
        W_gate (I, H), W_up (I, H), W_down (H, I)  — MLP weights (PT layout)
        z (B, S, H)                       — rotate_half input
        eps (0-d float)
    Returns: rms_norm, swiglu_mlp, rotate_half (each same shape as x).
    """
    eps = float(inputs["eps"])
    x = jnp.asarray(inputs["x"])
    var = jnp.mean(x ** 2, axis=-1, keepdims=True)
    rms = x * jax.lax.rsqrt(var + eps)
    rms_out = jnp.asarray(inputs["weight"]) * rms

    W_gate = jnp.asarray(inputs["W_gate"])
    W_up   = jnp.asarray(inputs["W_up"])
    W_down = jnp.asarray(inputs["W_down"])
    swish = jax.nn.silu(x @ W_gate.T)
    mlp_out = (swish * (x @ W_up.T)) @ W_down.T

    z = jnp.asarray(inputs["z"])
    half = z.shape[-1] // 2
    rh = jnp.concatenate([-z[..., half:], z[..., :half]], axis=-1)

    return {
        "rms_norm":    np.asarray(rms_out),
        "swiglu_mlp":  np.asarray(mlp_out),
        "rotate_half": np.asarray(rh),
    }


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = jnp.expand_dims(cos, unsqueeze_dim)
    sin = jnp.expand_dims(sin, unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x, n_rep):
    """(B, H_kv, S, D) -> (B, H_kv * n_rep, S, D)."""
    b, h_kv, s, d = x.shape
    x = x[:, :, None, :, :]
    x = jnp.broadcast_to(x, (b, h_kv, n_rep, s, d))
    return x.reshape(b, h_kv * n_rep, s, d)


def make_rope_cos_sin(seq_len, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.einsum("p,f->pf", pos, inv_freq)[None, :, :]    # (1, S, D/2)
    emb = jnp.concatenate([angles, angles], axis=-1)             # (1, S, D)
    return jnp.cos(emb), jnp.sin(emb)


class RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, h):
        weight = self.param("weight", nn.initializers.ones, (self.hidden_size,))
        var = jnp.mean(h ** 2, axis=-1, keepdims=True)
        h = h * jax.lax.rsqrt(var + self.eps)
        return weight * h


class MLP(nn.Module):
    hidden_size: int
    intermediate_size: int

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.intermediate_size, use_bias=False, name="W_gate")(x)
        up   = nn.Dense(self.intermediate_size, use_bias=False, name="W_up")(x)
        return nn.Dense(self.hidden_size, use_bias=False, name="W_down")(jax.nn.silu(gate) * up)


class RopeAttention(nn.Module):
    hidden_size: int
    num_heads: int
    kv_heads: int
    rope_theta: float = 10000.0

    @nn.compact
    def __call__(self, hidden_states, attention_mask=None):
        b, s, _ = hidden_states.shape
        head_dim = self.hidden_size // self.num_heads

        Q = nn.Dense(self.num_heads * head_dim, use_bias=False, name="W_query")(hidden_states)
        K = nn.Dense(self.kv_heads * head_dim,  use_bias=False, name="W_key")(hidden_states)
        V = nn.Dense(self.kv_heads * head_dim,  use_bias=False, name="W_value")(hidden_states)
        Q = Q.reshape(b, s, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(b, s, self.kv_heads,  head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(b, s, self.kv_heads,  head_dim).transpose(0, 2, 1, 3)

        cos, sin = make_rope_cos_sin(s, head_dim, self.rope_theta)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        n_rep = self.num_heads // self.kv_heads
        K = repeat_kv(K, n_rep)
        V = repeat_kv(V, n_rep)

        attn = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) / math.sqrt(head_dim)
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.matmul(attn, V)  # (B, H, S, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, -1)
        return nn.Dense(self.hidden_size, use_bias=False, name="W_output")(out)


class LlamaDecoder(nn.Module):
    hidden_size: int
    intermediate_size: int
    num_heads: int
    kv_heads: int

    @nn.compact
    def __call__(self, hidden_states, attention_mask):
        residual = hidden_states
        h = RMSNorm(self.hidden_size, name="pre_attn_rmsnorm")(hidden_states)
        s = h.shape[1]
        causal = jnp.triu(jnp.full((s, s), -jnp.inf), k=1)
        h = RopeAttention(self.hidden_size, self.num_heads, self.kv_heads,
                          name="self_attn")(h, attention_mask=causal)
        h = h + residual

        residual = h
        h = RMSNorm(self.hidden_size, name="pre_mlp_rmsnorm")(h)
        h = MLP(self.hidden_size, self.intermediate_size, name="mlp")(h)
        return h + residual


class SmolModel(nn.Module):
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_heads: int
    kv_heads: int

    @nn.compact
    def __call__(self, input_ids, attention_mask):
        embed_tokens = nn.Embed(self.vocab_size, self.hidden_size, name="embed_tokens")
        h = embed_tokens(input_ids)
        for i in range(self.num_hidden_layers):
            h = LlamaDecoder(self.hidden_size, self.intermediate_size, self.num_heads,
                             self.kv_heads, name=f"layer_{i}")(h, attention_mask)
        h = RMSNorm(self.hidden_size, name="norm")(h)
        return h, embed_tokens.embedding


class SmolLM(nn.Module):
    vocab_size: int = 49152
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_heads: int = 9
    kv_heads: int = 3

    @nn.compact
    def __call__(self, input_ids, attention_mask):
        h, embed_w = SmolModel(self.vocab_size, self.hidden_size, self.intermediate_size,
                                self.num_hidden_layers, self.num_heads, self.kv_heads,
                                name="model")(input_ids, attention_mask)
        # Tied lm_head: logits = h @ embed_w^T.
        logits = h @ embed_w.T
        return {"logits": logits.astype(jnp.float32)}


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    model = SmolLM()
    input_ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)
    params = model.init(key, input_ids, attention_mask)
    out = model.apply(params, input_ids, attention_mask)
    print("logits shape:", out["logits"].shape)
    assert out["logits"].shape == (1, 4, model.vocab_size)
