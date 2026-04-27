"""JAX translation of h3: small Transformer regression on summing sequences.

Faithful to PyTorch:
- Linear embed (1->16) -> TransformerEncoder(2 layers, 2 heads, ff_dim=64) ->
  mean over seq -> Linear(16->1).
- Target: sum of each input sequence (so y has shape (N, 1)).
- 1000 epochs, MSE, Adam(lr=1e-3), log every 100.

PyTorch's nn.TransformerEncoderLayer defaults: dropout=0.1, activation='relu',
norm_first=False. We replicate those.

Speed notes: a small transformer with these dims is well-suited to JAX +
optax + jit; expect comparable or slightly faster than PyTorch on CPU.
"""
import jax
import jax.numpy as jnp
import numpy as np


# ---- Contract API used by test_equivalence.py ------------------------------
def compute(inputs):
    """LayerNorm + FFN(Linear -> ReLU -> Linear) sub-block forward.

    Inputs:
        x (B, S, D)
        gamma (D,), beta (D,)        — LayerNorm weight/bias
        W1 (FF, D), b1 (FF,)         — FFN first linear (PyTorch nn.Linear: out, in)
        W2 (D, FF), b2 (D,)          — FFN second linear
    Returns: {"layer_norm": (B, S, D), "ffn": (B, S, D)}
    """
    x = jnp.asarray(inputs["x"])
    gamma = jnp.asarray(inputs["gamma"]); beta = jnp.asarray(inputs["beta"])
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    ln_out = (x - mean) / jnp.sqrt(var + 1e-5) * gamma + beta
    h = jax.nn.relu(x @ jnp.asarray(inputs["W1"]).T + jnp.asarray(inputs["b1"]))
    ffn = h @ jnp.asarray(inputs["W2"]).T + jnp.asarray(inputs["b2"])
    return {"layer_norm": np.asarray(ln_out), "ffn": np.asarray(ffn)}
import flax.linen as nn
import optax


class TransformerEncoderLayer(nn.Module):
    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic):
        # Self-attention.
        attn_out = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            dropout_rate=self.dropout,
            deterministic=deterministic,
        )(x)
        attn_out = nn.Dropout(self.dropout, deterministic=deterministic)(attn_out)
        x = nn.LayerNorm()(x + attn_out)
        # Feed-forward.
        ff = nn.Dense(self.ff_dim)(x)
        ff = nn.relu(ff)
        ff = nn.Dropout(self.dropout, deterministic=deterministic)(ff)
        ff = nn.Dense(self.embed_dim)(ff)
        ff = nn.Dropout(self.dropout, deterministic=deterministic)(ff)
        return nn.LayerNorm()(x + ff)


class TransformerModel(nn.Module):
    input_dim: int
    embed_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x, deterministic):
        x = nn.Dense(self.embed_dim)(x)
        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                self.embed_dim, self.num_heads, self.ff_dim
            )(x, deterministic=deterministic)
        x = jnp.mean(x, axis=1)
        return nn.Dense(self.output_dim)(x)


def main():
    key = jax.random.PRNGKey(42)
    seq_length = 10
    num_samples = 100
    input_dim = 1
    key, kx = jax.random.split(key)
    X = jax.random.uniform(kx, (num_samples, seq_length, input_dim))
    y = jnp.sum(X, axis=1)  # (N, 1)

    model = TransformerModel(input_dim=1, embed_dim=16, num_heads=2,
                             num_layers=2, ff_dim=64, output_dim=1)
    key, k_init, k_drop = jax.random.split(key, 3)
    params = model.init({"params": k_init, "dropout": k_drop},
                        X[:1], deterministic=False)
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    def loss_fn(params, x, y, dropout_key):
        pred = model.apply(params, x, deterministic=False,
                           rngs={"dropout": dropout_key})
        return jnp.mean((pred - y) ** 2)

    @jax.jit
    def train_step(params, opt_state, x, y, dropout_key):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y, dropout_key)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    epochs = 1000
    for epoch in range(epochs):
        key, k = jax.random.split(key)
        params, opt_state, loss = train_step(params, opt_state, X, y, k)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

    key, kt = jax.random.split(key)
    X_test = jax.random.uniform(kt, (2, seq_length, input_dim))
    preds = model.apply(params, X_test, deterministic=True)
    print(f"Predictions for {X_test.tolist()}: {preds.tolist()}")


if __name__ == "__main__":
    main()
