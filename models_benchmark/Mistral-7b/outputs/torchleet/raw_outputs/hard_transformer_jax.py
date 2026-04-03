import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from flax import struct
from jax import grad, jit, value_and_grad
from optax import adam

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = jnp.zeros((self.max_len, self.d_model))
        position = jnp.arange(0, self.max_len).reshape(1, -1)
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model))

        self.pe[:, 0::2] = jnp.sin(jnp.outer(position, div_term))  # Even indices
        self.pe[:, 1::2] = jnp.cos(jnp.outer(position, div_term))  # Odd indices

    @nn.compact
    def __call__(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]

class MultiHeadSelfAttention(nn.Module):
    def setup(self):
        self.d_model = self.params.d_model
        self.num_heads = self.params.num_heads
        self.head_dim = self.d_model // self.num_heads

        self.qkv_proj = nn.Dense(self.d_model * 3)
        self.out_proj = nn.Dense(self.d_model)

    @nn.compact
    def __call__(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)  # Shape: (B, T, 3*D)
        qkv = jnp.reshape(qkv, (B, T, 3, self.num_heads, self.head_dim))
        qkv = jnp.permute(qkv, (2, 0, 3, 1, 4))  # (B, num_heads, T, head_dim, T)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, T, head_dim)

        scores = jnp.matmul(q, k.transpose((0, 1, 3, 2))) / jnp.sqrt(self.head_dim)  # (B, num_heads, T, T)
        attn_weights = jnp.softmax(scores, axis=-1)
        attn_output = jnp.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)

        attn_output = jnp.reshape(attn_output, (B, T, self.d_model))
        return self.out_proj(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim):
        super().__init__()
        self.d_model = d_model
        self.ff_dim = ff_dim
        self.linear1 = nn.Dense(self.d_model, self.ff_dim)
        self.relu = jnp.tanh
        self.linear2 = nn.Dense(self.ff_dim, self.d_model)

    @nn.compact
    def __call__(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, ff_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    @nn.compact
    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerModel(nn.Module):
    def setup(self):
        self.vocab_size = self.params.vocab_size
        self.embed_dim = self.params.embed_dim
        self.num_heads = self.params.num_heads
        self.num_layers = self.params.num_layers
        self.ff_dim = self.params.ff_dim
        self.output_dim = self.params.output_dim

        self.cls_token = self.param("cls_token", jnp.zeros((1, 1, self.embed_dim)))

        self.embedding = nn.Embed(input_dim=self.vocab_size, output_dim=self.embed_dim)
        self.pos_encoding = PositionalEncoding(self.embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(self.embed_dim, self.num_heads, self.ff_dim)
            for _ in range(self.num_layers)
        ])
        self.output_proj = nn.Dense(self.embed_dim, self.output_dim)

    @nn.compact
    def __call__(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        cls_tokens = jnp.expand_dims(self.cls_token, axis=0)
        x = jnp.concatenate((cls_tokens, x), axis=0)

        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x[:, 0])

class TransformerModelParams(nn.Module):
    def setup(self):
        self.vocab_size = self.params.vocab_size
        self.embed_dim = self.params.embed_dim
        self.num_heads = self.params.num_heads
        self.num_layers = self.params.num_layers
        self.ff_dim = self.params.ff_dim
        self.output_dim = self.params.output_dim

        self.cls_token = self.param("cls_token", jnp.zeros((1, 1, self.embed_dim)))

class TransformerModelInit(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.model = TransformerModelParams(params)

    @nn.compact
    def __call__(self):
        return self.model

class TrainStep(nn.Module):
    def setup(self):
        self.model = self.params.model
        self.optimizer = adam(self.model.init_params())

    @nn.compact
    def __call__(self, X, y):
        with jax.grad():
            logits = self.model(X)
            loss = jnp.mean(jnp.cross_entropy(y, jnp.argmax(logits, axis=-1)))
        grads = jax.value_and_grad(self.model.init_params, has_aux=True)(())[1]
        self.optimizer.apply_update(grads)
        return loss

num_samples = 10000
seq_length = 10
vocab_size = seq_length

params = struct.ParamDict({
    "vocab_size": vocab_size,
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "ff_dim": 128,
    "output_dim": 2,
})

model = TransformerModelInit(params)

X = jnp.zeros((num_samples, seq_length), dtype=jnp.int32)
y = jnp.zeros