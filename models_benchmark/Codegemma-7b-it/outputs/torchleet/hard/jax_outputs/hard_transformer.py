import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import math

class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000

    @nn.compact
    def __call__(self, x):
        position = jnp.arange(self.max_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model))
        pe = jnp.zeros((self.max_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))  # Even indices
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))  # Odd indices
        seq_len = x.shape[1]
        return x + pe[None, :seq_len, :]


class MultiHeadSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int

    def setup(self):
        assert self.embed_dim % self.num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = self.embed_dim // self.num_heads
        self.qkv_proj = nn.Dense(3 * self.embed_dim)
        self.out_proj = nn.Dense(self.embed_dim)

    def __call__(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)  # Shape: (B, T, 3*D)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, T, head_dim)

        scores = (q @ k.swapaxes(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, T, T)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = attn_weights @ v  # (B, num_heads, T, head_dim)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    embed_dim: int
    ff_dim: int

    def setup(self):
        self.linear1 = nn.Dense(self.ff_dim)
        self.linear2 = nn.Dense(self.embed_dim)

    def __call__(self, x):
        return self.linear2(nn.relu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    embed_dim: int
    num_heads: int
    ff_dim: int

    def setup(self):
        self.attn = MultiHeadSelfAttention(embed_dim=self.embed_dim, num_heads=self.num_heads)
        self.ffn = FeedForward(embed_dim=self.embed_dim, ff_dim=self.ff_dim)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerModel(nn.Module):
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    output_dim: int

    def setup(self):
        self.cls_token = self.param('cls_token', nn.initializers.zeros, (1, 1, self.embed_dim))
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)
        self.pos_encoding = PositionalEncoding(d_model=self.embed_dim)
        self.layers = [
            TransformerEncoderLayer(embed_dim=self.embed_dim, num_heads=self.num_heads, ff_dim=self.ff_dim)
            for _ in range(self.num_layers)
        ]
        self.output_proj = nn.Dense(self.output_dim)

    def __call__(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Prepend CLS token to the sequence
        cls_tokens = jnp.broadcast_to(self.cls_token, (x.shape[0], 1, self.embed_dim))
        x = jnp.concatenate((cls_tokens, x), axis=1)

        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x[:, 0])  # Use CLS token for classification

key = jax.random.PRNGKey(42)

seq_length = 10
num_samples = 10000
vocab_size = seq_length  # tokens are 0..N-1

def create_mirror_data(key, num_samples, seq_length, vocab_size):
    half_len = seq_length // 2
    key, k1, k2, k3 = jax.random.split(key, 4)

    # Generate base halves: (num_samples, half_len)
    base_halves = jax.random.randint(k1, (num_samples, half_len), minval=0, maxval=vocab_size)
    # Create mirror sequences: (num_samples, seq_length)
    mirror_seqs = jnp.concatenate([base_halves, jnp.flip(base_halves, axis=1)], axis=1)

    # Decide positive vs negative
    is_positive = jax.random.uniform(k2, (num_samples,)) > 0.5

    # Generate permutation indices for negative samples
    perm_keys = jax.random.split(k3, num_samples)
    perms = jax.vmap(lambda k: jax.random.permutation(k, seq_length))(perm_keys)

    # Shuffle sequences for negative samples
    shuffled_seqs = jax.vmap(lambda seq, perm: seq[perm])(mirror_seqs, perms)

    # Select between mirror and shuffled
    X = jnp.where(is_positive[:, None], mirror_seqs, shuffled_seqs)
    y = is_positive.astype(jnp.int32)

    return X, y

key, data_key1, data_key2 = jax.random.split(key, 3)
X, y = create_mirror_data(data_key1, num_samples, seq_length, vocab_size)
X_test, y_test = create_mirror_data(data_key2, 1000, seq_length, vocab_size)


embed_dim = 64
num_heads = 4
num_layers = 2
ff_dim = 128

model = TransformerModel(
    vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads,
    num_layers=num_layers, ff_dim=ff_dim, output_dim=2
)

# Initialize parameters
key, init_key = jax.random.split(key)
dummy_input = jnp.zeros((1, seq_length), dtype=jnp.int32)
variables = model.init(init_key, dummy_input)
params = variables['params']

# Optimizer
optimizer = optax.adam(learning_rate=2e-4)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, X_batch, y_batch):
    def loss_fn(params):
        predictions = model.apply({'params': params}, X_batch)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions, y_batch))
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def eval_step(params, X_test, y_test):
    predictions = model.apply({'params': params}, X_test)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions, y_test))
    predicted_classes = jnp.argmax(predictions, axis=-1)
    accuracy = jnp.mean((predicted_classes == y_test).astype(jnp.float32))
    return loss, accuracy

epochs = 10
batch_size = 64
for epoch in range(epochs):
    avg_loss = 0.0
    for i in range(0, num_samples, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        # Forward pass, backward pass, and optimization
        params, opt_state, loss = train_step(params, opt_state, X_batch, y_batch)
        avg_loss += loss.item()

    test_loss, accuracy = eval_step(params, X_test, y_test)

    print(f"Epoch [{epoch + 1}/{epochs}], Avg Train Loss: {avg_loss/(num_samples//batch_size):.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# 1. Load a small subset of the SST-2 dataset
dataset = load_dataset("glue", "sst2", split="train[:5000]")

# 2. Tokenization (Turning words into numbers)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=16)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Convert to JAX arrays
train_X = jnp.array(tokenized_dataset['input_ids'])
train_y = jnp.array(tokenized_dataset['label'])

# Updated Hyperparameters for Sentiment Analysis
vocab_size_sst = tokenizer.vocab_size  # Usually ~30,522 for BERT
embed_dim_sst = 32                     # Small embedding for speed
num_heads_sst = 4
num_layers_sst = 2
ff_dim_sst = 128
output_dim_sst = 2                     # 0 for Negative, 1 for Positive

# Initialize the model with the new vocab size
model_sst = TransformerModel(
    vocab_size=vocab_size_sst, embed_dim=embed_dim_sst, num_heads=num_heads_sst,
    num_layers=num_layers_sst, ff_dim=ff_dim_sst, output_dim=output_dim_sst
)

key, init_key = jax.random.split(key)
dummy_input_sst = jnp.zeros((1, 16), dtype=jnp.int32)
variables_sst = model_sst.init(init_key, dummy_input_sst)
params_sst = variables_sst['params']

# Use CrossEntropyLoss for classification
optimizer_sst = optax.adam(learning_rate=0.0001)
opt_state_sst = optimizer_sst.init(params_sst)

@jax.jit
def train_step_sst(params, opt_state, X_batch, y_batch):
    def loss_fn(params):
        predictions = model_sst.apply({'params': params}, X_batch)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(predictions, y_batch))
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer_sst.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
num_train = len(train_X)
batch_size_sst = 32
epochs_sst = 50
for epoch in range(epochs_sst):
    # Shuffle data each epoch
    key, shuffle_key = jax.random.split(key)
    perm = jax.random.permutation(shuffle_key, num_train)
    train_X_shuffled = train_X[perm]
    train_y_shuffled = train_y[perm]

    avg_loss = 0.0
    num_batches = 0
    for i in range(0, num_train, batch_size_sst):
        X_batch = train_X_shuffled[i:i+batch_size_sst]
        y_batch = train_y_shuffled[i:i+batch_size_sst]

        # Forward pass
        params_sst, opt_state_sst, loss = train_step_sst(params_sst, opt_state_sst, X_batch, y_batch)
        avg_loss += loss.item()
        num_batches += 1

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epochs_sst}], Avg Loss: {avg_loss/num_batches:.4f}")

# 1. Test on sample sentences
test_sequences = [
    "This was the worst film I have ever seen.",
    "I absolutely loved this movie!"
]

# 2. Tokenize
tokenized_test = tokenizer(test_sequences, padding="max_length", truncation=True, max_length=16, return_tensors="np")
X_test_sst = jnp.array(tokenized_test['input_ids'])
logits = model_sst.apply({'params': params_sst}, X_test_sst)
probabilities = jax.nn.softmax(logits, axis=-1)
predictions = jnp.argmax(probabilities, axis=-1)

print(f"Raw Logits: {logits.tolist()}")
print(f"Probabilities: {probabilities.tolist()}")
print(f"Predicted Classes (0 negative, 1 positive): {predictions.tolist()}")
