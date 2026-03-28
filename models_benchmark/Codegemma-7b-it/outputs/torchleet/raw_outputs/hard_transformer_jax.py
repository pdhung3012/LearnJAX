import jax
import jax.numpy as jnp
from jax.example_libraries import flax
from flax import linen as nn
import optax

# Define the model architecture
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len).reshape(-1, 1)
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))

        pe[:, 0::2] = jnp.sin(position * div_term)  # Even indices
        pe[:, 1::2] = jnp.cos(position * div_term)  # Odd indices

        pe = pe.reshape(1, max_len, d_model)
        self.pe = flax.nn.Embed(pe)

    def __call__(self, x):
        return x + self.pe[:, :x.shape[1]]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Dense(3 * embed_dim)
        self.out_proj = nn.Dense(embed_dim)

    def __call__(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)  # Shape: (B, T, 3*D)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, T, head_dim)

        scores = (q @ k.transpose(-2, -1)) / jnp.sqrt(self.head_dim)  # (B, num_heads, T, T)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = attn_weights @ v  # (B, num_heads, T, head_dim)

        attn_output = attn_output.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(attn_output)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.linear1 = nn.Dense(ff_dim)
        self.relu = nn.relu
        self.linear2 = nn.Dense(embed_dim)

    def __call__(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, output_dim):
        super().__init__()

        self.cls_token = flax.nn.Embed(jnp.zeros((1, embed_dim)))

        self.embedding = nn.Embed(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Dense(output_dim)

    def __call__(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Prepend CLS token to the sequence
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = jnp.concatenate((cls_tokens, x), axis=1)

        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x[:, 0])  # Use CLS token for classification

# Generate random data for training
num_samples = 10000
seq_length = 10
vocab_size = seq_length  # tokens are 0..N-1

def create_mirror_data(num_samples, seq_length, vocab_size):
    half_len = seq_length // 2
    X = jnp.zeros((num_samples, seq_length), dtype=jnp.int32)
    y = jnp.zeros(num_samples, dtype=jnp.int32)

    for i in range(num_samples):
        # 1. Always create a mirror sequence first
        base_half = jnp.random.randint(0, vocab_size, (half_len,))
        mirror_seq = jnp.concatenate([base_half, jnp.flip(base_half, axis=0)])

        if jnp.random.rand() > 0.5:
            # Positive Case: Keep the mirror order
            X[i] = mirror_seq
            y[i] = 1
        else:
            # Negative Case: Randomly shuffle the mirror sequence
            # Now the tokens are the same, but the order is "broken"
            X[i] = mirror_seq[jnp.random.permutation(seq_length)]
            y[i] = 0

    return X, y

X, y = create_mirror_data(num_samples, seq_length, vocab_size)
X_test, y_test = create_mirror_data(1000, seq_length, vocab_size)

# Training hyperparameters
embed_dim = 64
num_heads = 4
num_layers = 2
ff_dim = 128

model = TransformerModel(vocab_size, embed_dim, num_heads, num_layers, ff_dim, output_dim=2)

# Define loss and optimizer
criterion = optax.softmax_cross_entropy_with_logits
optimizer = optax.adam(learning_rate=2e-4)

# Training loop
for epoch in range(10):
    avg_loss = 0.0
    for i in range(0, num_samples, 64):
        X_batch = X[i:i+64]
        y_batch = y[i:i+64]

        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        avg_loss += loss.mean().item()

        # Backward pass and optimization
        optimizer.update(loss, model.params)
        model = optax.apply_updates(model, optimizer.updates)

    model.eval()
    with jax.no_grad():
        test_predictions = model(X_test)
        test_loss = criterion(test_predictions, y_test).mean().item()
        accuracy = (jnp.argmax(test_predictions, axis=-1) == y_test).mean().item()
    model.train()

    print(f"Epoch [{epoch + 1}/{10}], Avg Train Loss: {avg_loss/num_samples/64:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Test the model
test_sequences = [
    jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    jnp.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
]

with jax.no_grad():
    logits = model(test_sequences)
    probabilities = jax.nn.softmax(logits, axis=-1)
    predictions = jnp.argmax(probabilities, axis=-1)

print(f"Raw Logits: {logits.tolist()}")
print(f"Probabilities: {probabilities.tolist()}")
print(f"Predicted Classes (0 negative, 1 positive): {predictions.tolist()}") # [1, 0] (hopefully!)