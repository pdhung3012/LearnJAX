import torch
import torch.nn as nn
import torch.optim as optim
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)  # Shape: (B, T, 3*D)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, T, head_dim)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, T, T)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = attn_weights @ v  # (B, num_heads, T, head_dim)

        attn_output = attn_output.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, output_dim):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        # Prepend CLS token to the sequence
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x[:, 0])  # Use CLS token for classification

torch.manual_seed(42)

seq_length = 10
num_samples = 10000
vocab_size = seq_length  # tokens are 0..N-1

def create_mirror_data(num_samples, seq_length, vocab_size):
    half_len = seq_length // 2
    X = torch.zeros((num_samples, seq_length), dtype=torch.long)
    y = torch.zeros(num_samples, dtype=torch.long)

    for i in range(num_samples):
        # 1. Always create a mirror sequence first
        base_half = torch.randint(0, vocab_size, (half_len,))
        mirror_seq = torch.cat([base_half, torch.flip(base_half, dims=[0])])

        if torch.rand(1) > 0.5:
            # Positive Case: Keep the mirror order
            X[i] = mirror_seq
            y[i] = 1
        else:
            # Negative Case: Randomly shuffle the mirror sequence
            # Now the tokens are the same, but the order is "broken"
            X[i] = mirror_seq[torch.randperm(seq_length)]
            y[i] = 0

    return X, y

X, y = create_mirror_data(num_samples, seq_length, vocab_size)
X_test, y_test = create_mirror_data(1000, seq_length, vocab_size)


embed_dim = 64
num_heads = 4
num_layers = 2
ff_dim = 128

model = TransformerModel(vocab_size, embed_dim, num_heads, num_layers, ff_dim, output_dim=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)

epochs = 10
batch_size = 64
for epoch in range(epochs):
    avg_loss = 0.0
    for i in range(0, num_samples, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        avg_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = criterion(test_predictions, y_test)
        _, predicted_classes = torch.max(test_predictions, 1)
        accuracy = (predicted_classes == y_test).float().mean().item()
    model.train()

    print(f"Epoch [{epoch + 1}/{epochs}], Avg Train Loss: {avg_loss/(num_samples//batch_size):.4f}, Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy:.4f}")

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 1. Load a small subset of the SST-2 dataset
dataset = load_dataset("glue", "sst2", split="train[:5000]")

# 2. Tokenization (Turning words into numbers)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=16)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'label'])

# 3. Create DataLoader
train_loader = DataLoader(tokenized_dataset, batch_size=32, shuffle=True)

# Updated Hyperparameters for Sentiment Analysis
vocab_size = tokenizer.vocab_size  # Usually ~30,522 for BERT
embed_dim = 32                     # Small embedding for speed
num_heads = 4
num_layers = 2
ff_dim = 128
output_dim = 2                     # 0 for Negative, 1 for Positive

# Initialize the model with the new vocab size
# Note: You'll need to update your TransformerModel class to include an nn.Embedding layer
model = TransformerModel(vocab_size, embed_dim, num_heads, num_layers, ff_dim, output_dim)

# Use CrossEntropyLoss for classification instead of MSELoss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
model.train()
epochs = 50
for epoch in range(epochs):
    avg_loss = 0.0
    for batch in train_loader:
        X = batch['input_ids']  # Shape: (batch_size, seq_length)
        y = batch['label']      # Shape: (batch_size,)

        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)
        avg_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss/len(train_loader):.4f}")

# 1. Put model in evaluation mode (disables dropout/batchnorm)
model.eval()

test_sequences = [
    "This was the worst film I have ever seen.",
    "I absolutely loved this movie!"
]

# 2. Tokenize (ensure return_tensors="pt" for PyTorch)
tokenized_test = tokenizer(test_sequences, padding="max_length", truncation=True, max_length=16, return_tensors="pt")
X_test = tokenized_test['input_ids']  # Shape: (2, seq_length)
with torch.no_grad():
    logits = model(X_test)
    probabilities = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)

print(f"Raw Logits: {logits.tolist()}")
print(f"Probabilities: {probabilities.tolist()}")
print(f"Predicted Classes (0 negative, 1 positive): {predictions.tolist()}") # [1, 0] (hopefully!)
