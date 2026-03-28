import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Define the Encoder
class Encoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        embedding = nn.Embed(input_dim, embed_dim)(x)
        outputs, (hidden, cell) = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)(embedding)
        return outputs, (hidden, cell)

# Define the Decoder with Attention
class Decoder(nn.Module):
    @nn.compact
    def __call__(self, x, encoder_outputs, hidden, cell):
        x = x.unsqueeze(1)  # Add sequence dimension
        embedded = nn.Embed(output_dim, embed_dim)(x)

        # Attention mechanism
        attention_weights = nn.softmax(nn.Linear(hidden_dim + embed_dim, src_seq_length)(jnp.concatenate((embedded.squeeze(1), hidden[-1]), axis=1)), axis=1)
        context_vector = jnp.einsum('lbh,bl->bh', attention_weights.unsqueeze(1), encoder_outputs)

        # Combine context and embedded input
        combined = jnp.tanh(nn.Linear(hidden_dim + embed_dim, embed_dim)(jnp.concatenate((embedded.squeeze(1), context_vector.squeeze(1)), axis=1)).unsqueeze(1)

        # LSTM and output
        lstm_out, (hidden, cell) = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)(combined, (hidden, cell))
        output = nn.Dense(output_dim)(lstm_out.squeeze(1))
        return output, hidden, cell

# Define synthetic training data
src_vocab_size = 20
tgt_vocab_size = 20
src_seq_length = 10
tgt_seq_length = 12
batch_size = 16

src_data = jnp.random.randint(0, src_vocab_size, (batch_size, src_seq_length))
tgt_data = jnp.random.randint(0, tgt_vocab_size, (batch_size, tgt_seq_length))

# Initialize models, loss function, and optimizer
input_dim = src_vocab_size
output_dim = tgt_vocab_size
embed_dim = 32
hidden_dim = 64
num_layers = 2

encoder = Encoder()
decoder = Decoder()

loss_fn = nn.CrossEntropyLoss()
optimizer = optax.adam(learning_rate=0.001)

# Training loop
@jax.jit
def train_step(params, batch):
    src_data, tgt_data = batch
    params = optax.apply_updates(params, jax.value_and_grad(loss_fn)(params, src_data, tgt_data))
    return params

@jax.jit
def predict(params, x):
    encoder_outputs, (hidden, cell) = encoder(x)
    decoder_input = jnp.zeros(1, dtype=jnp.int32)  # Start token
    output_sequence = []

    for _ in range(tgt_seq_length):
        output, hidden, cell = decoder(decoder_input, encoder_outputs, hidden, cell)
        predicted = jnp.argmax(output, axis=1)
        output_sequence.append(predicted.item())
        decoder_input = predicted

    return output_sequence

for epoch in range(epochs):
    params = train_step(params, (src_data, tgt_data))

    # Log progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss_fn(params, src_data, tgt_data):.4f}")

# Test the sequence-to-sequence model with new input
test_input = jnp.random.randint(0, src_vocab_size, (1, src_seq_length))
output_sequence = predict(params, test_input)

print(f"Input: {test_input.tolist()}, Output: {output_sequence}")