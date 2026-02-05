import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Define the Encoder
class Encoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        embedded = nn.Embed(num_embeddings=20, features=32)(x)
        # Layer 1
        carry1, outputs1 = nn.RNN(nn.LSTMCell(features=64), return_carry=True)(embedded)
        # Layer 2
        carry2, outputs2 = nn.RNN(nn.LSTMCell(features=64), return_carry=True)(outputs1)
        return outputs2, (carry1, carry2)

# Define the Decoder with Attention
class Decoder(nn.Module):
    @nn.compact
    def __call__(self, x, encoder_outputs, carry1, carry2):
        embedded = nn.Embed(num_embeddings=20, features=32)(x)

        # Attention mechanism using last layer's hidden state
        # LSTMCell carry convention: (c, h), so carry[1] is the hidden state
        h_last = carry2[1]

        attention_input = jnp.concatenate([embedded, h_last], axis=-1)
        attention_weights = jax.nn.softmax(nn.Dense(10)(attention_input), axis=-1)
        context_vector = jnp.einsum('bi,bij->bj', attention_weights, encoder_outputs)

        # Combine context and embedded input
        combined = jnp.concatenate([embedded, context_vector], axis=-1)
        combined = jnp.tanh(nn.Dense(32)(combined))

        # LSTM layers (single timestep)
        carry1, y1 = nn.LSTMCell(features=64)(carry1, combined)
        carry2, y2 = nn.LSTMCell(features=64)(carry2, y1)

        output = nn.Dense(20)(y2)
        return output, carry1, carry2

# Define synthetic training data
key = jax.random.PRNGKey(42)
src_vocab_size = 20
tgt_vocab_size = 20
src_seq_length = 10
tgt_seq_length = 12
batch_size = 16

src_data = jax.random.randint(key, shape=(batch_size, src_seq_length), minval=0, maxval=src_vocab_size)
tgt_data = jax.random.randint(key, shape=(batch_size, tgt_seq_length), minval=0, maxval=tgt_vocab_size)

# Initialize models
encoder = Encoder()
decoder = Decoder()

# Initialize parameters
key, enc_key, dec_key = jax.random.split(key, 3)
encoder_variables = encoder.init(enc_key, src_data)

# Run encoder to get dummy outputs for decoder initialization
dummy_enc_out, (dummy_carry1, dummy_carry2) = encoder.apply(encoder_variables, src_data)
dummy_dec_input = jnp.zeros((batch_size,), dtype=jnp.int32)
decoder_variables = decoder.init(dec_key, dummy_dec_input, dummy_enc_out, dummy_carry1, dummy_carry2)

# Combine parameters
params = {
    'encoder': encoder_variables['params'],
    'decoder': decoder_variables['params']
}

# Optimizer
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# Loss function
def compute_loss(params, src_data, tgt_data):
    encoder_outputs, (carry1, carry2) = encoder.apply({'params': params['encoder']}, src_data)
    loss = 0.0
    decoder_input = jnp.zeros((src_data.shape[0],), dtype=jnp.int32)  # Start token

    for t in range(tgt_seq_length):
        output, carry1, carry2 = decoder.apply(
            {'params': params['decoder']}, decoder_input, encoder_outputs, carry1, carry2
        )
        loss += jnp.mean(optax.softmax_cross_entropy_with_integer_labels(output, tgt_data[:, t]))
        decoder_input = tgt_data[:, t]  # Teacher forcing

    return loss

# Training step
@jax.jit
def train_step(params, opt_state, src_data, tgt_data):
    loss, grads = jax.value_and_grad(compute_loss)(params, src_data, tgt_data)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
epochs = 100
for epoch in range(epochs):
    params, opt_state, loss = train_step(params, opt_state, src_data, tgt_data)

    # Log progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss:.4f}")

# Test the sequence-to-sequence model with new input
test_input = jax.random.randint(key, shape=(1, src_seq_length), minval=0, maxval=src_vocab_size)
encoder_outputs, (carry1, carry2) = encoder.apply({'params': params['encoder']}, test_input)
decoder_input = jnp.zeros((1,), dtype=jnp.int32)  # Start token
output_sequence = []

for _ in range(tgt_seq_length):
    output, carry1, carry2 = decoder.apply(
        {'params': params['decoder']}, decoder_input, encoder_outputs, carry1, carry2
    )
    predicted = jnp.argmax(output, axis=-1)
    output_sequence.append(predicted[0].item())
    decoder_input = predicted

print(f"Input: {test_input.tolist()}, Output: {output_sequence}")
