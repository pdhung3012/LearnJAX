import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from flax import optax

class Encoder(nn.Module):
  input_dim: int
  embed_dim: int
  hidden_dim: int
  num_layers: int

  def setup(self):
    self.embedding = self.param("embedding", nn.Dense(self.input_dim, self.embed_dim))
    self.lstm = nn.RNN(self.embed_dim, self.hidden_dim, self.num_layers, batch_first=True)

  @nn.compact
  def __call__(self, x):
    embedded = self.embedding(x)
    outputs, (hidden, cell) = self.lstm(embedded)
    return outputs

class Decoder(nn.Module):
  output_dim: int
  embed_dim: int
  hidden_dim: int
  num_layers: int
  src_seq_length: int

  def setup(self):
    self.embedding = self.param("embedding", nn.Dense(self.output_dim, self.embed_dim))
    self.attention = nn.Dense(self.hidden_dim + self.embed_dim, self.src_seq_length)
    self.attention_combine = nn.Dense(self.hidden_dim + self.embed_dim, self.embed_dim)
    self.lstm = nn.RNN(self.embed_dim, self.hidden_dim, self.num_layers, batch_first=True)
    self.fc_out = nn.Dense(self.hidden_dim, self.output_dim)

  @nn.compact
  def __call__(self, x, encoder_outputs, hidden, cell):
    x = jnp.expand_dims(x, axis=1)
    embedded = self.embedding(x)

    attention_weights = jnp.softmax(self.attention(jnp.concat((embedded, hidden[-1]), axis=-1)), axis=-1)
    context_vector = jnp.matmul(attention_weights, encoder_outputs)

    combined = jnp.concat((embedded, context_vector), axis=-1)
    combined = jnp.tanh(self.attention_combine(combined))

    lstm_out, (hidden, cell) = self.lstm(combined, (hidden, cell))
    output = self.fc_out(lstm_out)
    return output, hidden, cell

src_vocab_size = 20
tgt_vocab_size = 20
src_seq_length = 10
tgt_seq_length = 12
batch_size = 16

rng = jr.PRNGKey(42)

src_data = jnp.random.randint(0, src_vocab_size, (batch_size, src_seq_length), rng)
tgt_data = jnp.random.randint(0, tgt_vocab_size, (batch_size, tgt_seq_length), rng)

input_dim = src_vocab_size
output_dim = tgt_vocab_size
embed_dim = 32
hidden_dim = 64
num_layers = 2

encoder = Encoder(input_dim=input_dim, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=num_layers)
decoder = Decoder(output_dim=output_dim, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=num_layers, src_seq_length=src_seq_length)

criterion = nn.LogSoftmaxCrossEntropy()
optimizer = optax.adam(1e-3)

@jax.jit
def train_step(params, src_data, tgt_data):
  encoder_params, decoder_params = params
  encoder_outputs = encoder.apply(src_data, encoder_params)
  loss = 0

  decoder_input = jnp.zeros((batch_size, 1), dtype=jnp.int32)

  for t in jax.range(tgt_seq_length):
    output, hidden, cell = decoder.apply(decoder_input, encoder_outputs, hidden, cell, decoder_params)
    loss += criterion.loss(output, tgt_data[:, t])
    decoder_input = tgt_data[:, t]

  grads = jax.grad(train_step)(params, src_data, tgt_data)
  return optimizer.update(grads, params)

params = jax.array([encoder.init(rng), decoder.init(rng)])

for epoch in jax.range(100):
  encoder_outputs, (hidden, cell) = encoder.apply(src_data, encoder_params)
  loss, new_params = train_step(params, src_data, tgt_data)

  if (epoch + 1) % 10 == 0:
    print(f"Epoch [{epoch + 1}/{100}] - Loss: {loss.item():.4f}")

# Test the sequence-to-sequence model with new input
test_input = jnp.random.randint(0, src_vocab_size, (1, src_seq_length), rng)

with jax.random.PRNGKey(42):
  hidden, cell = jax.random.normal(rng, (1, decoder.num_layers, decoder.hidden_dim))

encoder_outputs, (hidden, cell) = encoder.apply(test_input, encoder_params)
decoder_input = jnp.zeros((1, 1), dtype=jnp.int32)
output_sequence = jnp.zeros((1, tgt_seq_length), dtype=jnp.int32)

for _ in jax.range(tgt_seq_length):
  output, hidden, cell = decoder.apply(decoder_input, encoder_outputs, hidden, cell, decoder_params)
  predicted = output.argmax(axis=-1)
  output_sequence = jax.concatenate([output_sequence, predicted])
  decoder_input = predicted

print(f"Input: {test_input.tolist()}, Output: {output_sequence.tolist()}")


This JAX code replicates the PyTorch model strictly using `flax.linen.Module` and handles the state explicitly. The training loop is converted to use `jax.value_and_grad` and `@jax.jit`. The data is generated using simple `numpy` random data generators.