import jax
import jax.numpy as jnp
import optax
import flax
from flax import linen as nn

class LanguageModel(nn.Module):
  @nn.compact
  def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
    self.embedding = self.param("embedding", nn.Dense(vocab_size, embed_size))
    self.lstm = nn.RNN(
        self.lstm_cell, (hidden_size, hidden_size), name="lstm")
    self.fc = nn.Dense(vocab_size)

  @nn.compact
  def lstm_cell(self, inputs):
    i, j = jax.split(inputs, 2)
    tanh = jax.nn.tanh
    return jax.nn.rnn_cell_dropout(
        jax.nn.LSTMCell(hidden_size), 0.2)(i, j)

  @nn.compact
  def __call__(self, carry, inputs):
    embedded = self.embedding(inputs)
    lstm_out, _ = self.lstm(carry, embedded)
    output = self.fc(lstm_out)
    return output, (lstm_out, carry)

vocab_size = 50
seq_length = 10
batch_size = 32

rng = jax.random.PRNGKey(42)
X_train = jnp.random.randint(0, vocab_size, (batch_size, seq_length))
y_train = jnp.random.randint(0, vocab_size, (batch_size,))

class TrainStep(flax.training.TrainingStep):
  def __init__(self, opt_state, model):
    super().__init__()
    self.params = model.params
    self.opt_state = opt_state
    self.model = model

  @property
  def grad_fn(self):
    return self.model.value_and_grad

  def update(self, carry, grads):
    updates, new_opt_state = optax.update(self.opt_state, self.params, grads)
    return carry, updates, new_opt_state

model = LanguageModel(vocab_size, 64, 128, 2)
rng, rng_key = jax.random.split(rng)
params = model.init(rng_key, jnp.ones((batch_size, seq_length)))
opt_state = optax.init_optimizer(optax.adam, params)

@jax.jit
def train_step(carry, X_batch, y_batch):
  grads = model.grad(carry, X_batch)
  updates, opt_state = TrainStep(opt_state, model).update(carry, grads)
  return updates, opt_state

@jax.jit
def train(epochs, X_train, y_train):
  carry = model.init_rnn_state(jnp.ones((batch_size, hidden_size)))
  for epoch in range(epochs):
    for i in range(len(X_train) // batch_size):
      X_batch = X_train[i * batch_size : (i + 1) * batch_size]
      y_batch = y_train[i * batch_size : (i + 1) * batch_size]
      carry, updates, opt_state = train_step(carry, X_batch, y_batch)
      model = flax.linen.apply_updates(model, updates)
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {jnp.mean(model.loss(X_train, y_train)).item():.4f}")

train(epochs=5, X_train=X_train, y_train=y_train)


This JAX code replicates the PyTorch code strictly using `flax.linen.Module`, `jax.numpy`, `optax`, and `jax`. The training loop is converted to use `jax.value_and_grad` and `@jax.jit`. The data is generated using simple `numpy` random data generators.