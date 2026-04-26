"""JAX translation of h5: seq2seq with additive attention (LSTM encoder/decoder).

Faithful to PyTorch:
- Encoder: Embedding(20, 32) -> 2-layer LSTM(32, 64).
- Decoder: per timestep — embed input token (32-d), compute attention weights as
  softmax(Linear(64+32 -> src_seq_length)([embed, hidden[-1]])), context =
  attn @ encoder_outputs, combine = tanh(Linear(64+32 -> 32)([embed, ctx])),
  feed combined (with seq_len 1) into 2-layer LSTM with the rolled hidden/cell,
  fc_out: Linear(64 -> 20).
- Loss: sum of CrossEntropy across all 12 target timesteps (teacher forcing
  uses the previous target token as next input). Adam(lr=1e-3), 100 epochs,
  log every 10.

JAX implementation:
- We use flax.linen for everything. The decoder unroll is a Python loop over
  tgt_seq_length (12) — small enough that we don't bother with scan; jit
  unrolls it once.

Speed notes: JAX is typically faster on CPU here because the per-step Python
overhead in the PyTorch decoder loop is significant; jit fuses it.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


class Encoder(nn.Module):
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        embedded = nn.Embed(self.vocab_size, self.embed_dim)(x)
        # Stack of LSTM layers; each is a scanned LSTMCell.
        cur = embedded
        hiddens, cells = [], []
        for i in range(self.num_layers):
            cell = nn.OptimizedLSTMCell(features=self.hidden_dim, name=f"lstm_{i}")
            rnn = nn.RNN(cell, return_carry=True)
            (h, c), cur = rnn(cur)
            hiddens.append(h)
            cells.append(c)
        # Stack hidden/cell across layers: (num_layers, B, hidden).
        h_stack = jnp.stack(hiddens, axis=0)
        c_stack = jnp.stack(cells, axis=0)
        return cur, (h_stack, c_stack)


class Decoder(nn.Module):
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_layers: int
    src_seq_length: int

    @nn.compact
    def __call__(self, token, encoder_outputs, hidden, cell):
        # token: (B,), int. encoder_outputs: (B, src_seq_length, hidden_dim).
        # hidden, cell: (num_layers, B, hidden_dim).
        embedded = nn.Embed(self.vocab_size, self.embed_dim, name="embedding")(token)
        # Attention.
        attn_in = jnp.concatenate([embedded, hidden[-1]], axis=-1)
        attn_logits = nn.Dense(self.src_seq_length, name="attn")(attn_in)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)  # (B, src_seq)
        # Context = attn @ encoder_outputs -> (B, hidden_dim).
        context = jnp.einsum("bs,bsh->bh", attn_weights, encoder_outputs)

        combined = jnp.concatenate([embedded, context], axis=-1)
        combined = jnp.tanh(nn.Dense(self.embed_dim, name="attn_combine")(combined))

        # Run through layered LSTM, one step.
        new_h, new_c = [], []
        h_in = combined
        for i in range(self.num_layers):
            cell_mod = nn.OptimizedLSTMCell(features=self.hidden_dim, name=f"lstm_{i}")
            (h_i, c_i), h_in = cell_mod((hidden[i], cell[i]), h_in)
            new_h.append(h_i)
            new_c.append(c_i)
        new_h = jnp.stack(new_h, axis=0)
        new_c = jnp.stack(new_c, axis=0)

        out = nn.Dense(self.vocab_size, name="fc_out")(h_in)
        return out, new_h, new_c


def main():
    key = jax.random.PRNGKey(42)
    src_vocab_size = tgt_vocab_size = 20
    src_seq_length = 10
    tgt_seq_length = 12
    batch_size = 16

    key, k1, k2 = jax.random.split(key, 3)
    src_data = jax.random.randint(k1, (batch_size, src_seq_length), 0, src_vocab_size)
    tgt_data = jax.random.randint(k2, (batch_size, tgt_seq_length), 0, tgt_vocab_size)

    embed_dim = 32
    hidden_dim = 64
    num_layers = 2

    encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, num_layers)
    decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, num_layers, src_seq_length)

    key, ke, kd = jax.random.split(key, 3)
    enc_params = encoder.init(ke, src_data)
    # For decoder init we need a dummy hidden/cell.
    init_h = jnp.zeros((num_layers, batch_size, hidden_dim))
    init_c = jnp.zeros((num_layers, batch_size, hidden_dim))
    init_token = jnp.zeros((batch_size,), dtype=jnp.int32)
    init_enc_out = jnp.zeros((batch_size, src_seq_length, hidden_dim))
    dec_params = decoder.init(kd, init_token, init_enc_out, init_h, init_c)

    params = {"enc": enc_params, "dec": dec_params}
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    def loss_fn(params, src_data, tgt_data):
        enc_out, (hidden, cell) = encoder.apply(params["enc"], src_data)
        decoder_input = jnp.zeros((batch_size,), dtype=jnp.int32)
        loss = 0.0
        for t in range(tgt_seq_length):
            output, hidden, cell = decoder.apply(
                params["dec"], decoder_input, enc_out, hidden, cell
            )
            loss = loss + optax.softmax_cross_entropy_with_integer_labels(
                output, tgt_data[:, t]
            ).sum()
            decoder_input = tgt_data[:, t]
        return loss

    @jax.jit
    def train_step(params, opt_state, src_data, tgt_data):
        loss, grads = jax.value_and_grad(loss_fn)(params, src_data, tgt_data)
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    epochs = 100
    for epoch in range(epochs):
        params, opt_state, loss = train_step(params, opt_state, src_data, tgt_data)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss:.4f}")

    # Greedy decode for a single test sequence.
    key, kt = jax.random.split(key)
    test_input = jax.random.randint(kt, (1, src_seq_length), 0, src_vocab_size)
    enc_out, (hidden, cell) = encoder.apply(params["enc"], test_input)
    decoder_input = jnp.zeros((1,), dtype=jnp.int32)
    output_sequence = []
    for _ in range(tgt_seq_length):
        out, hidden, cell = decoder.apply(
            params["dec"], decoder_input, enc_out, hidden, cell
        )
        predicted = jnp.argmax(out, axis=-1)
        output_sequence.append(int(predicted[0]))
        decoder_input = predicted
    print(f"Input: {test_input.tolist()}, Output: {output_sequence}")


if __name__ == "__main__":
    main()
