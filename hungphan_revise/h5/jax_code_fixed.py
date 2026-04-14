import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
from functools import partial


class Encoder(nn.Module):
    input_dim: int
    embed_dim: int
    hidden_dim: int
    num_layers: int

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.input_dim, features=self.embed_dim)
        self.lstm_cells = [nn.LSTMCell(features=self.hidden_dim) for _ in range(self.num_layers)]

    def __call__(self, x):
        # x: (batch, seq_length)
        embedded = self.embedding(x)  # (batch, seq_length, embed_dim)
        batch, seq_length, _ = embedded.shape

        hidden_states = [jnp.zeros((batch, self.hidden_dim)) for _ in range(self.num_layers)]
        cell_states = [jnp.zeros((batch, self.hidden_dim)) for _ in range(self.num_layers)]
        outputs = []
        for t in range(seq_length):
            x_t = embedded[:, t, :]
            for i, cell in enumerate(self.lstm_cells):
                (cell_states[i], hidden_states[i]), x_t = cell((cell_states[i], hidden_states[i]), x_t)
            outputs.append(x_t)
        outputs = jnp.stack(outputs, axis=1)  # (batch, seq_length, hidden_dim)

        hidden_states = jnp.stack(hidden_states, axis=0)
        cell_states = jnp.stack(cell_states, axis=0)
        return outputs, (hidden_states, cell_states)


class Decoder(nn.Module):
    output_dim: int
    embed_dim: int
    hidden_dim: int
    num_layers: int
    src_seq_length: int

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.output_dim, features=self.embed_dim)
        self.attention = nn.Dense(self.src_seq_length)
        self.attention_combine = nn.Dense(self.embed_dim)
        self.lstm_cells = [nn.LSTMCell(features=self.hidden_dim) for _ in range(self.num_layers)]
        self.fc_out = nn.Dense(self.output_dim)

    def __call__(self, decoder_input, encoder_outputs, hidden_state, cell_state):
        # decoder_input: (batch,) or (batch, 1)
        embedded = self.embedding(decoder_input)  # (batch, embed_dim) or (batch, 1, embed_dim)
        if embedded.ndim == 3:
            embedded = embedded.squeeze(1)  # (batch, embed_dim)

        concat_input = jnp.concatenate([embedded, hidden_state[-1]], axis=-1)  # (batch, embed_dim + hidden_dim)
        attention_scores = self.attention(concat_input)  # (batch, src_seq_length)
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        context_vector = jnp.einsum('bs,bsh->bh', attention_weights, encoder_outputs)  # (batch, hidden_dim)

        combined = jnp.concatenate([embedded, context_vector], axis=-1)  # (batch, embed_dim + hidden_dim)
        combined = jax.nn.tanh(self.attention_combine(combined))  # (batch, embed_dim)
        
        new_hidden_states = []
        new_cell_states = []
        x = combined

        for i, cell in enumerate(self.lstm_cells):
            (new_cell, new_hidden), x = cell((cell_state[i], hidden_state[i]), x)
            new_hidden_states.append(new_hidden)
            new_cell_states.append(new_cell)
        new_hidden_states = jnp.stack(new_hidden_states, axis=0)  # (num_layers, batch, hidden_dim)
        new_cell_states = jnp.stack(new_cell_states, axis=0)      # (num_layers, batch, hidden_dim)
        output = self.fc_out(x)  # (batch, output_dim)
        return output, new_hidden_states, new_cell_states


def loss_fn(params, encoder, decoder, src, tgt):
    encoder_outputs, (enc_hidden, enc_cell) = encoder.apply({'params': params['encoder']}, src)
    loss = 0.0
    batch_size = src.shape[0]
    hidden_state, cell_state = enc_hidden, enc_cell

    decoder_input = jnp.zeros((batch_size,), dtype=jnp.int32)
    tgt_seq_length = tgt.shape[1]
    for t in range(tgt_seq_length):
        logits, hidden_state, cell_state = decoder.apply({'params': params['decoder']},
                                                           decoder_input,
                                                           encoder_outputs,
                                                           hidden_state,
                                                           cell_state)
        loss += jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, tgt[:, t]))

        decoder_input = tgt[:, t]
    return loss


def create_train_state(rng, encoder, decoder, src_vocab_size, tgt_vocab_size, src_seq_length):
    encoder_variables = encoder.init(rng, jnp.ones((1, src_seq_length), jnp.int32))
    decoder_variables = decoder.init(
        rng,
        jnp.ones((1,), jnp.int32),
        jnp.ones((1, src_seq_length, encoder.hidden_dim)),
        jnp.ones((encoder.num_layers, 1, encoder.hidden_dim)),
        jnp.ones((encoder.num_layers, 1, encoder.hidden_dim))
    )
    params = {
        'encoder': encoder_variables['params'],
        'decoder': decoder_variables['params']
    }
    tx = optax.adam(0.001)
    return train_state.TrainState.create(apply_fn=None, params=params, tx=tx)


@partial(jax.jit, static_argnums=(1, 2))
def train_step(state, encoder, decoder, src, tgt):
    # FIXED: Removed has_aux=True since loss_fn only returns loss
    loss, grads = jax.value_and_grad(loss_fn)(state.params, encoder, decoder, src, tgt)
    state = state.apply_gradients(grads=grads)
    return state, loss


def main():
    # FIXED: Changed vocabulary sizes from 1 to 20
    src_vocab_size = 20
    tgt_vocab_size = 20
    src_seq_length = 10
    tgt_seq_length = 12
    # FIXED: Changed batch_size from 1 to 16
    batch_size = 16
    embed_dim = 32
    hidden_dim = 64
    num_layers = 2

    rng = jax.random.PRNGKey(42)
    encoder = Encoder(input_dim=src_vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    decoder = Decoder(output_dim=tgt_vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
                      num_layers=num_layers, src_seq_length=src_seq_length)
    
    src_data = jax.random.randint(rng, (batch_size, src_seq_length), 0, src_vocab_size)
    tgt_data = jax.random.randint(rng, (batch_size, tgt_seq_length), 0, tgt_vocab_size)
    
    state = create_train_state(rng, encoder, decoder, src_vocab_size, tgt_vocab_size, src_seq_length)
    
    # FIXED: Changed epochs from 1000 to 100
    epochs = 100
    for epoch in range(epochs):
        rng, subkey = jax.random.split(rng)
        state, loss = train_step(state, encoder, decoder, src_data, tgt_data)
        # FIXED: Changed logging frequency from every 100 to every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss:.4f}")

    # Test the sequence-to-sequence model with new input
    test_input = jax.random.randint(rng, (1, src_seq_length), 0, src_vocab_size)
    
    # FIXED: Use trained encoder parameters instead of random initialization
    encoder_outputs, (enc_hidden, enc_cell) = encoder.apply({'params': state.params['encoder']}, test_input)
    
    hidden_state = enc_hidden
    cell_state = enc_cell
    
    decoder_input = jnp.array([0])
    
    output_sequence = []
    
    # FIXED: Use trained decoder parameters in decode_step
    @jax.jit
    def decode_step(decoder_input, hidden_state, cell_state, decoder_params, encoder_outputs):
        output, new_hidden_state, new_cell_state = decoder.apply(
            {'params': decoder_params}, 
            decoder_input, 
            encoder_outputs, 
            hidden_state, 
            cell_state
        )
        predicted = jnp.argmax(output, axis=1)
        return predicted, new_hidden_state, new_cell_state
    
    for _ in range(tgt_seq_length):
        predicted, hidden_state, cell_state = decode_step(
            decoder_input, 
            hidden_state, 
            cell_state, 
            state.params['decoder'],  # FIXED: Use trained parameters
            encoder_outputs
        )
        output_sequence.append(int(predicted.item()))
        decoder_input = predicted
    
    print(f"Input: {test_input.tolist()}, Output: {output_sequence}")


if __name__ == "__main__":
    main()