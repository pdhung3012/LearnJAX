import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

class LSTM(nn.Module):
    features: int  # Expected to be the sequence length (e.g., 10)

    def setup(self):
        # Define the Dense submodule with a fixed feature size.
        self.dummy_dense = nn.Dense(features=self.features, name='dummy_dense')

    def __call__(self, x):
        # Use the pre-defined Dense layer.
        x = self.dummy_dense(x)
        rng = self.make_rng('lstm')
        return process_sequence(x, rng)


def process_sequence(inputs, prng_key):
    # Ensure inputs is 2D (batch, seq_length).
    assert inputs.ndim == 2, f"Expected inputs to be 2D (batch, seq_length), got {inputs.shape}"
    # Transpose to (seq_length, batch) for scanning over time.
    inputs = jnp.swapaxes(inputs, 0, 1)

    def step(carry, input_data):
        # Example step: add the input to the carry.
        new_carry = carry + input_data
        return new_carry, new_carry

    batch = inputs.shape[1]  # Number of samples in the batch.
    initial_carry = jnp.zeros((batch,))
    outputs, _ = jax.lax.scan(step, initial_carry, inputs)
    if outputs.ndim == 1:  # If seq_length = 1, outputs is 1D
        outputs = outputs[None, :]
    #   outputs = jnp.swapaxes(outputs, 0, 1)
    outputs = jnp.swapaxes(outputs, 0, 1)
    return outputs


def loss_fn(params, X, y):
    # Use a fixed RNG for LSTM operations during loss computation.
    preds = model.apply({'params': params}, X, rngs={'lstm': jax.random.PRNGKey(0)})
    return jnp.mean((preds - y) ** 2)


def main():
    batch_size = 32
    seq_length = 10  # Number of time steps.
    num_epochs = 5
    key = jax.random.PRNGKey(0)

    # Generate explicit training data of shape (batch, seq_length).
    X_train = jax.random.normal(key, (batch_size, seq_length))
    key, subkey = jax.random.split(key)
    y_train = jax.random.normal(subkey, (batch_size, seq_length))

    global model
    # Initialize LSTM with fixed feature size (equal to seq_length).
    model = LSTM(features=seq_length)
    # Use separate PRNG keys for parameters and LSTM operations.
    params_key, lstm_key = jax.random.split(key)
    variables = model.init({'params': params_key, 'lstm': lstm_key}, X_train)
    params = variables['params']
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        outputs = process_sequence(X_train, subkey)
        current_loss = loss_fn(params, X_train, y_train)
        grad = jax.grad(loss_fn)(params, X_train, y_train)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {current_loss:.4f}")


if __name__ == "__main__":
    main()