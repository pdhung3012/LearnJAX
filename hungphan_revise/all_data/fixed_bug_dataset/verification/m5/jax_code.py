import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np

# RNN Cell Definition
class RNNCell(nn.Module):
    input_size: int
    hidden_size: int

    def setup(self):
        self.W_ih = self.param('W_ih', nn.initializers.xavier_uniform(), (self.input_size, self.hidden_size))
        self.W_hh = self.param('W_hh', nn.initializers.xavier_uniform(), (self.hidden_size, self.hidden_size))

    def __call__(self, carry, x):
        new_carry = jnp.tanh(jnp.dot(x, self.W_ih) + jnp.dot(carry, self.W_hh))
        return new_carry, None

# RNN Module Definition
class RNN(nn.Module):
    input_size: int
    hidden_size: int
    output_size: int

    def setup(self):
        # Wrap RNNCell with nn.scan for proper parameter handling
        self.scanned_rnn_cell = nn.scan(
            RNNCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0
        )(input_size=self.input_size, hidden_size=self.hidden_size)
        self.fc = nn.Dense(self.output_size)

    def __call__(self, x):
        # Transpose x from (batch, seq, feat) to (seq, batch, feat)
        x = jnp.transpose(x, (1, 0, 2))
        batch_size = x.shape[1]
        init_carry = jnp.zeros((batch_size, self.hidden_size))
        final_carry, _ = self.scanned_rnn_cell(init_carry, x)
        output = self.fc(final_carry)
        return output

# Loss Function
def compute_loss(params, model, x, targets):
    logits = model.apply(params, x)
    return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=targets))

# Main Function
def main():
    # Sample data for training
    x_train = jnp.array(np.random.rand(100, 10, 1))  # 100 samples, 10 timesteps, 1 feature
    y_train = jnp.array(np.random.randint(0, 2, (100, 2)))  # 2 classes, output at last timestep

    # Instantiate the RNN model
    model = RNN(input_size=1, hidden_size=16, output_size=2)
    params = model.init(jax.random.PRNGKey(0), x_train)

    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    # Training Loop
    epochs = 500
    for epoch in range(epochs):
        loss, grads = jax.value_and_grad(compute_loss)(params, model, x_train, y_train)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

    # Testing on new data
    X_test = np.linspace(4 * np.pi, 5 * np.pi, 10).reshape(1, 10, 1)  # 1 sample, 10 timesteps
    predictions = model.apply(params, X_test)
    print(f"Predictions for new sequence: {predictions.tolist()}")

if __name__ == "__main__":
    main()