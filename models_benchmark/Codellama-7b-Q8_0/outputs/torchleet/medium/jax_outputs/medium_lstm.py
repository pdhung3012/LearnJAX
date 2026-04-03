import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import matplotlib.pyplot as plt


def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i : i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.stack(in_seq), jnp.stack(out_seq)


def init_custom_lstm_params(key, input_dim, hidden_units):
    keys = jax.random.split(key, 10)
    bound = 1.0 / jnp.sqrt(hidden_units)

    def gate_params(k_wx, k_wh):
        return {
            "Wx": jax.random.normal(k_wx, (input_dim, hidden_units)),
            "Wh": jax.random.normal(k_wh, (hidden_units, hidden_units)),
            "b": jnp.zeros((hidden_units,)),
        }

    return {
        "i": gate_params(keys[0], keys[1]),
        "f": gate_params(keys[2], keys[3]),
        "o": gate_params(keys[4], keys[5]),
        "g": gate_params(keys[6], keys[7]),
        "fc_weight": jax.random.uniform(keys[8], (hidden_units, 1), minval=-bound, maxval=bound),
        "fc_bias": jax.random.uniform(keys[9], (1,), minval=-bound, maxval=bound),
    }


def custom_lstm_forward(params, inputs, h_c=None, state_key=None):
    batch_size, seq_len, _ = inputs.shape
    hidden_units = params["i"]["b"].shape[0]

    if h_c is None:
        if state_key is None:
            state_key = jax.random.PRNGKey(0)
        h_key, c_key = jax.random.split(state_key)
        h = jax.random.normal(h_key, (batch_size, hidden_units))
        c = jax.random.normal(c_key, (batch_size, hidden_units))
    else:
        h, c = h_c

    all_hidden_states = []
    for t in range(seq_len):
        x_t = inputs[:, t, :]

        i_t = jax.nn.sigmoid(x_t @ params["i"]["Wx"] + h @ params["i"]["Wh"] + params["i"]["b"])
        f_t = jax.nn.sigmoid(x_t @ params["f"]["Wx"] + h @ params["f"]["Wh"] + params["f"]["b"])
        o_t = jax.nn.sigmoid(x_t @ params["o"]["Wx"] + h @ params["o"]["Wh"] + params["o"]["b"])
        g_t = jnp.tanh(x_t @ params["g"]["Wx"] + h @ params["g"]["Wh"] + params["g"]["b"])

        c = f_t * c + i_t * g_t
        h = o_t * jnp.tanh(c)
        all_hidden_states.append(h[:, None, :])

    outputs = jnp.concatenate(all_hidden_states, axis=1)
    pred = outputs @ params["fc_weight"] + params["fc_bias"]
    return pred, (h, c)


class LSTMModel(nn.Module):
    hidden_size: int = 50

    @nn.compact
    def __call__(self, x):
        out = nn.RNN(nn.OptimizedLSTMCell(features=self.hidden_size), name="lstm")(x)
        out = nn.Dense(features=1, name="fc")(out[:, -1, :])
        return out


def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)


def make_custom_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, inputs, targets, state_key):
        def loss_fn(p):
            pred, _ = custom_lstm_forward(p, inputs, h_c=None, state_key=state_key)
            return mse_loss(pred[:, -1, :], targets)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, loss

    return train_step


def make_inbuilt_train_step(model, optimizer):
    @jax.jit
    def train_step(params, opt_state, inputs, targets):
        def loss_fn(p):
            pred = model.apply({"params": p}, inputs)
            return mse_loss(pred, targets)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, loss

    return train_step


def main():
    sequence_length = 10
    num_samples = 100

    x = jnp.linspace(0, 4 * 3.14159, num_samples).reshape(-1, 1)
    y = jnp.sin(x)
    x_seq, y_seq = create_in_out_sequences(y, sequence_length)

    model_custom_key, model_inbuilt_key, train_key, test_key = jax.random.split(jax.random.PRNGKey(42), 4)

    model_custom_params = init_custom_lstm_params(model_custom_key, input_dim=1, hidden_units=50)
    optimizer_custom = optax.adam(learning_rate=0.01)
    optimizer_custom_state = optimizer_custom.init(model_custom_params)
    custom_train_step = make_custom_train_step(optimizer_custom)

    model_inbuilt = LSTMModel(hidden_size=50)
    model_inbuilt_params = model_inbuilt.init(model_inbuilt_key, x_seq)["params"]
    optimizer_inbuilt = optax.adam(learning_rate=0.01)
    optimizer_inbuilt_state = optimizer_inbuilt.init(model_inbuilt_params)
    inbuilt_train_step = make_inbuilt_train_step(model_inbuilt, optimizer_inbuilt)

    epochs = 500
    for epoch in range(epochs):
        train_key, state_key = jax.random.split(train_key)
        model_custom_params, optimizer_custom_state, loss = custom_train_step(
            model_custom_params,
            optimizer_custom_state,
            x_seq,
            y_seq,
            state_key,
        )
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {float(loss):.4f}")

    epochs = 500
    for epoch in range(epochs):
        model_inbuilt_params, optimizer_inbuilt_state, loss = inbuilt_train_step(
            model_inbuilt_params,
            optimizer_inbuilt_state,
            x_seq,
            y_seq,
        )
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {float(loss):.4f}")

    test_steps = 100
    x_test = jnp.linspace(0, 5 * 3.14159, test_steps).reshape(-1, 1)
    y_test = jnp.sin(x_test)
    x_test_seq, _ = create_in_out_sequences(y_test, sequence_length)

    pred_custom, _ = custom_lstm_forward(model_custom_params, x_test_seq, h_c=None, state_key=test_key)
    pred_inbuilt = model_inbuilt.apply({"params": model_inbuilt_params}, x_test_seq)

    pred_custom = jnp.ravel(pred_custom[:, -1, :])
    pred_inbuilt = jnp.ravel(pred_inbuilt)

    print(f"Predictions with Custom Model for new sequence: {pred_custom.tolist()}")
    print(f"Predictions with In-Built Model: {pred_inbuilt.tolist()}")

    plt.figure()
    plt.plot(pred_custom, label="custom model")
    plt.plot(pred_inbuilt, label="inbuilt model")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
