import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import matplotlib.pyplot as plt

# --- Module Level Classes ---

class CustomLSTMModel(nn.Module):
    input_dim: int
    hidden_units: int

    @nn.compact
    def __call__(self, inputs, key, H_C=None):
        """
        PYTORCH EQUIVALENT: CustomLSTMModel(nn.Module)
        Uses explicit Python loop (matching PyTorch) for gradient equivalence.
        """
        batch_size, seq_len, _ = inputs.shape
        init_fn = nn.initializers.normal()
        bias_init = nn.initializers.zeros

        Wxi = self.param('Wxi', init_fn, (self.input_dim, self.hidden_units))
        Whi = self.param('Whi', init_fn, (self.hidden_units, self.hidden_units))
        bi = self.param('bi', bias_init, (self.hidden_units,))

        Wxf = self.param('Wxf', init_fn, (self.input_dim, self.hidden_units))
        Whf = self.param('Whf', init_fn, (self.hidden_units, self.hidden_units))
        bf = self.param('bf', bias_init, (self.hidden_units,))

        Wxo = self.param('Wxo', init_fn, (self.input_dim, self.hidden_units))
        Who = self.param('Who', init_fn, (self.hidden_units, self.hidden_units))
        bo = self.param('bo', bias_init, (self.hidden_units,))

        Wxc = self.param('Wxc', init_fn, (self.input_dim, self.hidden_units))
        Whc = self.param('Whc', init_fn, (self.hidden_units, self.hidden_units))
        bc = self.param('bc', bias_init, (self.hidden_units,))

        fc_layer = nn.Dense(1)

        if H_C is None:
            k1, k2 = jax.random.split(key)
            H = jax.random.normal(k1, (batch_size, self.hidden_units))
            C = jax.random.normal(k2, (batch_size, self.hidden_units))
        else:
            H, C = H_C

        all_hidden_states = []
        for t in range(seq_len):
            X_t = inputs[:, t, :]
            I_t = jax.nn.sigmoid(jnp.dot(X_t, Wxi) + jnp.dot(H, Whi) + bi)
            F_t = jax.nn.sigmoid(jnp.dot(X_t, Wxf) + jnp.dot(H, Whf) + bf)
            O_t = jax.nn.sigmoid(jnp.dot(X_t, Wxo) + jnp.dot(H, Who) + bo)
            C_tilde = jnp.tanh(jnp.dot(X_t, Wxc) + jnp.dot(H, Whc) + bc)
            C = F_t * C + I_t * C_tilde
            H = O_t * jnp.tanh(C)
            all_hidden_states.append(H)

        outputs = jnp.stack(all_hidden_states, axis=1)
        pred = fc_layer(outputs)
        return pred, (H, C)


class LSTMModel(nn.Module):
    """
    PYTORCH EQUIVALENT: LSTMModel(nn.Module) using nn.LSTM

    Uses per-gate Dense layers with BOTH input-side and hidden-side biases
    to match PyTorch's nn.LSTM which has separate bias_ih and bias_hh.

    PyTorch computes: gate = W_ih @ x + b_ih + W_hh @ h + b_hh
    So we need:       gate = dense_i(x) + dense_h(h)
    where both dense_i and dense_h have their own bias terms.

    Gate mapping:
      PyTorch i (input)     -> ii/hi
      PyTorch f (forget)    -> if/hf
      PyTorch g (candidate) -> ig/hg
      PyTorch o (output)    -> io/ho
    """
    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, input_dim = x.shape
        hidden_size = 50

        # Input-to-hidden: WITH bias (matching PyTorch bias_ih)
        dense_ii = nn.Dense(hidden_size, use_bias=True, name='ii')
        dense_ig = nn.Dense(hidden_size, use_bias=True, name='ig')
        dense_if = nn.Dense(hidden_size, use_bias=True, name='if')
        dense_io = nn.Dense(hidden_size, use_bias=True, name='io')

        # Hidden-to-hidden: WITH bias (matching PyTorch bias_hh)
        dense_hi = nn.Dense(hidden_size, use_bias=True, name='hi')
        dense_hg = nn.Dense(hidden_size, use_bias=True, name='hg')
        dense_hf = nn.Dense(hidden_size, use_bias=True, name='hf')
        dense_ho = nn.Dense(hidden_size, use_bias=True, name='ho')

        fc = nn.Dense(1)

        h = jnp.zeros((batch_size, hidden_size))
        c = jnp.zeros((batch_size, hidden_size))

        for t in range(seq_len):
            x_t = x[:, t, :]
            i = jax.nn.sigmoid(dense_ii(x_t) + dense_hi(h))
            g = jnp.tanh(dense_ig(x_t) + dense_hg(h))
            f = jax.nn.sigmoid(dense_if(x_t) + dense_hf(h))
            o = jax.nn.sigmoid(dense_io(x_t) + dense_ho(h))
            c = f * c + i * g
            h = o * jnp.tanh(c)

        out = fc(h)
        return out


# --- Helper Components ---

def generate_data():
    sequence_length = 10
    num_samples = 100
    X = jnp.linspace(0, 4 * 3.14159, num_samples).reshape(-1, 1)
    y = jnp.sin(X)

    in_seq = [y[i:i + sequence_length] for i in range(len(y) - sequence_length)]
    out_seq = [y[i + sequence_length] for i in range(len(y) - sequence_length)]
    return jnp.stack(in_seq), jnp.stack(out_seq)

def make_model(key, model_type="custom"):
    x_dummy = jnp.ones((1, 10, 1))
    if model_type == "custom":
        model = CustomLSTMModel(input_dim=1, hidden_units=50)
        params = model.init(key, x_dummy, key)['params']
    else:
        model = LSTMModel()
        params = model.init(key, x_dummy)['params']
    return model, params

def train_step(params, opt_state, X, y, model, optimizer, is_custom, key):
    def loss_fn(p):
        if is_custom:
            pred, _ = model.apply({'params': p}, X, key)
            loss = jnp.mean((pred[:, -1, :] - y) ** 2)
        else:
            pred = model.apply({'params': p}, X)
            loss = jnp.mean((pred - y) ** 2)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def train_model(X, y, params, model, optimizer, num_epochs, is_custom=True, key=None):
    opt_state = optimizer.init(params)
    jitted_step = jax.jit(train_step, static_argnums=(4, 5, 6))

    for epoch in range(num_epochs):
        step_key = jax.random.fold_in(key, epoch) if key is not None else jax.random.PRNGKey(0)
        params, opt_state, loss = jitted_step(params, opt_state, X, y, model, optimizer, is_custom, step_key)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")
    return params

def main():
    main_key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(main_key, 4)

    X_seq, y_seq = generate_data()

    model_custom, params_custom = make_model(k1, "custom")
    model_inbuilt, params_inbuilt = make_model(k2, "inbuilt")
    optimizer = optax.adam(0.01)

    print("Training Custom...")
    params_custom = train_model(X_seq, y_seq, params_custom, model_custom, optimizer, 500, is_custom=True, key=k3)

    print("\nTraining Inbuilt...")
    params_inbuilt = train_model(X_seq, y_seq, params_inbuilt, model_inbuilt, optimizer, 500, is_custom=False, key=k4)

    X_test = jnp.linspace(0, 5 * 3.14159, 100).reshape(-1, 1)
    y_test = jnp.sin(X_test)
    X_test_seq = jnp.stack([y_test[i:i+10] for i in range(len(y_test)-10)])

    pred_c, _ = model_custom.apply({'params': params_custom}, X_test_seq, jax.random.PRNGKey(0))
    pred_i = model_inbuilt.apply({'params': params_inbuilt}, X_test_seq)

    plt.plot(pred_c[:, -1, 0], label="Custom")
    plt.plot(pred_i[:, 0], label="Inbuilt")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
