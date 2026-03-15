import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def create_in_out_sequences(data, seq_length):
    """Slide a window of seq_length over data to produce input/output pairs.

    PYTORCH EQUIVALENT:
        def create_in_out_sequences(data, seq_length):
            in_seq = []
            out_seq = []
            for i in range(len(data) - seq_length):
                in_seq.append(data[i:i + seq_length])
                out_seq.append(data[i + seq_length])
            return torch.stack(in_seq), torch.stack(out_seq)

    TRANSLATION NOTES:
        - torch.stack is replaced by jnp.stack. The loop-and-append
          pattern is kept for clarity; both produce shape
          (num_samples - seq_length, seq_length, features) for inputs
          and (num_samples - seq_length, features) for outputs.

    MOCK INJECTION:
        data = jnp.array([[0.0], [0.1], [0.2], [0.3], [0.4]])
        seq_length = 2
        # Expected in_seq shape: (3, 2, 1), out_seq shape: (3, 1)
    """
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.stack(in_seq), jnp.stack(out_seq)


def generate_data():
    """Generate synthetic sine-wave sequential data.

    PYTORCH EQUIVALENT:
        def generate_data():
            sequence_length = 10
            num_samples = 100
            X = torch.linspace(0, 4 * 3.14159, steps=num_samples).unsqueeze(1)
            y = torch.sin(X)
            X_seq, y_seq = create_in_out_sequences(y, sequence_length)
            return X, y, X_seq, y_seq

    TRANSLATION NOTES:
        - torch.linspace maps directly to jnp.linspace.
        - .unsqueeze(1) maps to [..., None] or jnp.expand_dims.
        - No RNG is involved — this is purely deterministic.

    MOCK INJECTION:
        X = jnp.linspace(0, 1, 20)[..., None]
        y = jnp.sin(X)
        X_seq, y_seq = create_in_out_sequences(y, 3)
    """
    sequence_length = 10
    num_samples = 100

    X = jnp.linspace(0, 4 * 3.14159, num=num_samples)[..., jnp.newaxis]
    y = jnp.sin(X)

    X_seq, y_seq = create_in_out_sequences(y, sequence_length)
    return X, y, X_seq, y_seq


# ---------------------------------------------------------------------------
# Custom LSTM model (plain functions + params dict)
# ---------------------------------------------------------------------------

def init_custom_lstm_params(key, input_dim, hidden_units):
    """Initialise parameters for the custom LSTM model.

    PYTORCH EQUIVALENT:
        class CustomLSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_units):
                super().__init__()
                weights_biases_init = lambda: (
                    nn.Parameter(torch.randn(input_dim, hidden_units)),
                    nn.Parameter(torch.randn(hidden_units, hidden_units)),
                    nn.Parameter(torch.zeros(hidden_units)),
                )
                self.Wxi, self.Whi, self.bi = weights_biases_init()
                self.Wxf, self.Whf, self.bf = weights_biases_init()
                self.Wxo, self.Who, self.bo = weights_biases_init()
                self.Wxc, self.Whc, self.bc = weights_biases_init()
                self.fc = nn.Linear(hidden_units, 1)

    TRANSLATION NOTES:
        - PyTorch's torch.randn is mapped to jax.random.normal. Each
          weight matrix gets its own sub-key via sequential splits to
          avoid key reuse. The 4 gates produce 8 randn calls (Wx and Wh
          per gate); biases are zeros matching torch.zeros.
        - nn.Linear(hidden_units, 1) stores weight as (out, in) = (1, 50)
          and bias as (1,). PyTorch inits with kaiming_uniform for weight
          and uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) for bias. For
          fan_in=50 this is U(-1/sqrt(50), 1/sqrt(50)). We replicate this.
        - The key is split 10 times: 8 for the randn weight matrices,
          2 for the fc layer (weight + bias).

    MOCK INJECTION:
        params = {
            'Wxi': jnp.ones((1, 50)) * 0.01,
            'Whi': jnp.ones((50, 50)) * 0.01,
            'bi':  jnp.zeros(50),
            ... (same pattern for f, o, c gates)
            'fc_w': jnp.ones((50, 1)) * 0.01,
            'fc_b': jnp.zeros(1),
        }
    """
    keys = random.split(key, 10)

    # 4 gates, each with Wx (input_dim, hidden) via randn
    # and Wh (hidden, hidden) via randn, bias via zeros.
    # Key allocation: keys[0..7] for the 8 randn weight matrices,
    # keys[8] for fc weight, keys[9] for fc bias.
    params = {
        'Wxi': random.normal(keys[0], shape=(input_dim, hidden_units)),
        'Whi': random.normal(keys[1], shape=(hidden_units, hidden_units)),
        'bi':  jnp.zeros(hidden_units),
        'Wxf': random.normal(keys[2], shape=(input_dim, hidden_units)),
        'Whf': random.normal(keys[3], shape=(hidden_units, hidden_units)),
        'bf':  jnp.zeros(hidden_units),
        'Wxo': random.normal(keys[4], shape=(input_dim, hidden_units)),
        'Who': random.normal(keys[5], shape=(hidden_units, hidden_units)),
        'bo':  jnp.zeros(hidden_units),
        'Wxc': random.normal(keys[6], shape=(input_dim, hidden_units)),
        'Whc': random.normal(keys[7], shape=(hidden_units, hidden_units)),
        'bc':  jnp.zeros(hidden_units),
    }

    # fc layer: nn.Linear(hidden_units, 1)
    # PyTorch kaiming_uniform for fan_in=hidden_units
    bound_w = 1.0 / jnp.sqrt(jnp.array(hidden_units, dtype=jnp.float32))
    bound_b = 1.0 / jnp.sqrt(jnp.array(hidden_units, dtype=jnp.float32))
    params['fc_w'] = random.uniform(keys[8], shape=(hidden_units, 1),
                                    minval=-bound_w, maxval=bound_w)
    params['fc_b'] = random.uniform(keys[9], shape=(1,),
                                    minval=-bound_b, maxval=bound_b)

    return params


def custom_lstm_forward(params, inputs, H_C, key):
    """Forward pass for the custom LSTM model.

    PYTORCH EQUIVALENT:
        def forward(self, inputs, H_C=None):
            batch_size, seq_len, _ = inputs.shape
            if not H_C:
                H = torch.randn(batch_size, self.hidden_units)
                C = torch.randn(batch_size, self.hidden_units)
            else:
                H, C = H_C
            all_hidden_states = []
            for t in range(seq_len):
                X_t = inputs[:, t, :]
                I_t = torch.sigmoid(...)
                F_t = torch.sigmoid(...)
                O_t = torch.sigmoid(...)
                C_tilde = torch.tanh(...)
                C = F_t * C + I_t * C_tilde
                H = O_t * torch.tanh(C)
                all_hidden_states.append(H.unsqueeze(1))
            outputs = torch.cat(all_hidden_states, dim=1)
            pred = self.fc(outputs)
            return pred, (H, C)

    TRANSLATION NOTES:
        - H_C is a tuple (H, C) or None. When None, H and C are
          initialised via jax.random.normal using the provided key,
          mirroring torch.randn in the original forward pass. The key
          is split into two sub-keys for H and C.
        - The gate computations are identical: sigmoid/tanh applied to
          matmul(X_t, Wx) + matmul(H, Wh) + b.
        - jax.nn.sigmoid and jnp.tanh replace torch.sigmoid and
          torch.tanh respectively.
        - The timestep loop uses a Python for-loop over seq_len; this
          is compatible with JIT since seq_len is static (known at
          trace time from the input shape).
        - self.fc(outputs) becomes jnp.dot(outputs, fc_w) + fc_b.

    MOCK INJECTION:
        params = <use init_custom_lstm_params with a fixed key>
        inputs = jnp.ones((4, 5, 1))  # batch=4, seq=5, feat=1
        H_C = None
        key = jax.random.PRNGKey(99)
        # Verify output shape is (4, 5, 1) and state shapes are (4, 50).
    """
    batch_size, seq_len, _ = inputs.shape

    if H_C is None:
        key_h, key_c = random.split(key)
        hidden_units = params['bi'].shape[0]
        H = random.normal(key_h, shape=(batch_size, hidden_units))
        C = random.normal(key_c, shape=(batch_size, hidden_units))
    else:
        H, C = H_C

    all_hidden_states = []
    for t in range(seq_len):
        X_t = inputs[:, t, :]
        I_t = jax.nn.sigmoid(jnp.dot(X_t, params['Wxi']) + jnp.dot(H, params['Whi']) + params['bi'])
        F_t = jax.nn.sigmoid(jnp.dot(X_t, params['Wxf']) + jnp.dot(H, params['Whf']) + params['bf'])
        O_t = jax.nn.sigmoid(jnp.dot(X_t, params['Wxo']) + jnp.dot(H, params['Who']) + params['bo'])
        C_tilde = jnp.tanh(jnp.dot(X_t, params['Wxc']) + jnp.dot(H, params['Whc']) + params['bc'])
        C = F_t * C + I_t * C_tilde
        H = O_t * jnp.tanh(C)
        all_hidden_states.append(H[:, jnp.newaxis, :])

    outputs = jnp.concatenate(all_hidden_states, axis=1)
    pred = jnp.dot(outputs, params['fc_w']) + params['fc_b']
    return pred, (H, C)


# ---------------------------------------------------------------------------
# Inbuilt LSTM model (plain functions + params dict)
# ---------------------------------------------------------------------------

def init_inbuilt_lstm_params(key, input_size, hidden_size):
    """Initialise parameters for the inbuilt-equivalent LSTM model.

    PYTORCH EQUIVALENT:
        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=50,
                                    num_layers=1, batch_first=True)
                self.fc = nn.Linear(50, 1)

    TRANSLATION NOTES:
        - nn.LSTM internally stores 4 gates concatenated into single
          matrices: weight_ih of shape (4*hidden, input) and weight_hh
          of shape (4*hidden, hidden), plus two bias vectors of shape
          (4*hidden,). PyTorch uses uniform(-1/sqrt(hidden), 1/sqrt(hidden))
          for all LSTM parameters. We replicate this.
        - The gate order in PyTorch's nn.LSTM is [i, f, g, o] where g is
          the cell candidate (C_tilde). We store them as a single
          concatenated matrix and slice during forward.
        - nn.Linear(50, 1): same kaiming init as the custom model fc.
        - 4 keys consumed: weight_ih, weight_hh, fc_w, fc_b. Biases use
          the same uniform init (not zeros) to match nn.LSTM defaults.

    MOCK INJECTION:
        params = {
            'weight_ih': jnp.zeros((4*50, 1)),
            'weight_hh': jnp.zeros((4*50, 50)),
            'bias_ih': jnp.zeros(4*50),
            'bias_hh': jnp.zeros(4*50),
            'fc_w': jnp.zeros((50, 1)),
            'fc_b': jnp.zeros(1),
        }
    """
    keys = random.split(key, 6)
    bound = 1.0 / jnp.sqrt(jnp.array(hidden_size, dtype=jnp.float32))

    params = {
        # nn.LSTM stores weights as (4*hidden, input/hidden)
        'weight_ih': random.uniform(keys[0], shape=(4 * hidden_size, input_size),
                                    minval=-bound, maxval=bound),
        'weight_hh': random.uniform(keys[1], shape=(4 * hidden_size, hidden_size),
                                    minval=-bound, maxval=bound),
        'bias_ih':   random.uniform(keys[2], shape=(4 * hidden_size,),
                                    minval=-bound, maxval=bound),
        'bias_hh':   random.uniform(keys[3], shape=(4 * hidden_size,),
                                    minval=-bound, maxval=bound),
    }

    # fc layer: nn.Linear(hidden_size, 1)
    bound_fc = 1.0 / jnp.sqrt(jnp.array(hidden_size, dtype=jnp.float32))
    params['fc_w'] = random.uniform(keys[4], shape=(hidden_size, 1),
                                    minval=-bound_fc, maxval=bound_fc)
    params['fc_b'] = random.uniform(keys[5], shape=(1,),
                                    minval=-bound_fc, maxval=bound_fc)

    return params


def inbuilt_lstm_forward(params, x):
    """Forward pass for the inbuilt-equivalent LSTM model.

    PYTORCH EQUIVALENT:
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    TRANSLATION NOTES:
        - nn.LSTM with batch_first=True expects (batch, seq, features).
          We manually implement the LSTM cell loop to match PyTorch's
          nn.LSTM semantics exactly.
        - PyTorch's nn.LSTM gate order is [i, f, g, o]. The concatenated
          weights are sliced into 4 chunks of hidden_size each.
        - nn.LSTM initialises H and C to zeros when no initial state is
          provided, unlike the custom model which uses randn. No RNG key
          needed here.
        - Only the last timestep's hidden state is passed through fc,
          matching out[:, -1, :] in the original.

    MOCK INJECTION:
        params = <use init_inbuilt_lstm_params with a fixed key>
        x = jnp.ones((4, 5, 1))  # batch=4, seq=5, feat=1
        # Verify output shape is (4, 1).
    """
    batch_size, seq_len, _ = x.shape
    hidden_size = params['weight_hh'].shape[1]

    # nn.LSTM defaults to zero initial state
    H = jnp.zeros((batch_size, hidden_size))
    C = jnp.zeros((batch_size, hidden_size))

    # Slice concatenated weights into gates: [i, f, g, o]
    W_ih = params['weight_ih']   # (4*hidden, input)
    W_hh = params['weight_hh']   # (4*hidden, hidden)
    b_ih = params['bias_ih']     # (4*hidden,)
    b_hh = params['bias_hh']     # (4*hidden,)

    for t in range(seq_len):
        X_t = x[:, t, :]  # (batch, input)

        # gates = X_t @ W_ih.T + H @ W_hh.T + b_ih + b_hh
        gates = jnp.dot(X_t, W_ih.T) + jnp.dot(H, W_hh.T) + b_ih + b_hh

        # Split into 4 gates along last dimension
        i_gate = jax.nn.sigmoid(gates[:, 0*hidden_size:1*hidden_size])
        f_gate = jax.nn.sigmoid(gates[:, 1*hidden_size:2*hidden_size])
        g_gate = jnp.tanh(gates[:, 2*hidden_size:3*hidden_size])
        o_gate = jax.nn.sigmoid(gates[:, 3*hidden_size:4*hidden_size])

        C = f_gate * C + i_gate * g_gate
        H = o_gate * jnp.tanh(C)

    # fc on last hidden state only
    out = jnp.dot(H, params['fc_w']) + params['fc_b']
    return out


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def mse_loss_custom(params, X_seq, y_seq, key):
    """MSE loss for the custom LSTM model.

    PYTORCH EQUIVALENT:
        criterion = nn.MSELoss()
        pred, state = model(X_seq, state)  # state=None each epoch
        loss = criterion(pred[:, -1, :], y_seq)

    TRANSLATION NOTES:
        - The forward pass is called inside this function so that
          jax.value_and_grad can differentiate through both model and loss.
        - H_C is always None (state reset each epoch), so the key is
          passed through to generate random H and C.
        - pred[:, -1, :] selects the last timestep's prediction, matching
          the original training loop.

    MOCK INJECTION:
        params = <init_custom_lstm_params with a fixed key>
        X_seq = jnp.ones((80, 10, 1)); y_seq = jnp.zeros((80, 1))
        key = jax.random.PRNGKey(0)
        # Verify output is a scalar >= 0.
    """
    pred, _ = custom_lstm_forward(params, X_seq, None, key)
    return jnp.mean((pred[:, -1, :] - y_seq) ** 2)


def mse_loss_inbuilt(params, X_seq, y_seq):
    """MSE loss for the inbuilt-equivalent LSTM model.

    PYTORCH EQUIVALENT:
        criterion = nn.MSELoss()
        pred = model(X_seq)
        loss = criterion(pred, y_seq)

    TRANSLATION NOTES:
        - No RNG key needed since the inbuilt LSTM initialises H/C to
          zeros (deterministic).
        - The forward pass is called inside this function for
          jax.value_and_grad compatibility.

    MOCK INJECTION:
        params = <init_inbuilt_lstm_params with a fixed key>
        X_seq = jnp.ones((80, 10, 1)); y_seq = jnp.zeros((80, 1))
        # Verify output is a scalar >= 0.
    """
    pred = inbuilt_lstm_forward(params, X_seq)
    return jnp.mean((pred - y_seq) ** 2)


# ---------------------------------------------------------------------------
# Train steps (@jit — no side effects)
# ---------------------------------------------------------------------------

def _train_step_custom(params, opt_state, X_seq, y_seq, key, optimizer_update):
    """Single training step for the custom LSTM model.

    PYTORCH EQUIVALENT:
        state = None
        pred, state = model_custom(X_seq, state)
        loss = criterion(pred[:, -1, :], y_seq)
        optimizer_custom.zero_grad()
        loss.backward()
        optimizer_custom.step()

    TRANSLATION NOTES:
        - value_and_grad computes both the loss scalar and gradients in
          one pass. The key is passed through to mse_loss_custom for the
          random H/C initialisation.
        - optax replaces optim.Adam: optimizer_update is a pure function
          that computes parameter updates from gradients and optimizer
          state, with no mutation.
        - optimizer.zero_grad() is omitted; JAX computes fresh gradients
          each call.
        - No print() or side effects inside this @jit function.

    MOCK INJECTION:
        params = <init_custom_lstm_params with a fixed key>
        opt_state = optimizer.init(params)
        X_seq = jnp.ones((80, 10, 1)); y_seq = jnp.zeros((80, 1))
        key = jax.random.PRNGKey(0)
        # Verify loss is scalar and params dict has same keys.
    """
    loss, grads = value_and_grad(mse_loss_custom)(params, X_seq, y_seq, key)
    updates, new_opt_state = optimizer_update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

train_step_custom = jit(_train_step_custom, static_argnums=(5,))


def _train_step_inbuilt(params, opt_state, X_seq, y_seq, optimizer_update):
    """Single training step for the inbuilt-equivalent LSTM model.

    PYTORCH EQUIVALENT:
        pred = model_inbuilt(X_seq)
        loss = criterion(pred, y_seq)
        optimizer_inbuilt.zero_grad()
        loss.backward()
        optimizer_inbuilt.step()

    TRANSLATION NOTES:
        - Same pattern as train_step_custom but without a key argument,
          since the inbuilt LSTM's forward pass is deterministic (zero-
          initialised H/C).
        - optax.adam replaces optim.Adam with the same lr=0.01.

    MOCK INJECTION:
        params = <init_inbuilt_lstm_params with a fixed key>
        opt_state = optimizer.init(params)
        X_seq = jnp.ones((80, 10, 1)); y_seq = jnp.zeros((80, 1))
        # Verify loss is scalar and params dict has same keys.
    """
    loss, grads = value_and_grad(mse_loss_inbuilt)(params, X_seq, y_seq)
    updates, new_opt_state = optimizer_update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

train_step_inbuilt = jit(_train_step_inbuilt, static_argnums=(4,))


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_model_custom(X_seq, y_seq, params, opt_state, optimizer_update,
                       num_epochs, key):
    """Full training loop for the custom LSTM model.

    PYTORCH EQUIVALENT:
        epochs = 500
        for epoch in range(epochs):
            state = None
            pred, state = model_custom(X_seq, state)
            loss = criterion(pred[:, -1, :], y_seq)
            optimizer_custom.zero_grad()
            loss.backward()
            optimizer_custom.step()
            if (epoch + 1) % 50 == 0:
                print(...)

    TRANSLATION NOTES:
        - Accepts (X_seq, y_seq, params, ...) — data and params are
          passed in, never generated internally (Rule 2).
        - The key is split each epoch to provide a fresh sub-key for the
          random H/C initialisation in the forward pass. This mirrors
          PyTorch's behaviour where torch.randn is called each epoch
          (since state is reset to None every iteration) and consumes
          global RNG state sequentially.
        - print() is here, outside the @jit boundary.

    MOCK INJECTION:
        X_seq = jnp.ones((80, 10, 1)); y_seq = jnp.zeros((80, 1))
        params = <init_custom_lstm_params with a fixed key>
        opt_state = optimizer.init(params)
        key = jax.random.PRNGKey(0); num_epochs = 10
        # Verify loss decreases over epochs.
    """
    for epoch in range(num_epochs):
        key, subkey = random.split(key)
        params, opt_state, loss = train_step_custom(
            params, opt_state, X_seq, y_seq, subkey, optimizer_update
        )

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return params, opt_state, key


def train_model_inbuilt(X_seq, y_seq, params, opt_state, optimizer_update,
                        num_epochs):
    """Full training loop for the inbuilt-equivalent LSTM model.

    PYTORCH EQUIVALENT:
        epochs = 500
        for epoch in range(epochs):
            pred = model_inbuilt(X_seq)
            loss = criterion(pred, y_seq)
            optimizer_inbuilt.zero_grad()
            loss.backward()
            optimizer_inbuilt.step()
            if (epoch + 1) % 50 == 0:
                print(...)

    TRANSLATION NOTES:
        - No key argument needed since inbuilt LSTM forward is
          deterministic (zero-initialised state).
        - print() is here, outside the @jit boundary.

    MOCK INJECTION:
        X_seq = jnp.ones((80, 10, 1)); y_seq = jnp.zeros((80, 1))
        params = <init_inbuilt_lstm_params with a fixed key>
        opt_state = optimizer.init(params)
        num_epochs = 10
        # Verify loss decreases over epochs.
    """
    for epoch in range(num_epochs):
        params, opt_state, loss = train_step_inbuilt(
            params, opt_state, X_seq, y_seq, optimizer_update
        )

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return params, opt_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Entry point mirroring the PyTorch main().

    PYTORCH EQUIVALENT:
        def main():
            torch.manual_seed(42)
            X, y, X_seq, y_seq = generate_data()
            model_custom = make_model_custom()
            model_inbuilt = make_model_inbuilt()
            criterion = make_criterion()
            optimizer_custom = make_optimizer_custom(model_custom)
            optimizer_inbuilt = make_optimizer_inbuilt(model_inbuilt)
            train_model_custom(X_seq, y_seq, model_custom, ..., 500)
            train_model_inbuilt(X_seq, y_seq, model_inbuilt, ..., 500)
            # test and plot ...

    TRANSLATION NOTES:
        - torch.manual_seed(42) becomes jax.random.PRNGKey(42). The key
          is split sequentially to feed init_custom_lstm_params,
          init_inbuilt_lstm_params, and the custom training loop, preserving
          the original seed -> data -> custom_model -> inbuilt_model ->
          train_custom -> train_inbuilt ordering.
        - optax.adam(lr=0.01) replaces optim.Adam(lr=0.01) for both
          models. The optimizer object is split into an init function
          (producing opt_state) and an update function (passed to
          train_step).
        - make_criterion() and make_optimizer() have no direct standalone
          JAX counterparts; the loss is defined in mse_loss_custom /
          mse_loss_inbuilt and optax handles the optimizer.
        - 'with torch.no_grad()' is omitted; JAX only computes gradients
          when explicitly requested.

    MOCK INJECTION:
        Replace generate_data and init_* with fixed tensors:
            X_seq = jnp.ones((80, 10, 1))
            y_seq = jnp.zeros((80, 1))
            custom_params = <fixed dict>
            inbuilt_params = <fixed dict>
    """
    # RNG key at top, matching torch.manual_seed(42) placement.
    key = random.PRNGKey(42)
    key_custom, key_inbuilt, key_train = random.split(key, 3)

    # Data generation (deterministic — no key needed)
    X, y, X_seq, y_seq = generate_data()

    # Init models
    custom_params = init_custom_lstm_params(key_custom, input_dim=1, hidden_units=50)
    inbuilt_params = init_inbuilt_lstm_params(key_inbuilt, input_size=1, hidden_size=50)

    # Optimizers: optax.adam replaces optim.Adam(lr=0.01)
    custom_optimizer = optax.adam(learning_rate=0.01)
    custom_opt_state = custom_optimizer.init(custom_params)

    inbuilt_optimizer = optax.adam(learning_rate=0.01)
    inbuilt_opt_state = inbuilt_optimizer.init(inbuilt_params)

    # Train custom model first, then inbuilt — same order as original
    custom_params, custom_opt_state, key_train = train_model_custom(
        X_seq, y_seq, custom_params, custom_opt_state,
        custom_optimizer.update, num_epochs=500, key=key_train,
    )
    inbuilt_params, inbuilt_opt_state = train_model_inbuilt(
        X_seq, y_seq, inbuilt_params, inbuilt_opt_state,
        inbuilt_optimizer.update, num_epochs=500,
    )

    # Testing on new data
    sequence_length = 10
    test_steps = 100
    X_test = jnp.linspace(0, 5 * 3.14159, num=test_steps)[..., jnp.newaxis]
    y_test = jnp.sin(X_test)

    X_test_seq, _ = create_in_out_sequences(y_test, sequence_length)

    # Inference (torch.no_grad() omitted — JAX is explicit-grad)
    # For custom model, provide a key for the random H/C init during inference
    key_train, subkey = random.split(key_train)
    pred_custom, _ = custom_lstm_forward(custom_params, X_test_seq, None, subkey)
    pred_inbuilt = inbuilt_lstm_forward(inbuilt_params, X_test_seq)

    pred_custom = pred_custom[:, -1, :].reshape(-1)
    pred_inbuilt = pred_inbuilt.reshape(-1)
    print(f"Predictions with Custom Model for new sequence: {pred_custom.tolist()}")
    print(f"Predictions with In-Built Model: {pred_inbuilt.tolist()}")


if __name__ == "__main__":
    main()