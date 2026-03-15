"""
JAX translation of pytorch_refactored.py — a simple GAN.

Mirrors the PyTorch structure block-for-block:
    generate_data  → generate_data
    Generator      → generator_forward  (+ init_generator_params)
    Discriminator  → discriminator_forward (+ init_discriminator_params)
    make_model     → make_model
    make_criterion → (inlined as bce_loss)
    make_optimizer → make_optimizer  (optax.adam)
    train_model    → train_model  (contains train_step under @jit)
    main           → main
"""

import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax


# ---------------------------------------------------------------------------
# Weight initialisation helpers
# ---------------------------------------------------------------------------
# PyTorch nn.Linear default init: weight ~ U(-1/√fan_in, 1/√fan_in),
#                                  bias   ~ U(-1/√fan_in, 1/√fan_in)
# We replicate this exactly via jax.random.uniform.

def _init_linear(key, in_features, out_features):
    """Initialise a single Linear layer to match PyTorch defaults.

    PYTORCH EQUIVALENT:
        nn.Linear(in_features, out_features)
    TRANSLATION NOTES:
        PyTorch default init draws weight and bias from
        U(-1/√fan_in, 1/√fan_in).  We reproduce this with
        jax.random.uniform on two split sub-keys.
        Weight shape is (in, out) — we transpose during forward pass
        to match PyTorch's (out, in) @ x convention, but storing as
        (in, out) lets us write jnp.dot(x, w) + b directly.
    MOCK INJECTION:
        Supply {'w': jnp.ones((in, out)), 'b': jnp.zeros((out,))}
        to bypass random init.
    """
    k1, k2 = random.split(key)
    bound = 1.0 / jnp.sqrt(float(in_features))
    w = random.uniform(k1, (in_features, out_features),
                       minval=-bound, maxval=bound)
    b = random.uniform(k2, (out_features,),
                       minval=-bound, maxval=bound)
    return {'w': w, 'b': b}


# ---------------------------------------------------------------------------
# Generator  (plain-function + params dict)
# ---------------------------------------------------------------------------

def init_generator_params(key, input_dim, output_dim):
    """Create the parameter dict for the Generator.

    PYTORCH EQUIVALENT:
        Generator.__init__  (three nn.Linear layers: 10→128, 128→256, 256→1)
    TRANSLATION NOTES:
        Each nn.Linear becomes a (w, b) pair initialised with
        _init_linear.  Keys are split sequentially so the draw order
        is deterministic.
    MOCK INJECTION:
        Pass a dict with keys 'linear1', 'linear2', 'linear3', each
        containing {'w': <array>, 'b': <array>} of correct shapes.
    """
    k1, k2, k3 = random.split(key, 3)
    params = {
        'linear1': _init_linear(k1, input_dim, 128),
        'linear2': _init_linear(k2, 128, 256),
        'linear3': _init_linear(k3, 256, output_dim),
    }
    return params


def generator_forward(params, x):
    """Forward pass of the Generator.

    PYTORCH EQUIVALENT:
        Generator.forward — Sequential(Linear, ReLU, Linear, ReLU,
                                       Linear, Tanh)
    TRANSLATION NOTES:
        jnp.dot(x, w) + b replaces nn.Linear.  jax.nn.relu and
        jax.nn.tanh replace nn.ReLU / nn.Tanh.
    MOCK INJECTION:
        Provide params dict and x = jnp.ones((batch, input_dim)).
    """
    x = jax.nn.relu(jnp.dot(x, params['linear1']['w']) + params['linear1']['b'])
    x = jax.nn.relu(jnp.dot(x, params['linear2']['w']) + params['linear2']['b'])
    x = jnp.tanh(jnp.dot(x, params['linear3']['w']) + params['linear3']['b'])
    return x


# ---------------------------------------------------------------------------
# Discriminator  (plain-function + params dict)
# ---------------------------------------------------------------------------

def init_discriminator_params(key, input_dim):
    """Create the parameter dict for the Discriminator.

    PYTORCH EQUIVALENT:
        Discriminator.__init__  (three nn.Linear layers: 1→256, 256→128, 128→1)
    TRANSLATION NOTES:
        LeakyReLU(0.2) → jax.nn.leaky_relu(x, negative_slope=0.2).
        Sigmoid → jax.nn.sigmoid.
    MOCK INJECTION:
        Pass a dict with keys 'linear1', 'linear2', 'linear3', each
        containing {'w': <array>, 'b': <array>} of correct shapes.
    """
    k1, k2, k3 = random.split(key, 3)
    params = {
        'linear1': _init_linear(k1, input_dim, 256),
        'linear2': _init_linear(k2, 256, 128),
        'linear3': _init_linear(k3, 128, 1),
    }
    return params


def discriminator_forward(params, x):
    """Forward pass of the Discriminator.

    PYTORCH EQUIVALENT:
        Discriminator.forward — Sequential(Linear, LeakyReLU(0.2),
                                           Linear, LeakyReLU(0.2),
                                           Linear, Sigmoid)
    TRANSLATION NOTES:
        jax.nn.leaky_relu uses negative_slope=0.2 to match PyTorch's
        nn.LeakyReLU(0.2).
    MOCK INJECTION:
        Provide params dict and x = jnp.ones((batch, 1)).
    """
    x = jax.nn.leaky_relu(
        jnp.dot(x, params['linear1']['w']) + params['linear1']['b'],
        negative_slope=0.2)
    x = jax.nn.leaky_relu(
        jnp.dot(x, params['linear2']['w']) + params['linear2']['b'],
        negative_slope=0.2)
    x = jax.nn.sigmoid(
        jnp.dot(x, params['linear3']['w']) + params['linear3']['b'])
    return x


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(key):
    """Generate synthetic training data: 100 samples in [-1, 1].

    PYTORCH EQUIVALENT:
        torch.manual_seed(42)
        real_data = torch.rand(100, 1) * 2 - 1
    TRANSLATION NOTES:
        torch.rand → jax.random.uniform.  The key is passed in
        explicitly (JAX functional PRNG) rather than relying on
        global state.  y is None — the original GAN has no explicit
        labels for real data; we return (X, y) per the interface
        contract.
    MOCK INJECTION:
        Return (jnp.linspace(-1, 1, 100).reshape(100, 1), None)
        for deterministic testing without RNG.
    """
    real_data = random.uniform(key, (100, 1), minval=-1.0, maxval=1.0)
    y = None
    return real_data, y


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(key, latent_dim=10, data_dim=1):
    """Instantiate Generator and Discriminator parameter dicts.

    PYTORCH EQUIVALENT:
        G = Generator(latent_dim, data_dim)
        D = Discriminator(data_dim)
    TRANSLATION NOTES:
        Two sub-keys are split from the incoming key — one for G,
        one for D — preserving deterministic initialisation order.
    MOCK INJECTION:
        Supply pre-built params dicts directly instead of calling
        this function.
    """
    key_g, key_d = random.split(key)
    g_params = init_generator_params(key_g, latent_dim, data_dim)
    d_params = init_discriminator_params(key_d, data_dim)
    return g_params, d_params


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def bce_loss(predictions, targets):
    """Binary cross-entropy loss (mean-reduced).

    PYTORCH EQUIVALENT:
        nn.BCELoss()
    TRANSLATION NOTES:
        PyTorch's BCELoss computes -[t*log(p) + (1-t)*log(1-p)]
        averaged over all elements.  We clip predictions to avoid
        log(0).
    MOCK INJECTION:
        bce_loss(jnp.array([[0.5]]), jnp.array([[1.0]])) should
        return ≈ 0.6931.
    """
    eps = 1e-7
    predictions = jnp.clip(predictions, eps, 1.0 - eps)
    return -jnp.mean(targets * jnp.log(predictions)
                     + (1.0 - targets) * jnp.log(1.0 - predictions))


def make_criterion():
    """Return the loss function.

    PYTORCH EQUIVALENT:
        criterion = nn.BCELoss()
    TRANSLATION NOTES:
        Returns the bce_loss callable directly.
    MOCK INJECTION:
        N/A — stateless function reference.
    """
    return bce_loss


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def make_optimizer(lr=0.001):
    """Return an optax Adam optimizer matching the original PyTorch config.

    PYTORCH EQUIVALENT:
        optim.Adam(model.parameters(), lr=0.001)
    TRANSLATION NOTES:
        optax.adam defaults (b1=0.9, b2=0.999, eps=1e-8) match
        PyTorch Adam defaults.  We return (optimizer, init_fn) so
        the caller can create state per param set.
    MOCK INJECTION:
        Use optax.set_to_zero() as a no-op optimizer for gradient
        isolation tests.
    """
    return optax.adam(lr)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_train_step(criterion, optimizer_g, optimizer_d):
    """Build and return the JIT-compiled train_step closure.

    We wrap this in a factory so that `criterion`, `optimizer_g`, and
    `optimizer_d` are captured at trace time.  optax GradientTransformation
    objects are Python callables (not JAX arrays) and cannot be passed
    as arguments to a @jit-traced function — they must be closed over.
    """

    @jit
    def train_step(g_params, d_params, opt_state_g, opt_state_d,
                   real_data, key):
        """Single GAN training step (discriminator + generator).

        PYTORCH EQUIVALENT:
            One iteration of the training for-loop (both D and G
            updates).
        TRANSLATION NOTES:
            - torch.randn → jax.random.normal with explicit keys.
            - loss.backward() + optimizer.step() →
              jax.value_and_grad + optax.update + param arithmetic.
            - .detach() on fake_data for D training is implicit:
              we simply don't differentiate G params in the D loss.
            - optimizer.zero_grad() is unnecessary — JAX grads are
              freshly computed each call.
            - No print() or side effects here (required by rules).
            - optimizer_g / optimizer_d are captured via closure, not
              passed as args, because they are non-array Python objects.
        MOCK INJECTION:
            Pass small (e.g. 4×1) real_data and pre-built params to
            verify shapes and gradient flow.
        """
        latent_dim = g_params['linear1']['w'].shape[0]
        batch_size = real_data.shape[0]
        key, k1, k2 = random.split(key, 3)

        real_labels = jnp.ones((batch_size, 1))
        fake_labels = jnp.zeros((batch_size, 1))

        # --- Train Discriminator -------------------------------------------
        latent_samples = random.normal(k1, (batch_size, latent_dim))
        fake_data_detached = generator_forward(g_params, latent_samples)
        # We stop gradient through G by not including g_params in the
        # differentiated arguments (argnums=0 differentiates d_params only).

        def d_loss_fn(d_p):
            real_loss = criterion(discriminator_forward(d_p, real_data),
                                  real_labels)
            fake_loss = criterion(discriminator_forward(d_p, fake_data_detached),
                                  fake_labels)
            return real_loss + fake_loss

        loss_d, grads_d = value_and_grad(d_loss_fn)(d_params)
        updates_d, opt_state_d = optimizer_d.update(grads_d, opt_state_d,
                                                    d_params)
        d_params = optax.apply_updates(d_params, updates_d)

        # --- Train Generator -----------------------------------------------
        latent_samples = random.normal(k2, (batch_size, latent_dim))

        def g_loss_fn(g_p):
            fake_data = generator_forward(g_p, latent_samples)
            return criterion(discriminator_forward(d_params, fake_data),
                             real_labels)

        loss_g, grads_g = value_and_grad(g_loss_fn)(g_params)
        updates_g, opt_state_g = optimizer_g.update(grads_g, opt_state_g,
                                                    g_params)
        g_params = optax.apply_updates(g_params, updates_g)

        return g_params, d_params, opt_state_g, opt_state_d, loss_d, loss_g, key

    return train_step


def train_model(X, y, g_params, d_params, optimizer_g, optimizer_d,
                opt_state_g, opt_state_d, criterion, num_epochs, key):
    """Full training loop.

    PYTORCH EQUIVALENT:
        The for-loop over `epochs` in the original script, including
        the every-100-epoch print statement.
    TRANSLATION NOTES:
        - train_step is @jit compiled; all side effects (print) live
          here in the outer Python loop.
        - The key is threaded through each step so that each
          iteration draws fresh random latent samples.
        - Losses are pulled out of JAX arrays via .item() for
          printing, matching the PyTorch f-string format exactly.
    MOCK INJECTION:
        Pass num_epochs=1, small real_data, and pre-built params /
        optimizer states.  Verify that returned params differ from
        input (gradients flowed) and that no exceptions are raised.
    """
    real_data = X
    train_step = _make_train_step(criterion, optimizer_g, optimizer_d)

    for epoch in range(num_epochs):
        (g_params, d_params, opt_state_g, opt_state_d,
         loss_d, loss_g, key) = train_step(
            g_params, d_params, opt_state_g, opt_state_d,
            real_data, key)

        # Log progress every 100 epochs — outside @jit
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] - "
                  f"Loss D: {loss_d.item():.4f}, "
                  f"Loss G: {loss_g.item():.4f}")

    return g_params, d_params, opt_state_g, opt_state_d, key


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Entry point — mirrors the original script's top-level execution.

    PYTORCH EQUIVALENT:
        Everything from torch.manual_seed(42) through the final
        print of generated data.
    TRANSLATION NOTES:
        - torch.manual_seed(42) → jax.random.PRNGKey(42), then
          split for each consuming function.
        - Call order preserved: seed → generate_data → make_model →
          make_criterion → make_optimizer (G then D) → train →
          generate samples.
        - The final generation block omits `with torch.no_grad():`
          because JAX does not track gradients by default.
    MOCK INJECTION:
        Not applicable — integration entry point.  Test individual
        functions instead.
    """
    key = random.PRNGKey(42)
    # Split keys preserving call order: data, model, training
    key, key_data, key_model, key_train = random.split(key, 4)

    latent_dim = 10
    data_dim = 1
    epochs = 1000

    # --- Data (matches original generate_data position) --------------------
    X, y = generate_data(key_data)

    # --- Model (matches original make_model position) ----------------------
    g_params, d_params = make_model(key_model, latent_dim, data_dim)

    # --- Criterion ---------------------------------------------------------
    criterion = make_criterion()

    # --- Optimizers (G then D, matching original order) --------------------
    optimizer_g = make_optimizer(lr=0.001)
    optimizer_d = make_optimizer(lr=0.001)
    opt_state_g = optimizer_g.init(g_params)
    opt_state_d = optimizer_d.init(d_params)

    # --- Training ----------------------------------------------------------
    g_params, d_params, opt_state_g, opt_state_d, key_train = train_model(
        X, y, g_params, d_params,
        optimizer_g, optimizer_d,
        opt_state_g, opt_state_d,
        criterion, epochs, key_train)

    # --- Generate new samples with trained Generator -----------------------
    key_train, key_gen = random.split(key_train)
    latent_samples = random.normal(key_gen, (5, latent_dim))
    generated_data = generator_forward(g_params, latent_samples)
    print(f"Generated data: {generated_data.tolist()}")


if __name__ == '__main__':
    main()