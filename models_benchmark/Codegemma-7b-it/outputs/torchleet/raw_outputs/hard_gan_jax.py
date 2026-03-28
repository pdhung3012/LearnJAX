import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# Define the Generator
class Generator(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        x = nn.tanh(x)
        return x

# Define the Discriminator
class Discriminator(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.leaky_relu(x, 0.2)
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x, 0.2)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        return x

# Generate synthetic data for training
rng = jax.random.PRNGKey(42)
real_data = jax.random.normal(rng, (100, 1)) * 2 - 1  # 100 samples in the range [-1, 1]

# Initialize models, loss, and optimizers
latent_dim = 10
data_dim = 1
G = Generator()
D = Discriminator()

loss_fn = optax.sigmoid_binary_cross_entropy_with_logits
opt_G = optax.rmsprop(lr=0.001)
opt_D = optax.rmsprop(lr=0.001)

# Training loop
@jax.jit
def train_step(params_D, params_G, batch_stats_D, batch_stats_G, real_data):
    rng, key = jax.random.split(rng)
    latent_samples = jax.random.normal(key, (real_data.shape[0], latent_dim))
    fake_data = G.apply({'params': params_G, 'batch_stats': batch_stats_G}, latent_samples)

    # Train Discriminator
    def loss_fn_D(params_D, batch_stats_D):
        real_logits = D.apply({'params': params_D, 'batch_stats': batch_stats_D}, real_data)
        fake_logits = D.apply({'params': params_D, 'batch_stats': batch_stats_D}, fake_data)
        loss_D = (
            loss_fn(real_logits, jnp.ones_like(real_logits))
            + loss_fn(fake_logits, jnp.zeros_like(fake_logits))
        ) / 2
        return loss_D, (params_D, batch_stats_D)

    grad_fn_D = jax.value_and_grad(loss_fn_D, has_aux=True)
    loss_D, (params_D, batch_stats_D) = grad_fn_D(params_D, batch_stats_D)
    updates_D, new_opt_state_D = opt_D.update(loss_D, opt_D.state, params_D)
    params_D = optax.apply_updates(params_D, updates_D)

    # Train Generator
    def loss_fn_G(params_G, batch_stats_G):
        latent_samples = jax.random.normal(rng, (real_data.shape[0], latent_dim))
        fake_data = G.apply({'params': params_G, 'batch_stats': batch_stats_G}, latent_samples)
        fake_logits = D.apply({'params': params_D, 'batch_stats': batch_stats_D}, fake_data)
        loss_G = loss_fn(fake_logits, jnp.ones_like(fake_logits))
        return loss_G, (params_G, batch_stats_G)

    grad_fn_G = jax.value_and_grad(loss_fn_G, has_aux=True)
    loss_G, (params_G, batch_stats_G) = grad_fn_G(params_G, batch_stats_G)
    updates_G, new_opt_state_G = opt_G.update(loss_G, opt_G.state, params_G)
    params_G = optax.apply_updates(params_G, updates_G)

    return params_D, params_G, batch_stats_D, batch_stats_G, rng

# Training
num_epochs = 1000
params_D, params_G, batch_stats_D, batch_stats_G, rng = jax.jit(train_step)(
    D.init(rng, jnp.ones((1,))),
    G.init(rng, jnp.ones((1,))),
    D.init_batch_stats(rng, jnp.ones((1,))),
    G.init_batch_stats(rng, jnp.ones((1,))),
    real_data
)

for epoch in range(num_epochs):
    params_D, params_G, batch_stats_D, batch_stats_G, rng = jax.jit(train_step)(
        params_D, params_G, batch_stats_D, batch_stats_G, real_data
    )

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}")

# Generate new samples with the trained Generator
rng, key = jax.random.split(rng)
latent_samples = jax.random.normal(key, (5, latent_dim))
generated_data = G.apply({'params': params_G, 'batch_stats': batch_stats_G}, latent_samples)
print(f"Generated data: {generated_data.tolist()}")