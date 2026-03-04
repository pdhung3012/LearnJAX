import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
from functools import partial

# Define the Generator
class Generator(nn.Module):
    latent_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, z):
        x = nn.Dense(128)(z)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        x = nn.tanh(x)
        return x

# Define the Discriminator
class Discriminator(nn.Module):
    input_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        return x

def bce_loss(predictions, targets):
    eps = 1e-7  # small constant for numerical stability
    return -jnp.mean(targets * jnp.log(predictions + eps) + (1 - targets) * jnp.log(1 - predictions + eps))

def main():
    # Set up PRNG keys
    key = jax.random.PRNGKey(42)

    # Generate synthetic real data
    num_samples = 100
    data_dim = 1
    key, subkey = jax.random.split(key)
    real_data = jax.random.uniform(subkey, shape=(num_samples, data_dim), minval=-1, maxval=1)

    latent_dim = 10

    # Instantiate the models
    gen_model = Generator(latent_dim=latent_dim, output_dim=data_dim)
    disc_model = Discriminator(input_dim=data_dim)

    # Initialize model parameters
    key, subkey = jax.random.split(key)
    gen_params = gen_model.init(subkey, jnp.ones((1, latent_dim)))
    key, subkey = jax.random.split(key)
    disc_params = disc_model.init(subkey, jnp.ones((1, data_dim)))

    # Set up optimizers
    gen_optimizer = optax.adam(learning_rate=0.001)
    disc_optimizer = optax.adam(learning_rate=0.001)
    gen_opt_state = gen_optimizer.init(gen_params)
    disc_opt_state = disc_optimizer.init(disc_params)

    # Define a jitted discriminator training step.
    # No static_argnums needed here because batch_size is derived from real_data.shape
    @jax.jit
    def disc_train_step(disc_params, disc_opt_state, gen_params, real_data, key):
        batch_size = real_data.shape[0]
        key, subkey = jax.random.split(key)
        latent_samples = jax.random.normal(subkey, shape=(batch_size, latent_dim))
        
        # Stop gradient ensures we don't update Generator during this step
        fake_data = jax.lax.stop_gradient(gen_model.apply(gen_params, latent_samples))
        real_labels = jnp.ones((batch_size, 1))
        fake_labels = jnp.zeros((batch_size, 1))

        def loss_disc_fn(params):
            real_pred = disc_model.apply(params, real_data)
            fake_pred = disc_model.apply(params, fake_data)
            return bce_loss(real_pred, real_labels) + bce_loss(fake_pred, fake_labels)

        loss_D, grads = jax.value_and_grad(loss_disc_fn)(disc_params)
        updates, disc_opt_state = disc_optimizer.update(grads, disc_opt_state)
        disc_params = optax.apply_updates(disc_params, updates)
        return disc_params, disc_opt_state, loss_D, key

    # FIXED: Added static_argnums for batch_size (argument index 3)
    @partial(jax.jit, static_argnums=(3,))
    def gen_train_step(gen_params, gen_opt_state, disc_params, batch_size, key):
        key, subkey = jax.random.split(key)
        latent_samples = jax.random.normal(subkey, shape=(batch_size, latent_dim))

        def loss_gen_fn(params):
            fake_data = gen_model.apply(params, latent_samples)
            pred = disc_model.apply(disc_params, fake_data)
            return bce_loss(pred, jnp.ones((batch_size, 1)))

        loss_G, grads = jax.value_and_grad(loss_gen_fn)(gen_params)
        updates, gen_opt_state = gen_optimizer.update(grads, gen_opt_state)
        gen_params = optax.apply_updates(gen_params, updates)
        return gen_params, gen_opt_state, loss_G, key

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        disc_params, disc_opt_state, loss_D, key = disc_train_step(
            disc_params, disc_opt_state, gen_params, real_data, key
        )
        gen_params, gen_opt_state, loss_G, key = gen_train_step(
            gen_params, gen_opt_state, disc_params, real_data.shape[0], key
        )
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}")

    # Generate new samples
    key, subkey = jax.random.split(key)
    latent_samples = jax.random.normal(subkey, shape=(5, latent_dim))
    generated_data = gen_model.apply(gen_params, latent_samples)
    print("Generated data:", np.array(generated_data).tolist())

if __name__ == "__main__":
    main()
    