# pip install jax flax optax

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state


# Define the Generator
class Generator(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        x = nn.tanh(x)
        return x


# Define the Discriminator
class Discriminator(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        return x


def bce_loss(preds, labels, eps=1e-7):
    preds = jnp.clip(preds, eps, 1.0 - eps)
    return -jnp.mean(labels * jnp.log(preds) + (1.0 - labels) * jnp.log(1.0 - preds))


@jax.jit
def train_discriminator(g_state, d_state, real_data, latent_samples):
    fake_data = g_state.apply_fn({"params": g_state.params}, latent_samples)
    fake_data = jax.lax.stop_gradient(fake_data)  # like .detach()

    batch_size = real_data.shape[0]
    real_labels = jnp.ones((batch_size, 1))
    fake_labels = jnp.zeros((batch_size, 1))

    def loss_fn(d_params):
        real_preds = d_state.apply_fn({"params": d_params}, real_data)
        fake_preds = d_state.apply_fn({"params": d_params}, fake_data)

        real_loss = bce_loss(real_preds, real_labels)
        fake_loss = bce_loss(fake_preds, fake_labels)
        return real_loss + fake_loss

    loss, grads = jax.value_and_grad(loss_fn)(d_state.params)
    d_state = d_state.apply_gradients(grads=grads)
    return d_state, loss


@jax.jit
def train_generator(g_state, d_state, latent_samples):
    batch_size = latent_samples.shape[0]
    real_labels = jnp.ones((batch_size, 1))

    def loss_fn(g_params):
        fake_data = g_state.apply_fn({"params": g_params}, latent_samples)
        fake_preds = d_state.apply_fn({"params": d_state.params}, fake_data)
        return bce_loss(fake_preds, real_labels)

    loss, grads = jax.value_and_grad(loss_fn)(g_state.params)
    g_state = g_state.apply_gradients(grads=grads)
    return g_state, loss


def main():
    # Generate synthetic data for training
    key = jax.random.PRNGKey(42)
    key, data_key, g_init_key, d_init_key = jax.random.split(key, 4)

    real_data = jax.random.uniform(
        data_key,
        shape=(100, 1),
        minval=-1.0,
        maxval=1.0
    )  # 100 samples in the range [-1, 1]

    # Initialize models and optimizers
    latent_dim = 10
    data_dim = 1

    G = Generator(output_dim=data_dim)
    D = Discriminator()

    g_params = G.init(g_init_key, jnp.ones((1, latent_dim)))["params"]
    d_params = D.init(d_init_key, jnp.ones((1, data_dim)))["params"]

    optimizer_G = optax.adam(learning_rate=0.001)
    optimizer_D = optax.adam(learning_rate=0.001)

    g_state = train_state.TrainState.create(
        apply_fn=G.apply,
        params=g_params,
        tx=optimizer_G,
    )

    d_state = train_state.TrainState.create(
        apply_fn=D.apply,
        params=d_params,
        tx=optimizer_D,
    )

    # Training loop
    epochs = 1000
    batch_size = real_data.shape[0]

    for epoch in range(epochs):
        key, d_key, g_key = jax.random.split(key, 3)

        d_latent_samples = jax.random.normal(d_key, (batch_size, latent_dim))
        g_latent_samples = jax.random.normal(g_key, (batch_size, latent_dim))

        d_state, loss_D = train_discriminator(
            g_state, d_state, real_data, d_latent_samples
        )

        g_state, loss_G = train_generator(
            g_state, d_state, g_latent_samples
        )

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] - "
                f"Loss D: {float(loss_D):.4f}, Loss G: {float(loss_G):.4f}"
            )

    # Generate new samples with the trained Generator
    key, sample_key = jax.random.split(key)
    latent_samples = jax.random.normal(sample_key, (5, latent_dim))
    generated_data = G.apply({"params": g_state.params}, latent_samples)
    print(f"Generated data: {generated_data.tolist()}")


if __name__ == "__main__":
    main()