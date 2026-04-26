"""JAX translation of h4: small GAN on 1-D real data.

Faithful to PyTorch:
- Generator: Linear(10->128, ReLU), Linear(128->256, ReLU), Linear(256->1, Tanh).
- Discriminator: Linear(1->256, LeakyReLU 0.2), Linear(256->128, LeakyReLU 0.2),
  Linear(128->1, Sigmoid).
- Real data: 100 uniform samples in [-1, 1]. BCE loss. Adam(lr=1e-3) for both.
- Per epoch: D step on (real, fake_detached); G step on fresh latents.
- 1000 epochs, log every 100. Print 5 generated samples at the end.

Speed notes: jit'd D-step + G-step run inside one fused XLA program each;
typical speed result: JAX faster than PyTorch on CPU for this size, because of
Python overhead per step in PyTorch.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


class Generator(nn.Module):
    output_dim: int = 1

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(256)(x))
        return jnp.tanh(nn.Dense(self.output_dim)(x))


class Discriminator(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.leaky_relu(nn.Dense(256)(x), negative_slope=0.2)
        x = nn.leaky_relu(nn.Dense(128)(x), negative_slope=0.2)
        return jax.nn.sigmoid(nn.Dense(1)(x))


def bce_loss(pred, target, eps=1e-7):
    pred = jnp.clip(pred, eps, 1 - eps)
    return -jnp.mean(target * jnp.log(pred) + (1 - target) * jnp.log(1 - pred))


def main():
    key = jax.random.PRNGKey(42)
    key, k_real = jax.random.split(key)
    real_data = jax.random.uniform(k_real, (100, 1)) * 2 - 1

    latent_dim = 10
    G = Generator(output_dim=1)
    D = Discriminator()

    key, kg, kd = jax.random.split(key, 3)
    G_params = G.init(kg, jnp.zeros((1, latent_dim)))
    D_params = D.init(kd, jnp.zeros((1, 1)))

    g_opt = optax.adam(1e-3)
    d_opt = optax.adam(1e-3)
    g_state = g_opt.init(G_params)
    d_state = d_opt.init(D_params)

    def d_loss(D_params, G_params, real, latent):
        real_labels = jnp.ones((real.shape[0], 1))
        fake_labels = jnp.zeros((real.shape[0], 1))
        fake = G.apply(G_params, latent)
        d_real = D.apply(D_params, real)
        d_fake = D.apply(D_params, fake)
        return bce_loss(d_real, real_labels) + bce_loss(d_fake, fake_labels)

    def g_loss(G_params, D_params, latent):
        fake = G.apply(G_params, latent)
        d_fake = D.apply(D_params, fake)
        real_labels = jnp.ones((latent.shape[0], 1))
        return bce_loss(d_fake, real_labels)

    @jax.jit
    def step(G_params, D_params, g_state, d_state, real, key):
        k1, k2 = jax.random.split(key)
        latent_d = jax.random.normal(k1, (real.shape[0], latent_dim))
        loss_D, dD = jax.value_and_grad(d_loss)(D_params, G_params, real, latent_d)
        d_updates, d_state = d_opt.update(dD, d_state)
        D_params = optax.apply_updates(D_params, d_updates)

        latent_g = jax.random.normal(k2, (real.shape[0], latent_dim))
        loss_G, dG = jax.value_and_grad(g_loss)(G_params, D_params, latent_g)
        g_updates, g_state = g_opt.update(dG, g_state)
        G_params = optax.apply_updates(G_params, g_updates)

        return G_params, D_params, g_state, d_state, loss_D, loss_G

    epochs = 1000
    for epoch in range(epochs):
        key, k = jax.random.split(key)
        G_params, D_params, g_state, d_state, loss_D, loss_G = step(
            G_params, D_params, g_state, d_state, real_data, k
        )
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}")

    key, kt = jax.random.split(key)
    latent = jax.random.normal(kt, (5, latent_dim))
    generated = G.apply(G_params, latent)
    print(f"Generated data: {generated.tolist()}")


if __name__ == "__main__":
    main()
