import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from flax import optax

class Generator(nn.Module):
    @nn.compact
    def __init__(self, input_dim, output_dim):
        self.fc1 = nn.Dense(128)
        self.act_fc1 = nn.ReLU()
        self.fc2 = nn.Dense(256)
        self.act_fc2 = nn.ReLU()
        self.fc3 = nn.Dense(output_dim)
        self.act_fc3 = nn.Tanh()

    @nn.compact
    def __call__(self, x):
        x = self.fc1(x)
        x = self.act_fc1(x)
        x = self.fc2(x)
        x = self.act_fc2(x)
        x = self.fc3(x)
        x = self.act_fc3(x)
        return x

class Discriminator(nn.Module):
    @nn.compact
    def __init__(self, input_dim):
        self.fc1 = nn.Dense(256)
        self.act_fc1 = nn.LeakyReLU(0.2)
        self.fc2 = nn.Dense(128)
        self.act_fc2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Dense(1)
        self.act_fc3 = nn.Sigmoid()

    @nn.compact
    def __call__(self, x):
        x = self.fc1(x)
        x = self.act_fc1(x)
        x = self.fc2(x)
        x = self.act_fc2(x)
        x = self.fc3(x)
        x = self.act_fc3(x)
        return x

# Generate synthetic data for training
rng = jr.PRNGKey(42)
real_data = jnp.random.uniform(min=-1.0, max=1.0, shape=(100, 1), key=rng)

# Initialize models, loss, and optimizers
latent_dim = 10
data_dim = 1
G = Generator(latent_dim, data_dim)
D = Discriminator(data_dim)

criterion = nn.BCE
loss = criterion()

opt_G = optax.adam(G.parameters(), learning_rate=0.001)
opt_D = optax.adam(D.parameters(), learning_rate=0.001)

# Training loop
epochs = 1000

@jax.jit
def train_step(rng, G, D, real_data, latent_samples, criterion, opt_G, opt_D):
    # Train Discriminator
    fake_data = G.apply(latent_samples, rng)
    real_labels = jnp.ones(real_data.shape[0], dtype=jnp.float32)
    fake_labels = jnp.zeros(real_data.shape[0], dtype=jnp.float32)

    grads_D, _ = jax.value_and_grad(D.value)(real_data)(rng)
    loss_D, grads_D = criterion.backward(real_labels, grads_D)
    opt_D.update(D.parameters(), opt_D.get_update(grads_D))

    grads_D, _ = jax.value_and_grad(D.value)(fake_data)(rng)
    loss_D += criterion.backward(fake_labels, grads_D)

    # Train Generator
    grads_G, _ = jax.value_and_grad(G.value)(latent_samples)(rng)
    loss_G = criterion.backward(real_labels, grads_G)

    opt_G.update(G.parameters(), opt_G.get_update(grads_G))

    return loss_D, loss_G

for epoch in range(epochs):
    rng = jr.PRNGKey(epoch)
    loss_D, loss_G = train_step(rng, G, D, real_data, jr.PRNGKey(epoch), criterion, opt_G, opt_D)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}")

# Generate new samples with the trained Generator
latent_samples = jr.PRNGKey(epoch)
generated_data = G.apply(latent_samples, jr.PRNGKey(epoch))
print(f"Generated data: {generated_data.tolist()}")


This JAX code replicates the PyTorch code strictly using `flax.linen.Module`, `jax.numpy`, `jax`, and `optax`. The training loop is converted to use `jax.value_and_grad` and `@jax.jit`. The data is generated using simple `numpy` random data generators.