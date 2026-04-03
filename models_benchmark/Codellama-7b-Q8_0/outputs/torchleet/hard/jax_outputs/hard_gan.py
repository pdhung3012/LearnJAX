import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax

# Define the Generator
def generator(params, x):
    h1 = jnp.maximum(0, jnp.dot(x, params['W1']) + params['b1'])
    h2 = jnp.maximum(0, jnp.dot(h1, params['W2']) + params['b2'])
    out = jnp.tanh(jnp.dot(h2, params['W3']) + params['b3'])
    return out

# Define the Discriminator (LeakyReLU with negative_slope=0.2)
def leaky_relu(x, negative_slope=0.2):
    return jnp.where(x > 0, x, negative_slope * x)

def discriminator(params, x):
    h1 = leaky_relu(jnp.dot(x, params['W1']) + params['b1'])
    h2 = leaky_relu(jnp.dot(h1, params['W2']) + params['b2'])
    out = jax.nn.sigmoid(jnp.dot(h2, params['W3']) + params['b3'])
    return out

# Initialize parameters with proper JAX random
# PyTorch nn.Linear uses Kaiming uniform: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
key = jax.random.PRNGKey(42)
input_dim = 10
output_dim = 1
latent_dim = 10
data_dim = 1

def init_weight(key, shape):
    fan_in = shape[0]
    bound = 1.0 / jnp.sqrt(fan_in)
    return jax.random.uniform(key, shape, minval=-bound, maxval=bound)

key, *subkeys = jax.random.split(key, 7)
params_generator = {
    'W1': init_weight(subkeys[0], (input_dim, 128)),
    'b1': jax.random.uniform(subkeys[0], (128,), minval=-1.0/jnp.sqrt(10), maxval=1.0/jnp.sqrt(10)),
    'W2': init_weight(subkeys[1], (128, 256)),
    'b2': jax.random.uniform(subkeys[1], (256,), minval=-1.0/jnp.sqrt(128), maxval=1.0/jnp.sqrt(128)),
    'W3': init_weight(subkeys[2], (256, output_dim)),
    'b3': jax.random.uniform(subkeys[2], (output_dim,), minval=-1.0/jnp.sqrt(256), maxval=1.0/jnp.sqrt(256)),
}

params_discriminator = {
    'W1': init_weight(subkeys[3], (data_dim, 256)),
    'b1': jax.random.uniform(subkeys[3], (256,), minval=-1.0/jnp.sqrt(1), maxval=1.0/jnp.sqrt(1)),
    'W2': init_weight(subkeys[4], (256, 128)),
    'b2': jax.random.uniform(subkeys[4], (128,), minval=-1.0/jnp.sqrt(256), maxval=1.0/jnp.sqrt(256)),
    'W3': init_weight(subkeys[5], (128, 1)),
    'b3': jax.random.uniform(subkeys[5], (1,), minval=-1.0/jnp.sqrt(128), maxval=1.0/jnp.sqrt(128)),
}

# Define loss functions
def loss_discriminator(params_dis, real_data, fake_data):
    eps = 1e-7
    real_loss = -jnp.mean(jnp.log(discriminator(params_dis, real_data) + eps))
    fake_loss = -jnp.mean(jnp.log(1 - discriminator(params_dis, fake_data) + eps))
    return real_loss + fake_loss

def loss_generator(params_gen, params_dis, latent_samples):
    eps = 1e-7
    fake_data = generator(params_gen, latent_samples)
    return -jnp.mean(jnp.log(discriminator(params_dis, fake_data) + eps))

# Define optimizers using optax
optimizer_gen = optax.adam(learning_rate=0.001)
opt_state_gen = optimizer_gen.init(params_generator)

optimizer_dis = optax.adam(learning_rate=0.001)
opt_state_dis = optimizer_dis.init(params_discriminator)

# Generate synthetic data for training
real_data = jax.random.uniform(key, shape=(100, 1)) * 2 - 1

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Train Discriminator
    key, subkey = jax.random.split(key)
    latent_samples = jax.random.normal(subkey, shape=(real_data.shape[0], latent_dim))
    # stop_gradient on fake_data matches PyTorch's .detach()
    fake_data = jax.lax.stop_gradient(generator(params_generator, latent_samples))

    loss_d, grads_dis = value_and_grad(loss_discriminator)(params_discriminator, real_data, fake_data)
    updates_dis, opt_state_dis = optimizer_dis.update(grads_dis, opt_state_dis)
    params_discriminator = optax.apply_updates(params_discriminator, updates_dis)

    # Train Generator
    key, subkey = jax.random.split(key)
    latent_samples = jax.random.normal(subkey, shape=(real_data.shape[0], latent_dim))

    loss_g, grads_gen = value_and_grad(loss_generator)(params_generator, params_discriminator, latent_samples)
    updates_gen, opt_state_gen = optimizer_gen.update(grads_gen, opt_state_gen)
    params_generator = optax.apply_updates(params_generator, updates_gen)

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss D: {loss_d:.4f}, Loss G: {loss_g:.4f}")

# Generate new samples with the trained Generator
key, subkey = jax.random.split(key)
latent_samples = jax.random.normal(subkey, shape=(5, latent_dim))
generated_data = generator(params_generator, latent_samples)
print(f"Generated data: {generated_data.tolist()}")
