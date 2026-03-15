import jax
import jax.numpy as jnp
from jax import random, value_and_grad
import flax.linen as nn
import optax

class Generator(nn.Module):
    latent_dim: int
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
    bce = - (targets * jnp.log(predictions + 1e-8) + (1 - targets) * jnp.log(1 - predictions + 1e-8))
    return jnp.mean(bce)

def train_step(G_params, D_params, G_opt_state, D_opt_state, real_data, key, latent_dim, G, D, G_optimizer, D_optimizer):
    # Train Discriminator
    key, subkey = random.split(key)
    latent_samples = random.normal(subkey, (real_data.shape[0], latent_dim))
    # FIXED: Use stop_gradient to match PyTorch's .detach()
    fake_data = jax.lax.stop_gradient(G.apply({'params': G_params}, latent_samples))
    
    real_labels = jnp.ones((real_data.shape[0], 1))
    fake_labels = jnp.zeros((real_data.shape[0], 1))
    
    def d_loss_fn(D_params):
        real_logits = D.apply({'params': D_params}, real_data)
        fake_logits = D.apply({'params': D_params}, fake_data)
        real_loss = bce_loss(real_logits, real_labels)
        fake_loss = bce_loss(fake_logits, fake_labels)
        loss = real_loss + fake_loss
        return loss
    
    d_loss, d_grads = value_and_grad(d_loss_fn)(D_params)
    D_updates, D_opt_state = D_optimizer.update(d_grads, D_opt_state, D_params)
    D_params = optax.apply_updates(D_params, D_updates)
    
    # Train Generator
    key, subkey = random.split(key)
    latent_samples = random.normal(subkey, (real_data.shape[0], latent_dim))
    
    def g_loss_fn(G_params):
        fake_data = G.apply({'params': G_params}, latent_samples)
        logits = D.apply({'params': D_params}, fake_data)
        loss = bce_loss(logits, real_labels) 
        return loss
    
    g_loss, g_grads = value_and_grad(g_loss_fn)(G_params)
    G_updates, G_opt_state = G_optimizer.update(g_grads, G_opt_state, G_params)
    G_params = optax.apply_updates(G_params, G_updates)
    
    return G_params, D_params, G_opt_state, D_opt_state, d_loss, g_loss, key

def main():
    """Main function to execute the training and generation of samples.
    This function initializes the model parameters, trains the Generator (G) 
    and Discriminator (D) models, and generates new samples after training.
    """
    # FIXED: Change random seed from 0 to 42
    key = random.PRNGKey(42)
    latent_dim = 10
    data_dim = 1
    
    # FIXED: Match PyTorch's two-step scaling: rand(0,1) * 2 - 1
    key, subkey = random.split(key)
    real_data = random.uniform(subkey, (100, data_dim)) * 2 - 1
    
    G = Generator(latent_dim=latent_dim, output_dim=data_dim)
    D = Discriminator(input_dim=data_dim)
    
    # FIXED: Extract 'params' from init() result
    key, subkey = random.split(key)
    G_params = G.init(subkey, jnp.ones((1, latent_dim)))['params']
    key, subkey = random.split(key)
    D_params = D.init(subkey, jnp.ones((1, data_dim)))['params']
    
    G_optimizer = optax.adam(learning_rate=0.001)
    D_optimizer = optax.adam(learning_rate=0.001)
    G_opt_state = G_optimizer.init(G_params)
    D_opt_state = D_optimizer.init(D_params)
    
    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        G_params, D_params, G_opt_state, D_opt_state, d_loss, g_loss, key = train_step(
            G_params, D_params, G_opt_state, D_opt_state, real_data, key, latent_dim, G, D, G_optimizer, D_optimizer
        )
        
        # Log progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss D: {d_loss:.4f}, Loss G: {g_loss:.4f}")
    
    # Generate new samples with the trained Generator
    latent_samples = random.normal(key, (5, latent_dim))
    generated_data = G.apply({'params': G_params}, latent_samples)
    print(f"Generated data: {generated_data.tolist()}")

if __name__ == "__main__":
    main()