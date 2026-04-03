import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from flax import optax
import optax.scipy as scipy_optax

# Generate synthetic data
rng = jr.PRNGKey(42)
X = jnp.ones((100, 1)) * 10  # 100 data points between 0 and 10
y = 3 * X + 5 + jr.normal(rng, (100, 1))  # Linear relationship with noise

class LinearRegressionModel(nn.Module):
    @nn.compact
    def __call__(self, X):
        return self.linear(X)

    @nn.compact
    def setup(self):
        self.linear = nn.Dense(1)

# Initialize RNG and optimizer
rng = jr.PRNGKey(42)
rng_state = jr.PRNGState(rng)
opt_state = optax.init_optimizer(scipy_optax.sgd, step_size=0.01)

# Training loop
@jax.jit
def train_step(rng_state, opt_state, X, y):
    params = opt_state.params
    grads = jax.value_and_grad(LinearRegressionModel().__call__)(X)[1]
    updates, opt_state = optax.apply_updates(opt_state, [(params, grads)])
    return updates, opt_state

@jax.jit
def train(X, y, num_steps=100):
    rng_state = jr.PRNGState(rng)
    opt_state = opt_state
    for i in jax.range(num_steps):
        updates, opt_state = train_step(rng_state, opt_state, X, y)
        rng_state = jr.foldl(jax.ops.index_update, rng_state, jax.index_slice(0, 1, jnp.arange(num_steps)), jnp.zeros((num_steps,), jnp.int32))
    return opt_state

# Initialize the model and train
opt_state = train(X, y)

# Log loss to a simple text file
def log_loss(opt_state, X, y):
    loss = jnp.mean(jnp.square(LinearRegressionModel()(X) - y))
    jax.lib.io.save("loss.npy", loss)
    print(f"Loss: {loss:.4f}")

log_loss(opt_state, X, y)


This JAX code generates synthetic data, defines a Linear Regression Model using `flax.linen.Module`, initializes the optimizer, and trains the model using a custom `train_step` function. The loss is logged to a simple text file. Note that TensorBoard is not supported directly in JAX, so the logs are saved to a text file instead.