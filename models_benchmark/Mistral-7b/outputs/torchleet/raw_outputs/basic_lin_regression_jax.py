import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn
from flax import optax

# Generate synthetic data
rng = jr.PRNGKey(42)
X = jnp.ones((100, 1)) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + jr.normal(rng, (100, 1))  # Linear relationship with noise

class LinearRegressionModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return self.linear(x)

    @nn.compact
    def setup(self):
        self.linear = nn.Dense(1)

class LinearRegression(nn.Module):
    model = LinearRegressionModel

# Initialize the model, loss function, and optimizer
params = jax.random.normal(rng, LinearRegression.model.setup_rng_key(), LinearRegression.model.setup())
optimizer_state = optax.init_optimizer_state(optax.sgd, params, step_size=0.01)

@jax.jit
def train_step(params, optimizer_state, X, y):
    predictions = LinearRegression.model()(params)(X)
    loss = jnp.mean((predictions - y) ** 2)
    grads = jax.grad(loss)(params)
    updates, new_optimizer_state = optax.update(optimizer_state, grads)
    return params.at(0, jax.ops.index_update(params[0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:, 0], jax.ops.index_update(params[0][0, 0], jax.ops.index[:,