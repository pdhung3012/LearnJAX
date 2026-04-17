import jax
import jax.numpy as jnp  # Ensured consistent import statement
from flax import linen as nn
from jax import random, grad, jit, vmap
import optax
from flax.training import train_state


class TransformerEncoderLayer(nn.Module):
    embed_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = True):
        attn = nn.SelfAttention(num_heads=self.num_heads,
                        qkv_features=self.embed_dim,
                        dropout_rate=self.dropout_rate,
                        deterministic=not train)(x)
        attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic=not train)
        x = x + attn
        x = nn.LayerNorm()(x)

        ff = nn.Dense(self.ff_dim)(x)
        ff = jax.nn.relu(ff)
        ff = nn.Dropout(rate=self.dropout_rate)(ff, deterministic=not train)
        ff = nn.Dense(self.embed_dim)(ff)
        x = x + ff
        x = nn.LayerNorm()(x)
        return x

class TransformerModel(nn.Module):
    input_dim: int
    embed_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    output_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Dense(self.embed_dim, name="embedding")(x)

        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(embed_dim=self.embed_dim,
                                        num_heads=self.num_heads,
                                        ff_dim=self.ff_dim,
                                        dropout_rate=self.dropout_rate)(x, train=train)

        x = jnp.mean(x, axis=1)
        x = nn.Dense(self.output_dim)(x)
        return x



def compute_loss(predictions, targets):
    return jnp.mean((predictions - targets) ** 2)



def create_train_state(rng, model, learning_rate, input_shape):
    params = model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jit
def train_step(state, batch, dropout_rng):
    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, batch['X'], train=True, rngs={'dropout': dropout_rng})
        loss = compute_loss(predictions, batch['y'])
        return loss, predictions
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, preds), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss



def train_model(X, y, num_epochs, key):
    model = TransformerModel(
        input_dim=1,
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        ff_dim=64,
        output_dim=1
    )
    state = create_train_state(key, model, learning_rate=0.001, input_shape=X.shape)

    dataset_size = X.shape[0]
    
    for epoch in range(num_epochs):
        key, subkey = random.split(key)
        perm = random.permutation(subkey, dataset_size)
        X_shuffled = X[perm]
        y_shuffled = y[perm]
        
        total_loss = 0.0
        num_batches = 0
        for i in range(0, dataset_size, 32):
            key, dropout_key = random.split(key)
            batch = {
                'X': X_shuffled[i:i + 32],
                'y': y_shuffled[i:i + 32]
            }
            state, batch_loss = train_step(state, batch, dropout_key)
            total_loss += batch_loss
            num_batches += 1
            
        if (epoch + 1) % 100 == 0:
            predictions = state.apply_fn({'params': state.params}, X, train=False)
            loss_value = compute_loss(predictions, y)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_value:.4f}")

    return state, model



def main():
    """Main entry point for the script."""
    # Example data generation with explicit PRNG key
    key = random.PRNGKey(0)
    num_samples = 100
    seq_length = 10
    input_dim = 1
    key, subkey = random.split(key)
    X = random.uniform(subkey, (num_samples, seq_length, input_dim))
    y = jnp.sum(X, axis=1)

    num_epochs = 1000
    state, model = train_model(X, y, num_epochs, key)
    
    # Testing on new data
    key, subkey = random.split(key)
    X_test = random.uniform(subkey, (2, seq_length, input_dim))
    predictions = state.apply_fn({'params': state.params}, X_test, train=False)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")



if __name__ == "__main__":
    main()