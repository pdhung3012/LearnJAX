import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import pickle
from functools import partial
# Rule 1 & 3: Model equivalent defined at module level
def simple_model(params, x):
    """
    PYTORCH EQUIVALENT: class SimpleModel(nn.Module) -> self.fc(x)
    TRANSLATION NOTES: JAX models are pure functions. 'params' is a dict containing 'w' and 'b'.
    MOCK INJECTION: Inject params={'w': jnp.array([[1.0]]), 'b': jnp.array([0.0])}
    """
    return jnp.dot(x, params['w']) + params['b']

# Rule 2: Wrap data generation
def generate_data(key):
    """
    PYTORCH EQUIVALENT: torch.rand(100, 1) and torch.randn(100, 1)
    TRANSLATION NOTES: Uses jax.random.split to derive new keys for X and noise.
    MOCK INJECTION: Pass a key generated from jax.random.PRNGKey(42).
    """
    key_x, key_noise = random.split(key)
    X = random.uniform(key_x, (100, 1))
    y = 3 * X + 2 + random.normal(key_noise, (100, 1)) * 0.1
    return X, y

# Rule 4: Instantiate model (Initialize parameters)
def make_model(key):
    """
    PYTORCH EQUIVALENT: model = SimpleModel() (Implicit weight init)
    TRANSLATION NOTES: In JAX, we must explicitly initialize the weight matrix and bias vector.
    MOCK INJECTION: Use jax.random.PRNGKey(0) to get deterministic initial weights.
    """
    w_key, b_key = random.split(key)
    params = {
        'w': random.normal(w_key, (1, 1)),
        'b': jnp.zeros((1,))
    }
    return params

# Rule 5: Return loss function
def make_criterion():
    """
    PYTORCH EQUIVALENT: nn.MSELoss()
    TRANSLATION NOTES: Returns a pure function calculating Mean Squared Error.
    MOCK INJECTION: Test with jnp.array([1.0]), jnp.array([2.0]) to expect 1.0.
    """
    def mse_loss(params, x, y):
        preds = simple_model(params, x)
        return jnp.mean((preds - y) ** 2)
    return mse_loss

# Rule 6: Return optimizer (Step function in JAX)
def make_optimizer(lr=0.01):
    """
    PYTORCH EQUIVALENT: optim.SGD(model.parameters(), lr=0.01)
    TRANSLATION NOTES: Returns a function that performs a manual gradient descent update.
    MOCK INJECTION: Use a gradient of 1.0 and lr 0.01 to see a 0.01 parameter shift.
    """
    def update(params, grads):
        return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return update

# Rule 5 & 7: @jit applied to the training step logic
# Rule 5 & 7: Updated with static_argnames to fix the TypeError
@partial(jit, static_argnames=['criterion', 'optimizer_update'])
def train_step(params, X, y, criterion, optimizer_update):
    """
    PYTORCH EQUIVALENT: optimizer.zero_grad(), loss.backward(), optimizer.step()
    TRANSLATION NOTES: Using functools.partial is the standard JAX way to 
                       pass static arguments to the @jit decorator.
    MOCK INJECTION: Use static X, y and check if params change after one call.
    """
    loss, grads = value_and_grad(criterion)(params, X, y)
    new_params = optimizer_update(params, grads)
    return new_params, loss

def train_model(X, y, params, optimizer_update, criterion, num_epochs):
    """
    PYTORCH EQUIVALENT: for epoch in range(epochs): ...
    TRANSLATION NOTES: Standard Python loop. Logging occurs here outside of @jit.
    MOCK INJECTION: Pass small X, y and num_epochs=1 to verify flow.
    """
    for epoch in range(num_epochs):
        params, loss = train_step(params, X, y, criterion, optimizer_update)
    return params

def main():
    # Rule 8: Manual seed equivalent
    main_key = random.PRNGKey(42)
    model_key, data_key = random.split(main_key)
    
    # 1. Initialize weights
    params = make_model(model_key)
    
    # 2. Generate data
    X, y = generate_data(data_key)
    
    criterion = make_criterion()
    optimizer_update = make_optimizer(lr=0.01)
    
    # Training
    epochs = 100
    params = train_model(X, y, params, optimizer_update, criterion, epochs)

    # Save the model (Rule 11: Using pickle for dict storage)
    with open("model.pkl", "wb") as f:
        pickle.dump(params, f)

    # Load the model back
    with open("model.pkl", "rb") as f:
        loaded_params = pickle.load(f)

    # Rule 10: Verify identical output structure
    X_test = jnp.array([[0.5], [1.0], [1.5]])
    predictions = simple_model(loaded_params, X_test)
    print(f"Predictions after loading: {predictions}")

if __name__ == '__main__':
    main()