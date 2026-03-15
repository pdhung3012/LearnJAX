import jax
import jax.numpy as jnp

## --- Model Definition ---

def model_fn(params, x):
    """
    PYTORCH EQUIVALENT: LinearRegressionModel class and forward() method.
    TRANSLATION NOTES: Replaced nn.Module with a pure function. Parameters are explicit.
    MOCK INJECTION: params={'w': jnp.array([[2.0]]), 'b': jnp.array([3.0])} with x=1.0.
    """
    return jnp.dot(x, params['w']) + params['b']

## --- Component Factories ---

def generate_data(key):
    """
    PYTORCH EQUIVALENT: generate_data()
    TRANSLATION NOTES: Replaced torch.rand with jax.random. Requires PRNGKey splitting.
    MOCK INJECTION: Use jax.random.PRNGKey(42) for consistent data.
    """
    k1, k2 = jax.random.split(key)
    X = jax.random.uniform(k1, (100, 1)) * 10
    y = 2 * X + 3 + jax.random.normal(k2, (100, 1))
    return X, y

def make_model(key):
    """
    PYTORCH EQUIVALENT: LinearRegressionModel.__init__()
    TRANSLATION NOTES: Dictionary-based parameter initialization.
    MOCK INJECTION: jax.random.PRNGKey(0) for repeatable weights.
    """
    k1, k2 = jax.random.split(key)
    params = {
        'w': jax.random.normal(k1, (1, 1)),
        'b': jnp.zeros((1,))
    }
    return params

def make_criterion():
    """
    PYTORCH EQUIVALENT: nn.MSELoss()
    TRANSLATION NOTES: Returns a pure function for loss calculation.
    MOCK INJECTION: Test with pred=1.0, y=3.0 to get loss 4.0.
    """
    def mse_loss(predictions, targets):
        return jnp.mean((predictions - targets) ** 2)
    return mse_loss

def make_optimizer():
    """
    PYTORCH EQUIVALENT: optim.SGD(lr=0.01)
    TRANSLATION NOTES: In this manual SGD setup, the optimizer is just the learning rate.
    MOCK INJECTION: Returns 0.01.
    """
    return 0.01

## --- Training Logic ---

def train_step(params, X, y, lr, criterion_fn):
    """
    PYTORCH EQUIVALENT: loss.backward() and optimizer.step()
    TRANSLATION NOTES: Uses value_and_grad for gradients. Updates via tree_map.
    MOCK INJECTION: params, X, y, 0.01, and mse_loss function.
    """
    def loss_fn(p):
        preds = model_fn(p, X)
        return criterion_fn(preds, y)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss

def train_model(X, y, params, lr, criterion, num_epochs):
    """
    PYTORCH EQUIVALENT: train_model() loop.
    TRANSLATION NOTES: The train_step is JITed here. static_argnums=(4,) 
                       points to 'criterion_fn'.
    MOCK INJECTION: Runs training for specified epochs.
    """
    # FIX: Explicitly jit the function and mark the 5th argument (index 4) as static
    jit_step = jax.jit(train_step, static_argnums=(4,))
    
    for epoch in range(num_epochs):
        params, loss = jit_step(params, X, y, lr, criterion)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")
    return params

## --- Execution ---

def main():
    # 1. Set seed
    key = jax.random.PRNGKey(42)
    data_key, model_key = jax.random.split(key)

    # 2. Call order preserved
    X, y = generate_data(data_key)
    params = make_model(model_key)
    criterion = make_criterion()
    lr = make_optimizer()

    # 3. Train
    epochs = 1000
    params = train_model(X, y, params, lr, criterion, epochs)

    # 4. Display results
    w, b = params['w'], params['b']
    print(f"Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}")

    # 5. Testing
    X_test = jnp.array([[4.0], [7.0]])
    predictions = model_fn(params, X_test)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")

if __name__ == '__main__':
    main()