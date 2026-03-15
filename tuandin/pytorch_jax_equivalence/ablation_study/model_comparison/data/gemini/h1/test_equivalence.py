import torch
import numpy as np
import jax.numpy as jnp
import jax

# Rule 1: Imports
from pytorch_refactored import SimpleModel, make_model, make_criterion, make_optimizer, train_model as pt_train_model
from jax_code_fixed import simple_model as jax_model, make_criterion as jax_make_criterion, make_optimizer as jax_make_optimizer, train_step as jax_train_step, train_model as jax_train_model

# Rule 2: Shared constants
X_NP = np.linspace(0, 1, 10).reshape(10, 1).astype(np.float32)
Y_NP = (X_NP * 3.0 + 2.0).astype(np.float32)
W_INIT = np.array([[0.5]], dtype=np.float32) # PyTorch format (Out, In)
B_INIT = np.array([0.1], dtype=np.float32)
X_TEST_NP = np.array([[0.5], [1.0]], dtype=np.float32)

def get_fresh_pt():
    model = make_model()
    with torch.no_grad():
        # PyTorch expects (Out, In) -> (1, 1)
        model.fc.weight.copy_(torch.from_numpy(W_INIT))
        model.fc.bias.copy_(torch.from_numpy(B_INIT))
    return model

def get_fresh_jax():
    return {
        # JAX dot(x, w) expects (In, Out). 
        # For (1,1) it's the same, but .T ensures architectural consistency.
        'w': jnp.array(W_INIT).T, 
        'b': jnp.array(B_INIT)
    }

def test_forward_pass(tol=1e-6):
    pt_model = get_fresh_pt()
    jax_params = get_fresh_jax()
    pt_out = pt_model(torch.from_numpy(X_NP)).detach().numpy()
    jax_out = np.array(jax_model(jax_params, jnp.array(X_NP)))
    diff = np.abs(pt_out - jax_out).max()
    return diff < tol, diff

def test_loss(tol=1e-6):
    pt_model = get_fresh_pt()
    jax_params = get_fresh_jax()
    pt_loss = make_criterion()(pt_model(torch.from_numpy(X_NP)), torch.from_numpy(Y_NP)).item()
    jax_loss = float(jax_make_criterion()(jax_params, jnp.array(X_NP), jnp.array(Y_NP)))
    diff = abs(pt_loss - jax_loss)
    return diff < tol, diff

def test_train_step(tol=1e-6):
    # PT Step
    pt_model = get_fresh_pt()
    optimizer = make_optimizer(pt_model)
    criterion = make_criterion()
    optimizer.zero_grad()
    loss = criterion(pt_model(torch.from_numpy(X_NP)), torch.from_numpy(Y_NP))
    loss.backward()
    optimizer.step()
    
    # JAX Step
    jax_params = get_fresh_jax()
    # Ensure JAX optimizer uses exactly 0.01
    jax_params_new, _ = jax_train_step(
        jax_params, jnp.array(X_NP), jnp.array(Y_NP), 
        jax_make_criterion(), jax_make_optimizer(lr=0.01)
    )
    
    pt_w = pt_model.fc.weight.data.numpy()
    # CRITICAL: Transpose JAX back to (Out, In) to compare with PT
    jax_w = np.array(jax_params_new['w']).T 
    
    diff = np.abs(pt_w - jax_w).max()
    return diff < tol, diff

def test_convergence(tol=1e-2):
    epochs = 100
    pt_model = get_fresh_pt()
    pt_train_model(torch.from_numpy(X_NP), torch.from_numpy(Y_NP), 
                   pt_model, make_optimizer(pt_model), make_criterion(), epochs)
    pt_preds = pt_model(torch.from_numpy(X_TEST_NP)).detach().numpy()
    
    jax_params = jax_train_model(jnp.array(X_NP), jnp.array(Y_NP), get_fresh_jax(), 
                                 jax_make_optimizer(lr=0.01), jax_make_criterion(), epochs)
    jax_preds = np.array(jax_model(jax_params, jnp.array(X_TEST_NP)))
    
    diff = np.abs(pt_preds - jax_preds).max()
    return diff < tol, diff

if __name__ == "__main__":
    results = [
        ("Forward Pass", test_forward_pass()),
        ("Loss Calc", test_loss()),
        ("Single Step", test_train_step()),
        ("100-Epoch Conv", test_convergence()),
    ]
    print(f"{'Test':<20} | {'Status':<7} | {'Max Diff'}")
    print("-" * 45)
    for name, (passed, diff) in results:
        print(f"{name:<20} | {'PASS' if passed else 'FAIL':<7} | {diff:.8f}")