import torch
import jax
import jax.numpy as jnp
import numpy as np

# Import PyTorch components
from pytorch_refactored import (
    LinearRegressionModel,
    make_model as pt_make_model,
    make_criterion as pt_make_criterion,
    make_optimizer as pt_make_optimizer,
    train_model as pt_train_model
)

# Import JAX components
from jax_code_fixed import (
    model_fn as jax_model,
    make_criterion as jax_make_criterion,
    train_step as jax_train_step,
    train_model as jax_train_model
)

## --- Shared Hardcoded Tensors ---

X_np = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
y_np = np.array([[5.0], [7.0], [9.0]], dtype=np.float32)
W_np = np.array([[0.5]], dtype=np.float32)
B_np = np.array([0.2], dtype=np.float32)
X_test_np = np.array([[4.0], [7.0]], dtype=np.float32)

## --- Parameter Injection Helpers ---

def inject_pytorch_params(model, w_np, b_np):
    """Injects hardcoded params into PyTorch model bypassing init."""
    with torch.no_grad():
        model.linear.weight.copy_(torch.from_numpy(w_np.T))
        model.linear.bias.copy_(torch.from_numpy(b_np))
    return model

def make_jax_params(w_np, b_np):
    """Constructs the JAX params dict from numpy arrays."""
    return {
        'w': jnp.array(w_np),
        'b': jnp.array(b_np)
    }

## --- Tier 1: Unit Equivalence ---

def test_forward_pass(tol=1e-5):
    # Setup
    pt_model = inject_pytorch_params(pt_make_model(), W_np, B_np)
    jax_params = make_jax_params(W_np, B_np)
    
    # Execute
    pt_out = pt_model(torch.from_numpy(X_np)).detach().numpy()
    jax_out = jax_model(jax_params, jnp.array(X_np))
    
    diff = np.abs(pt_out - jax_out).max()
    assert diff < tol, f"Forward pass failed. Diff: {diff}"
    print(f"PASS: test_forward_pass (diff={diff:.2e})")

def test_loss(tol=1e-5):
    # Setup
    pt_criterion = pt_make_criterion()
    jax_criterion = jax_make_criterion()
    
    y_pred_np = np.array([[4.5], [6.5], [8.5]], dtype=np.float32)
    
    # Execute
    pt_loss = pt_criterion(torch.from_numpy(y_pred_np), torch.from_numpy(y_np)).item()
    jax_loss = jax_criterion(jnp.array(y_pred_np), jnp.array(y_np))
    
    diff = abs(pt_loss - jax_loss)
    assert diff < tol, f"Loss equivalence failed. Diff: {diff}"
    print(f"PASS: test_loss (diff={diff:.2e})")

def test_train_step(tol=1e-5):
    # Setup
    pt_model = inject_pytorch_params(pt_make_model(), W_np, B_np)
    pt_opt = pt_make_optimizer(pt_model)
    pt_crit = pt_make_criterion()
    
    jax_params = make_jax_params(W_np, B_np)
    jax_crit = jax_make_criterion()
    lr = 0.01

    # PyTorch step
    pt_opt.zero_grad()
    pt_out = pt_model(torch.from_numpy(X_np))
    pt_loss = pt_crit(pt_out, torch.from_numpy(y_np))
    pt_loss.backward()
    pt_opt.step()
    
    # JAX step
    new_jax_params, jax_loss = jax_train_step(
        jax_params, jnp.array(X_np), jnp.array(y_np), lr, jax_crit
    )
    
    # Compare
    pt_w = pt_model.linear.weight.detach().numpy().T
    pt_b = pt_model.linear.bias.detach().numpy()
    
    w_diff = np.abs(pt_w - new_jax_params['w']).max()
    b_diff = np.abs(pt_b - new_jax_params['b']).max()
    loss_diff = abs(pt_loss.item() - jax_loss)
    
    assert w_diff < tol and b_diff < tol and loss_diff < tol
    print(f"PASS: test_train_step (w_diff={w_diff:.2e}, loss_diff={loss_diff:.2e})")

## --- Tier 2: Convergence Equivalence ---

def test_convergence(tol=1e-2):
    # Setup
    pt_model = inject_pytorch_params(pt_make_model(), W_np, B_np)
    pt_opt = pt_make_optimizer(pt_model)
    pt_crit = pt_make_criterion()
    
    jax_params = make_jax_params(W_np, B_np)
    jax_crit = jax_make_criterion()
    lr = 0.01
    epochs = 100
    
    # Train both
    pt_train_model(torch.from_numpy(X_np), torch.from_numpy(y_np), 
                   pt_model, pt_opt, pt_crit, epochs)
    
    final_jax_params = jax_train_model(jnp.array(X_np), jnp.array(y_np), 
                                       jax_params, lr, jax_crit, epochs)
    
    # Final Predictions
    pt_preds = pt_model(torch.from_numpy(X_test_np)).detach().numpy()
    jax_preds = jax_model(final_jax_params, jnp.array(X_test_np))
    
    pred_diff = np.abs(pt_preds - jax_preds).max()
    assert pred_diff < tol, f"Convergence failed. Diff: {pred_diff}"
    print(f"PASS: test_convergence (pred_diff={pred_diff:.2e})")

## --- Runner ---

if __name__ == "__main__":
    print("Starting Equivalence Tests (Tolerances: Unit=1e-5, Conv=1e-2)...")
    try:
        test_forward_pass()
        test_loss()
        test_train_step()
        test_convergence()
        print("\nALL TESTS PASSED SUCCESSFULLY.")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")