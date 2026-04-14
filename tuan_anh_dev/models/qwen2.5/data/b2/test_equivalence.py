import numpy as np
import pandas as pd
import torch
import jax.numpy as jnp

from pytorch_refactored import (
    LinearRegressionModel,
    make_model,
    make_criterion,
    make_optimizer,
    train_model as pt_train_model,
)
from jax_code_fixed import (
    model as jax_model,
    loss_fn,
    train_step,
    train_model as jax_train_model,
)

# Hardcoded tensors — defined once, shared across all tests
X_np = np.array([[1.0], [2.0], [3.0], [4.0], [5.0],
                 [6.0], [7.0], [8.0], [9.0], [10.0]], dtype=np.float32)
y_np = np.array([[5.1], [6.9], [9.2], [11.0], [12.8],
                 [15.1], [17.0], [18.9], [21.2], [22.8]], dtype=np.float32)
W_np = np.array([[0.5]], dtype=np.float32)   # shape (1, 1)
B_np = np.array([0.1], dtype=np.float32)     # shape (1,)
X_test_np = np.array([[4.0], [7.0]], dtype=np.float32)


def _write_data_csv():
    """Write hardcoded data to data.csv so PyTorch's DataLoader can read it."""
    df = pd.DataFrame({'X': X_np.flatten(), 'y': y_np.flatten()})
    df.to_csv('data.csv', index=False)


def inject_pytorch_params(model, w_np, b_np):
    """Inject hardcoded params into PyTorch model via .weight.data and .bias.data."""
    with torch.no_grad():
        model.linear.weight.data = torch.tensor(w_np.T)
        model.linear.bias.data = torch.tensor(b_np)


def make_jax_params(w_np, b_np):
    """Construct JAX params dict from the same numpy arrays."""
    return {'w': jnp.array(w_np), 'b': jnp.array(b_np)}


# ---------- Tier 1 — tolerance 1e-5 ----------

def test_forward_pass():
    model = make_model()
    inject_pytorch_params(model, W_np, B_np)
    with torch.no_grad():
        pt_out = model(torch.tensor(X_test_np)).numpy()

    jax_params = make_jax_params(W_np, B_np)
    jax_out = np.array(jax_model(jax_params, jnp.array(X_test_np)))

    diff = float(np.max(np.abs(pt_out - jax_out)))
    passed = diff < 1e-5
    print(f"test_forward_pass: {'PASS' if passed else 'FAIL'} "
          f"(max diff: {diff:.2e}, tol: 1e-5)")
    return passed


def test_loss():
    model = make_model()
    inject_pytorch_params(model, W_np, B_np)
    criterion = make_criterion()
    with torch.no_grad():
        pt_pred = model(torch.tensor(X_np))
        pt_loss = criterion(pt_pred, torch.tensor(y_np)).item()

    jax_params = make_jax_params(W_np, B_np)
    jax_loss = float(loss_fn(jax_params, jnp.array(X_np), jnp.array(y_np)))

    diff = abs(pt_loss - jax_loss)
    passed = diff < 1e-5
    print(f"test_loss: {'PASS' if passed else 'FAIL'} "
          f"(PT: {pt_loss:.6f}, JAX: {jax_loss:.6f}, diff: {diff:.2e}, tol: 1e-5)")
    return passed


def test_train_step():
    lr = 0.01

    # PyTorch — one manual step (bypassing DataLoader)
    model = make_model()
    inject_pytorch_params(model, W_np, B_np)
    optimizer = make_optimizer(model)
    criterion = make_criterion()
    pred = model(torch.tensor(X_np))
    loss = criterion(pred, torch.tensor(y_np))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pt_loss = loss.item()
    pt_w = model.linear.weight.data.numpy()
    pt_b = model.linear.bias.data.numpy()

    # JAX
    jax_params = make_jax_params(W_np, B_np)
    new_params, jax_loss_val = train_step(jax_params, jnp.array(X_np),
                                          jnp.array(y_np), lr)
    jax_loss = float(jax_loss_val)
    jax_w = np.array(new_params['w'])
    jax_b = np.array(new_params['b'])

    loss_diff = abs(pt_loss - jax_loss)
    w_diff = float(np.max(np.abs(pt_w - jax_w.T)))
    b_diff = float(np.max(np.abs(pt_b - jax_b)))

    passed = loss_diff < 1e-5 and w_diff < 1e-5 and b_diff < 1e-5
    print(f"test_train_step: {'PASS' if passed else 'FAIL'} "
          f"(loss diff: {loss_diff:.2e}, w diff: {w_diff:.2e}, "
          f"b diff: {b_diff:.2e}, tol: 1e-5)")
    return passed


# ---------- Tier 2 — tolerance 1e-2 ----------

def test_convergence():
    """Full training from identical starting params.

    NOTE: PyTorch uses mini-batch SGD (DataLoader batch_size=32, shuffle=True)
    while JAX uses full-batch SGD. Results may diverge due to this structural
    difference.
    """
    num_epochs = 200
    lr = 0.01

    # Write hardcoded data to CSV for PyTorch's DataLoader
    _write_data_csv()

    # PyTorch
    model = make_model()
    inject_pytorch_params(model, W_np, B_np)
    optimizer = make_optimizer(model)
    criterion = make_criterion()
    pt_train_model(torch.tensor(X_np), torch.tensor(y_np),
                   model, optimizer, criterion, num_epochs)
    pt_w = model.linear.weight.data.numpy()
    pt_b = model.linear.bias.data.numpy()

    # JAX
    jax_params = make_jax_params(W_np, B_np)
    jax_final = jax_train_model(jnp.array(X_np), jnp.array(y_np),
                                jax_params, num_epochs, lr)
    jax_w = np.array(jax_final['w'])
    jax_b = np.array(jax_final['b'])

    w_diff = float(np.max(np.abs(pt_w - jax_w.T)))
    b_diff = float(np.max(np.abs(pt_b - jax_b)))

    passed = w_diff < 1e-2 and b_diff < 1e-2
    print(f"test_convergence: {'PASS' if passed else 'FAIL'} "
          f"(w diff: {w_diff:.2e}, b diff: {b_diff:.2e}, tol: 1e-2)")
    return passed


if __name__ == '__main__':
    results = []
    results.append(("test_forward_pass", test_forward_pass()))
    results.append(("test_loss", test_loss()))
    results.append(("test_train_step", test_train_step()))
    results.append(("test_convergence", test_convergence()))

    print("\n--- Summary ---")
    all_passed = True
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
