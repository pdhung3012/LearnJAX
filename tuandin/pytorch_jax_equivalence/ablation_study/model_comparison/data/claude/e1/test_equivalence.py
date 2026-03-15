"""
test_equivalence.py

Cross-framework equivalence tests between pytorch_refactored.py and
jax_code_fixed.py. All tests use hardcoded tensors — no RNG anywhere.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import torch
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Imports from pytorch_refactored.py (actual function names)
# ---------------------------------------------------------------------------
from pytorch_refactored import (
    LinearRegressionModel,
    make_model,
    make_criterion,
    make_optimizer,
    train_model as pt_train_model,
)

# ---------------------------------------------------------------------------
# Imports from jax_code_fixed.py (actual function names)
# ---------------------------------------------------------------------------
from jax_code_fixed import (
    linear_model as jax_model,
    mse_loss as loss_fn,
    train_step,
    train_model as jax_train_model,
)

# ---------------------------------------------------------------------------
# Hardcoded numpy tensors shared across ALL tests — no RNG anywhere.
# ---------------------------------------------------------------------------
X_np = np.array(
    [[1.0], [2.0], [3.0], [4.0], [5.0],
     [6.0], [7.0], [8.0], [9.0], [10.0]],
    dtype=np.float32,
)
y_np = np.array(
    [[5.1], [6.9], [9.2], [11.0], [12.8],
     [15.1], [16.9], [19.2], [21.0], [22.8]],
    dtype=np.float32,
)
W_np = np.array([[0.5]], dtype=np.float32)
B_np = np.array([0.1], dtype=np.float32)
X_test_np = np.array([[4.0], [7.0]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helper: inject params into PyTorch model (bypasses random init)
# ---------------------------------------------------------------------------
def inject_pytorch_params(model, w_np, b_np):
    """Set model.linear.weight and model.linear.bias from numpy arrays.

    nn.Linear(1,1) stores weight as shape (out_features, in_features) = (1,1)
    and bias as shape (out_features,) = (1,).
    """
    model.linear.weight.data = torch.from_numpy(np.array(w_np))
    model.linear.bias.data = torch.from_numpy(np.array(b_np))


# ---------------------------------------------------------------------------
# Helper: build JAX params dict from the same numpy arrays
# ---------------------------------------------------------------------------
def make_jax_params(w_np, b_np):
    """Construct the JAX params dict matching init_params() structure.

    linear_model() computes jnp.dot(x, params['w']) + params['b'].
    nn.Linear computes x @ weight.T + bias.
    For weight shape (1,1), weight.T == weight, so the same (1,1) array
    is used directly as params['w'] — no transpose needed for this shape.
    """
    return {
        'w': jnp.array(w_np),
        'b': jnp.array(b_np),
    }


# ===================================================================
# Tier 1 — Unit equivalence tests  (tolerance 1e-5)
# ===================================================================
UNIT_TOL = 1e-5


def test_forward_pass():
    """Identical X and params → same output from both frameworks."""
    # PyTorch forward
    pt_model = make_model()
    inject_pytorch_params(pt_model, W_np, B_np)
    with torch.no_grad():
        pt_out = pt_model(torch.tensor(X_test_np)).numpy()

    # JAX forward
    jax_params = make_jax_params(W_np, B_np)
    jax_out = np.array(jax_model(jax_params, jnp.array(X_test_np)))

    diff = np.max(np.abs(pt_out - jax_out))
    passed = diff < UNIT_TOL
    return passed, diff


def test_loss():
    """Identical inputs → same MSE loss from both frameworks."""
    # Use float64 to eliminate cross-framework float32 reduction rounding.
    X_64 = X_np.astype(np.float64)
    y_64 = y_np.astype(np.float64)
    W_64 = W_np.astype(np.float64)
    B_64 = B_np.astype(np.float64)

    # PyTorch loss
    pt_model = make_model()
    pt_model.double()
    inject_pytorch_params(pt_model, W_64, B_64)
    criterion = make_criterion()
    with torch.no_grad():
        pt_pred = pt_model(torch.tensor(X_64))
        pt_loss = criterion(pt_pred, torch.tensor(y_64)).item()

    # JAX loss
    jax_params = make_jax_params(W_64, B_64)
    jax_loss_val = float(loss_fn(jax_params, jnp.array(X_64), jnp.array(y_64)))

    diff = abs(pt_loss - jax_loss_val)
    passed = diff < UNIT_TOL
    return passed, diff


def test_train_step():
    """One SGD step from identical state → same updated params and loss."""
    lr = 0.01
    # Use float64 to eliminate cross-framework float32 reduction rounding.
    X_64 = X_np.astype(np.float64)
    y_64 = y_np.astype(np.float64)
    W_64 = W_np.astype(np.float64)
    B_64 = B_np.astype(np.float64)

    # --- PyTorch: one manual step ---
    pt_model = make_model()
    pt_model.double()
    inject_pytorch_params(pt_model, W_64, B_64)
    criterion = make_criterion()
    optimizer = make_optimizer(pt_model)

    pt_pred = pt_model(torch.tensor(X_64))
    pt_loss = criterion(pt_pred, torch.tensor(y_64))
    optimizer.zero_grad()
    pt_loss.backward()
    optimizer.step()

    pt_loss_val = pt_loss.item()
    pt_w = pt_model.linear.weight.data.numpy().copy()
    pt_b = pt_model.linear.bias.data.numpy().copy()

    # --- JAX: one step via train_step ---
    jax_params = make_jax_params(W_64, B_64)
    jax_updated, jax_loss_val = train_step(
        jax_params, jnp.array(X_64), jnp.array(y_64), lr
    )
    jax_w = np.array(jax_updated['w'])
    jax_b = np.array(jax_updated['b'])
    jax_loss_scalar = float(jax_loss_val)

    loss_diff = abs(pt_loss_val - jax_loss_scalar)
    w_diff = np.max(np.abs(pt_w - jax_w))
    b_diff = np.max(np.abs(pt_b - jax_b))
    max_diff = max(loss_diff, w_diff, b_diff)
    passed = max_diff < UNIT_TOL
    return passed, max_diff


# ===================================================================
# Tier 2 — Convergence equivalence test  (tolerance 1e-2)
# ===================================================================
CONV_TOL = 1e-2
CONV_EPOCHS = 1000


def test_convergence():
    """Full training from identical start → learned params and predictions agree.

    Calls pt_train_model() and jax_train_model() directly.
    """
    lr = 0.01

    # --- PyTorch full training ---
    pt_model = make_model()
    inject_pytorch_params(pt_model, W_np, B_np)
    criterion = make_criterion()
    optimizer = make_optimizer(pt_model)

    pt_train_model(
        torch.tensor(X_np), torch.tensor(y_np),
        pt_model, optimizer, criterion, num_epochs=CONV_EPOCHS,
    )

    with torch.no_grad():
        pt_w = pt_model.linear.weight.data.numpy().copy()
        pt_b = pt_model.linear.bias.data.numpy().copy()
        pt_preds = pt_model(torch.tensor(X_test_np)).numpy()

    # --- JAX full training ---
    jax_params = make_jax_params(W_np, B_np)
    jax_params = jax_train_model(
        jnp.array(X_np), jnp.array(y_np),
        jax_params, lr, num_epochs=CONV_EPOCHS,
    )

    jax_w = np.array(jax_params['w'])
    jax_b = np.array(jax_params['b'])
    jax_preds = np.array(jax_model(jax_params, jnp.array(X_test_np)))

    w_diff = np.max(np.abs(pt_w - jax_w))
    b_diff = np.max(np.abs(pt_b - jax_b))
    pred_diff = np.max(np.abs(pt_preds - jax_preds))
    max_diff = max(w_diff, b_diff, pred_diff)
    passed = max_diff < CONV_TOL
    return passed, max_diff


# ===================================================================
# __main__ runner
# ===================================================================
if __name__ == "__main__":
    tests = [
        ("test_forward_pass",  test_forward_pass,  UNIT_TOL),
        ("test_loss",          test_loss,           UNIT_TOL),
        ("test_train_step",    test_train_step,     UNIT_TOL),
        ("test_convergence",   test_convergence,    CONV_TOL),
    ]

    all_passed = True
    print("=" * 64)
    print("Cross-framework equivalence tests: PyTorch ↔ JAX")
    print("=" * 64)

    for name, fn, tol in tests:
        passed, diff = fn()
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name:<25s}  max_diff={diff:.2e}  tol={tol:.0e}")
        if not passed:
            all_passed = False

    print("=" * 64)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 64)