"""
test_equivalence.py
Cross-framework equivalence tests between PyTorch and JAX implementations.

All tests use hardcoded numpy tensors — no RNG calls anywhere in this file.
"""

import numpy as np
import torch
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Imports from pytorch_refactored.py  (Rule 1)
# ---------------------------------------------------------------------------
from pytorch_refactored import (
    SimpleModel,
    make_model,
    make_criterion,
    make_optimizer,
    train_model as pt_train_model,
)

# ---------------------------------------------------------------------------
# Imports from jax_code_fixed.py  (Rule 1)
# ---------------------------------------------------------------------------
from jax_code_fixed import (
    forward as jax_model,
    mse_loss as loss_fn,
    train_step,
    train_model as jax_train_model,
)

# ---------------------------------------------------------------------------
# Hardcoded numpy tensors shared across all tests  (Rule 2)
# ---------------------------------------------------------------------------
X_np = np.array(
    [[0.1], [0.2], [0.3], [0.4], [0.5],
     [0.6], [0.7], [0.8], [0.9], [1.0]],
    dtype=np.float32,
)
y_np = (3.0 * X_np + 2.0).astype(np.float32)   # Noiseless targets

W_np = np.array([[0.5]], dtype=np.float32)       # weight (out, in) = (1, 1)
B_np = np.array([0.1], dtype=np.float32)         # bias   (out,)   = (1,)

X_test_np = np.array([[0.25], [0.75], [1.25]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helper: inject params into a PyTorch model  (Rule 3)
# ---------------------------------------------------------------------------
def inject_pytorch_params(model, w_np, b_np):
    """Set model.fc weight and bias from numpy arrays, bypassing random init."""
    with torch.no_grad():
        model.fc.weight.data = torch.tensor(w_np)
        model.fc.bias.data = torch.tensor(b_np)


# ---------------------------------------------------------------------------
# Helper: build JAX params dict from numpy arrays  (Rule 4)
# ---------------------------------------------------------------------------
def make_jax_params(w_np, b_np):
    """Construct the JAX params dict expected by forward / mse_loss / train_step."""
    return {
        "w": jnp.array(w_np),
        "b": jnp.array(b_np),
    }


# ===================================================================
# Tier 1 — Unit equivalence tests  (tolerance 1e-5)
# ===================================================================

UNIT_TOL = 1e-5


def test_forward_pass():
    """Identical X and params → same output from both frameworks.

    Rule 5, bullet 1.
    """
    # PyTorch
    pt_model = make_model()
    inject_pytorch_params(pt_model, W_np, B_np)
    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model(torch.tensor(X_np)).numpy()

    # JAX
    jax_params = make_jax_params(W_np, B_np)
    jax_out = np.asarray(jax_model(jax_params, jnp.array(X_np)))

    diff = np.max(np.abs(pt_out - jax_out))
    passed = diff < UNIT_TOL
    return passed, diff


def test_loss():
    """Identical inputs → same MSE loss from both frameworks.

    Rule 5, bullet 2.
    """
    # PyTorch
    pt_model = make_model()
    inject_pytorch_params(pt_model, W_np, B_np)
    pt_model.eval()
    criterion = make_criterion()
    with torch.no_grad():
        pt_preds = pt_model(torch.tensor(X_np))
        pt_loss = criterion(pt_preds, torch.tensor(y_np)).item()

    # JAX
    jax_params = make_jax_params(W_np, B_np)
    jax_loss_val = float(loss_fn(jax_params, jnp.array(X_np), jnp.array(y_np)))

    diff = abs(pt_loss - jax_loss_val)
    passed = diff < UNIT_TOL
    return passed, diff


def test_train_step():
    """One SGD step from identical state → same updated params and loss.

    Rule 5, bullet 3.
    """
    lr = 0.01

    # --- PyTorch: one manual step ---
    pt_model = make_model()
    inject_pytorch_params(pt_model, W_np, B_np)
    criterion = make_criterion()
    optimizer = make_optimizer(pt_model)
    # Override optimizer lr (make_optimizer already uses 0.01, but be explicit)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.zero_grad()
    pt_preds = pt_model(torch.tensor(X_np))
    pt_loss = criterion(pt_preds, torch.tensor(y_np))
    pt_loss.backward()
    optimizer.step()

    pt_w_after = pt_model.fc.weight.data.numpy()
    pt_b_after = pt_model.fc.bias.data.numpy()
    pt_loss_val = pt_loss.item()

    # --- JAX: one step ---
    jax_params = make_jax_params(W_np, B_np)
    jax_params_after, jax_loss_val = train_step(
        jax_params, jnp.array(X_np), jnp.array(y_np), lr
    )
    jax_w_after = np.asarray(jax_params_after["w"])
    jax_b_after = np.asarray(jax_params_after["b"])
    jax_loss_val = float(jax_loss_val)

    w_diff = np.max(np.abs(pt_w_after - jax_w_after))
    b_diff = np.max(np.abs(pt_b_after - jax_b_after))
    loss_diff = abs(pt_loss_val - jax_loss_val)
    max_diff = max(w_diff, b_diff, loss_diff)

    passed = max_diff < UNIT_TOL
    return passed, max_diff


# ===================================================================
# Tier 2 — Convergence equivalence test  (tolerance 1e-2)
# ===================================================================

CONV_TOL = 1e-2


def test_convergence():
    """Full training from identical starting params and data →
    learned params and predictions agree.

    Rule 6: calls pt_train_model() and jax_train_model() directly.
    """
    num_epochs = 100
    lr = 0.01

    # --- PyTorch ---
    pt_model = make_model()
    inject_pytorch_params(pt_model, W_np, B_np)
    criterion = make_criterion()
    optimizer = make_optimizer(pt_model)

    pt_train_model(
        torch.tensor(X_np),
        torch.tensor(y_np),
        pt_model,
        optimizer,
        criterion,
        num_epochs,
    )

    pt_model.eval()
    with torch.no_grad():
        pt_preds = pt_model(torch.tensor(X_test_np)).numpy()
    pt_w = pt_model.fc.weight.data.numpy()
    pt_b = pt_model.fc.bias.data.numpy()

    # --- JAX ---
    jax_params = make_jax_params(W_np, B_np)
    jax_params = jax_train_model(
        jnp.array(X_np),
        jnp.array(y_np),
        jax_params,
        num_epochs,
        lr,
    )

    jax_preds = np.asarray(jax_model(jax_params, jnp.array(X_test_np)))
    jax_w = np.asarray(jax_params["w"])
    jax_b = np.asarray(jax_params["b"])

    w_diff = np.max(np.abs(pt_w - jax_w))
    b_diff = np.max(np.abs(pt_b - jax_b))
    pred_diff = np.max(np.abs(pt_preds - jax_preds))
    max_diff = max(w_diff, b_diff, pred_diff)

    passed = max_diff < CONV_TOL
    return passed, max_diff


# ===================================================================
# __main__ runner  (Rule 8)
# ===================================================================

if __name__ == "__main__":
    tests = [
        ("test_forward_pass",  test_forward_pass,  UNIT_TOL),
        ("test_loss",          test_loss,           UNIT_TOL),
        ("test_train_step",    test_train_step,     UNIT_TOL),
        ("test_convergence",   test_convergence,    CONV_TOL),
    ]

    all_passed = True
    for name, fn, tol in tests:
        passed, diff = fn()
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name:25s}  max_diff={diff:.2e}  tol={tol:.0e}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed.")
    else:
        print("Some tests FAILED.")