"""
test_equivalence.py
===================
Equivalence tests between the PyTorch and JAX linear-regression implementations.

Expected directory layout
--------------------------
    <problem_dir>/          # e.g. e1/  or  m1/
        pytorch_code.py     # original PyTorch implementation
        jax_code_fixed.py   # reviewed & fixed JAX translation
        test_equivalence.py # this file

Both sibling modules are imported directly — no code is copied here.
If either sibling changes, the tests automatically exercise the updated version.

Structure
---------
IMPORTS
    pytorch_code    → LinearRegressionModel
    jax_code_fixed  → forward, mse_loss, train_step, train_model

SHARED FIXTURES
    Hardcoded numpy arrays used by every test.
    make_pytorch_model()  – injects fixed weights into the imported PyTorch model.
    make_jax_params()     – constructs the JAX params dict from the same arrays.

TIER 1 — Unit equivalence  (tolerance 1e-5)
    test_forward_pass()   – identical output given identical X and params.
    test_loss()           – identical MSE loss given identical inputs.
    test_train_step()     – identical updated params and loss after one SGD step.

TIER 2 — Convergence equivalence  (tolerance 1e-2)
    test_convergence()    – after full training from identical starts, learned
                            params and test predictions agree.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import jax.numpy as jnp

# =============================================================================
# SIBLING IMPORTS
#
# We import by module name so that:
#   • Any change to pytorch_code.py or jax_code_fixed.py is picked up
#     automatically — there is no stale copy here to drift out of sync.
#   • A missing or broken sibling produces an ImportError with a clear message
#     rather than a silent wrong-answer from a stale local copy.
#
# Run from the problem directory so Python finds the siblings:
#   cd e1 && python test_equivalence.py
#   pytest e1/test_equivalence.py     # pytest adds the test dir to sys.path
# =============================================================================

try:
    from pytorch_code import LinearRegressionModel
except ImportError as exc:
    raise ImportError(
        "Could not import LinearRegressionModel from pytorch_code.py. "
        "Make sure pytorch_code.py is in the same directory as this file "
        "and run from that directory (e.g. `cd e1 && python test_equivalence.py`)."
    ) from exc

try:
    from jax_code_fixed import forward, mse_loss, train_step, train_model
except ImportError as exc:
    raise ImportError(
        "Could not import forward/mse_loss/train_step/train_model from "
        "jax_code_fixed.py. Make sure jax_code_fixed.py is in the same "
        "directory as this file."
    ) from exc


# =============================================================================
# SHARED FIXTURES
# Hardcoded numpy arrays shared across ALL tests.
# Using small, human-verifiable values (noise-free y = 2x + 3) so failures
# are easy to diagnose without a debugger.
# =============================================================================

# --- Training data: 5 points on y = 2x + 3, no noise ---
X_np  = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]],   dtype=np.float32)
y_np  = np.array([[5.0], [7.0], [9.0], [11.0], [13.0]], dtype=np.float32)

# --- Initial model parameters (deliberately off from ground truth w=2, b=3) ---
W_np  = np.array([[0.5]], dtype=np.float32)   # shape (1, 1)
B_np  = np.array([0.0],  dtype=np.float32)   # shape (1,)

# --- Test inputs ---
X_test_np = np.array([[4.0], [7.0]], dtype=np.float32)

# Tolerance constants
TOL_UNIT        = 1e-5   # Tier 1: single-op equivalence across backends
TOL_CONVERGENCE = 1e-2   # Tier 2: accumulated float32 drift over 2 000 steps


# =============================================================================
# FIXTURES — model factories
# Both factories accept the same numpy arrays so both frameworks start from
# an identical numerical state.
# =============================================================================

def make_pytorch_model(w_np: np.ndarray, b_np: np.ndarray) -> nn.Module:
    """
    Instantiate the imported LinearRegressionModel and inject hardcoded
    weights, bypassing its random __init__.

    Parameters
    ----------
    w_np : (1, 1) float32 ndarray  – injected into model.linear.weight
    b_np : (1,)  float32 ndarray  – injected into model.linear.bias

    Returns
    -------
    nn.Module with .linear.weight == w_np and .linear.bias == b_np
    """
    model = LinearRegressionModel()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor(w_np))   # (1, 1)
        model.linear.bias.copy_(torch.tensor(b_np))     # (1,)
    return model


def make_jax_params(w_np: np.ndarray, b_np: np.ndarray) -> dict:
    """
    Construct the JAX params dict expected by forward() / mse_loss() /
    train_step() / train_model() in jax_code_fixed.py.

    Parameters
    ----------
    w_np : (1, 1) float32 ndarray
    b_np : (1,)  float32 ndarray

    Returns
    -------
    {'w': jnp.array (1,1), 'b': jnp.array (1,)}
    """
    return {
        'w': jnp.array(w_np),
        'b': jnp.array(b_np),
    }


# =============================================================================
# TIER 1 — Unit equivalence tests  (tolerance 1e-5)
# =============================================================================

def test_forward_pass() -> bool:
    """
    Verify that the forward pass produces the same output in both frameworks
    given identical inputs and parameters.

    PyTorch path : model(X_torch)         via nn.Linear  (x @ w.T + b)
    JAX path     : forward(params, X_jax) via jnp.dot(x, w) + b

    Shape note: PyTorch stores weight as (out, in)=(1,1) and transposes
    internally; jax_code_fixed stores 'w' as (in, out)=(1,1).  For (1,1)
    the transpose is a no-op, so outputs are numerically identical.

    Tolerance : 1e-5
    """
    X_torch = torch.tensor(X_np)
    model   = make_pytorch_model(W_np, B_np)
    params  = make_jax_params(W_np, B_np)
    X_jax   = jnp.array(X_np)

    with torch.no_grad():
        pt_out = model(X_torch).numpy()

    jax_out = np.array(forward(params, X_jax))

    match = np.allclose(pt_out, jax_out, atol=TOL_UNIT)
    _report("test_forward_pass", match,
            f"max_diff={np.max(np.abs(pt_out - jax_out)):.2e}")
    return match


def test_loss() -> bool:
    """
    Verify that MSE loss is identical in both frameworks.

    PyTorch path : nn.MSELoss()(predictions, y)  reduction='mean'
    JAX path     : mse_loss(params, x, y)         jnp.mean((pred-y)**2)

    Tolerance : 1e-5
    """
    X_torch = torch.tensor(X_np)
    y_torch = torch.tensor(y_np)
    model   = make_pytorch_model(W_np, B_np)
    params  = make_jax_params(W_np, B_np)
    X_jax   = jnp.array(X_np)
    y_jax   = jnp.array(y_np)

    with torch.no_grad():
        pt_loss = nn.MSELoss()(model(X_torch), y_torch).item()

    jax_loss = float(mse_loss(params, X_jax, y_jax))

    match = abs(pt_loss - jax_loss) < TOL_UNIT
    _report("test_loss", match,
            f"pytorch={pt_loss:.6f}  jax={jax_loss:.6f}  "
            f"diff={abs(pt_loss - jax_loss):.2e}")
    return match


def test_train_step() -> bool:
    """
    Verify that a single SGD step produces identical updated parameters
    and loss in both frameworks.

    PyTorch path : zero_grad → loss.backward() → optimizer.step()
    JAX path     : train_step(params, x, y, lr)  from jax_code_fixed.py
                   (value_and_grad → manual SGD dict comprehension)

    Gradient shape note:
      PyTorch grad for linear.weight : (1, 1) — matches weight shape (out, in)
      JAX grad for params['w']       : (1, 1) — matches 'w' shape (in, out)
      Both are scalars for this 1-D model, so they agree numerically.

    Tolerance : 1e-5
    """
    LR = 0.01

    # --- PyTorch ---
    X_torch   = torch.tensor(X_np)
    y_torch   = torch.tensor(y_np)
    model     = make_pytorch_model(W_np, B_np)
    optimizer = optim.SGD(model.parameters(), lr=LR)

    loss_pt = nn.MSELoss()(model(X_torch), y_torch)
    optimizer.zero_grad()
    loss_pt.backward()
    optimizer.step()

    pt_w  = model.linear.weight.detach().numpy()   # (1, 1)
    pt_b  = model.linear.bias.detach().numpy()     # (1,)
    pt_lv = loss_pt.item()

    # --- JAX (via imported train_step) ---
    params = make_jax_params(W_np, B_np)
    X_jax  = jnp.array(X_np)
    y_jax  = jnp.array(y_np)

    updated, jax_loss = train_step(params, X_jax, y_jax, LR)

    jax_w  = np.array(updated['w'])   # (1, 1)
    jax_b  = np.array(updated['b'])   # (1,)
    jax_lv = float(jax_loss)

    w_ok = np.allclose(pt_w, jax_w, atol=TOL_UNIT)
    b_ok = np.allclose(pt_b, jax_b, atol=TOL_UNIT)
    l_ok = abs(pt_lv - jax_lv) < TOL_UNIT

    match = w_ok and b_ok and l_ok
    _report(
        "test_train_step", match,
        f"w_diff={np.max(np.abs(pt_w - jax_w)):.2e}  "
        f"b_diff={np.max(np.abs(pt_b - jax_b)):.2e}  "
        f"loss_diff={abs(pt_lv - jax_lv):.2e}"
    )
    return match


# =============================================================================
# TIER 2 — Convergence equivalence test  (tolerance 1e-2)
# =============================================================================

def test_convergence() -> bool:
    """
    Train both implementations for 2 000 epochs from identical starting params
    on identical data, then assert that learned params and test predictions
    agree within 1e-2.

    Why 2 000 epochs?
    At lr=0.01 on this 5-point noise-free dataset, 1 000 epochs leaves
    |w-2| ≈ 0.022, clipping the 1e-2 tolerance.  2 000 epochs yields
    |w-2| ≈ 7.6e-4 with ~13× margin inside the band.

    Why a looser tolerance than Tier 1?
    Float32 arithmetic is non-associative; JAX's XLA-compiled loop and
    PyTorch's eager loop accumulate rounding errors differently over 2 000
    steps.  1e-2 confirms convergence to the same solution while absorbing
    that backend-level numerical divergence.

    JAX path uses train_model() + forward() imported from jax_code_fixed.py.
    """
    LR     = 0.01
    EPOCHS = 2000

    # --- PyTorch ---
    X_torch      = torch.tensor(X_np)
    y_torch      = torch.tensor(y_np)
    X_test_torch = torch.tensor(X_test_np)
    model        = make_pytorch_model(W_np, B_np)
    optimizer    = optim.SGD(model.parameters(), lr=LR)

    for _ in range(EPOCHS):
        loss = nn.MSELoss()(model(X_torch), y_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pt_w = model.linear.weight.detach().numpy()       # (1, 1)
    pt_b = model.linear.bias.detach().numpy()         # (1,)
    with torch.no_grad():
        pt_preds = model(X_test_torch).numpy()        # (2, 1)

    # --- JAX (via imported train_model + forward) ---
    params     = make_jax_params(W_np, B_np)
    X_jax      = jnp.array(X_np)
    y_jax      = jnp.array(y_np)
    X_test_jax = jnp.array(X_test_np)

    params    = train_model(X_jax, y_jax, params, epochs=EPOCHS, lr=LR)
    jax_w     = np.array(params['w'])                           # (1, 1)
    jax_b     = np.array(params['b'])                           # (1,)
    jax_preds = np.array(forward(params, X_test_jax))           # (2, 1)

    w_ok    = np.allclose(pt_w,     jax_w,     atol=TOL_CONVERGENCE)
    b_ok    = np.allclose(pt_b,     jax_b,     atol=TOL_CONVERGENCE)
    pred_ok = np.allclose(pt_preds, jax_preds, atol=TOL_CONVERGENCE)

    match = w_ok and b_ok and pred_ok
    _report(
        "test_convergence", match,
        f"w:     pt={pt_w[0,0]:.5f}  jax={jax_w[0,0]:.5f}  "
        f"diff={abs(pt_w[0,0]-jax_w[0,0]):.2e}\n"
        f"        b:     pt={pt_b[0]:.5f}   jax={jax_b[0]:.5f}   "
        f"diff={abs(pt_b[0]-jax_b[0]):.2e}\n"
        f"        preds: max_diff={np.max(np.abs(pt_preds - jax_preds)):.2e}"
    )
    return match


# =============================================================================
# HELPERS
# =============================================================================

def _report(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS ✅" if passed else "FAIL ❌"
    print(f"  [{status}]  {name}")
    if detail:
        print(f"        {detail}")


# =============================================================================
# RUNNER
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PyTorch ↔ JAX Equivalence Test Suite")
    print("  pytorch_code.py  ↔  jax_code_fixed.py")
    print("=" * 60)

    print("\nTIER 1 — Unit equivalence  (tolerance 1e-5)")
    print("-" * 60)
    r1 = test_forward_pass()
    r2 = test_loss()
    r3 = test_train_step()

    print("\nTIER 2 — Convergence equivalence  (tolerance 1e-2)")
    print("-" * 60)
    r4 = test_convergence()

    passed = sum([r1, r2, r3, r4])
    total  = 4
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} tests passed")
    print("=" * 60 + "\n")

    sys.exit(0 if passed == total else 1)
