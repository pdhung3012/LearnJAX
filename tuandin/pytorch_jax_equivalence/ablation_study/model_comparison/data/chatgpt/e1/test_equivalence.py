# test_equivalence.py

import numpy as np

# Hardcoded numpy tensors shared across all tests (no RNG anywhere in this file)
X_np = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
y_np = 2.0 * X_np + 3.0  # no noise, deterministic
W_np = np.array([[0.5]], dtype=np.float32)
B_np = np.array([1.0], dtype=np.float32)
X_test_np = np.array([[4.0], [7.0]], dtype=np.float32)

# ---- Imports from pytorch_refactored.py (do not copy code) ----
from pytorch_refactored import (
    LinearRegressionModel,
    make_model as pt_make_model,
    make_criterion as pt_make_criterion,
    make_optimizer as pt_make_optimizer,
    train_model as pt_train_model,
)

# ---- Imports from jax_code_fixed.py (do not copy code) ----
from jax_code_fixed import (
    model_apply as jax_model,
    loss_fn,
    train_step,
    train_model as jax_train_model,
    make_criterion as jax_make_criterion,
    make_optimizer as jax_make_optimizer,
)

import torch
import jax.numpy as jnp


def inject_pytorch_params(model: LinearRegressionModel, w_np: np.ndarray, b_np: np.ndarray) -> None:
    """
    Inject hardcoded params into the PyTorch model, bypassing random initialization.
    """
    with torch.no_grad():
        # IMPORTANT: clone() to avoid sharing memory with NumPy arrays.
        # Otherwise, optimizer.step() will mutate W_np/B_np in-place and break equivalence tests.
        model.linear.weight.data = torch.from_numpy(w_np).clone().to(model.linear.weight.data.dtype)
        model.linear.bias.data = torch.from_numpy(b_np).clone().to(model.linear.bias.data.dtype)


def make_jax_params(w_np: np.ndarray, b_np: np.ndarray):
    """
    Construct JAX params dict from the same numpy arrays.
    """
    return {"w": jnp.array(w_np), "b": jnp.array(b_np)}


def _assert_allclose(name: str, a, b, tol: float) -> None:
    a_np = np.array(a)
    b_np = np.array(b)
    if not np.allclose(a_np, b_np, atol=tol, rtol=0.0):
        diff = np.max(np.abs(a_np - b_np))
        raise AssertionError(f"{name} failed: max_abs_diff={diff} > tol={tol}\nA={a_np}\nB={b_np}")


# ---------------- Tier 1: Unit equivalence (1e-5) ----------------

def test_forward_pass(tol: float = 1e-5) -> None:
    # PyTorch forward
    pt_model = pt_make_model()
    inject_pytorch_params(pt_model, W_np, B_np)
    X_t = torch.from_numpy(X_np)
    with torch.no_grad():
        pt_out = pt_model(X_t).cpu().numpy()

    # JAX forward
    jax_params = make_jax_params(W_np, B_np)
    jax_out = np.array(jax_model(jax_params, jnp.array(X_np)))

    _assert_allclose("forward_pass", pt_out, jax_out, tol)


def test_loss(tol: float = 1e-5) -> None:
    # PyTorch loss
    pt_model = pt_make_model()
    inject_pytorch_params(pt_model, W_np, B_np)
    pt_criterion = pt_make_criterion()
    X_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(y_np)
    with torch.no_grad():
        pt_pred = pt_model(X_t)
        pt_loss = pt_criterion(pt_pred, y_t).item()

    # JAX loss
    jax_params = make_jax_params(W_np, B_np)
    jax_criterion = jax_make_criterion()
    jax_loss = float(loss_fn(jax_params, jnp.array(X_np), jnp.array(y_np), jax_criterion))

    _assert_allclose("loss", pt_loss, jax_loss, tol)


def test_train_step(tol: float = 1e-5) -> None:
    # PyTorch one step (SGD lr=0.01)
    pt_model = pt_make_model()
    inject_pytorch_params(pt_model, W_np, B_np)
    pt_criterion = pt_make_criterion()
    pt_optimizer = pt_make_optimizer(pt_model)

    X_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(y_np)

    pt_optimizer.zero_grad()
    pt_pred = pt_model(X_t)
    pt_loss_tensor = pt_criterion(pt_pred, y_t)
    pt_loss_tensor.backward()
    pt_optimizer.step()

    pt_w = pt_model.linear.weight.detach().cpu().numpy()
    pt_b = pt_model.linear.bias.detach().cpu().numpy()
    pt_loss = float(pt_loss_tensor.detach().cpu().item())

    # JAX one step (manual SGD lr=0.01)
    jax_params = make_jax_params(W_np, B_np)
    jax_criterion = jax_make_criterion()
    jax_optimizer = jax_make_optimizer(lr=0.01)

    new_jax_params, jax_loss = train_step(
        jax_params,
        jnp.array(X_np),
        jnp.array(y_np),
        jax_optimizer,
        jax_criterion,
    )

    jax_w = np.array(new_jax_params["w"])
    jax_b = np.array(new_jax_params["b"])
    jax_loss = float(jax_loss)

    _assert_allclose("train_step_loss", pt_loss, jax_loss, tol)
    _assert_allclose("train_step_w", pt_w, jax_w, tol)
    _assert_allclose("train_step_b", pt_b, jax_b, tol)


# ---------------- Tier 2: Convergence equivalence (1e-2) ----------------

def test_convergence(tol: float = 1e-2) -> None:
    num_epochs = 500

    # PyTorch training
    pt_model = pt_make_model()
    inject_pytorch_params(pt_model, W_np, B_np)
    pt_criterion = pt_make_criterion()
    pt_optimizer = pt_make_optimizer(pt_model)

    X_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(y_np)

    pt_train_model(X_t, y_t, pt_model, pt_optimizer, pt_criterion, num_epochs)

    pt_w = float(pt_model.linear.weight.detach().cpu().numpy().reshape(-1)[0])
    pt_b = float(pt_model.linear.bias.detach().cpu().numpy().reshape(-1)[0])

    X_test_t = torch.from_numpy(X_test_np)
    with torch.no_grad():
        pt_pred_test = pt_model(X_test_t).cpu().numpy()

    # JAX training
    jax_params = make_jax_params(W_np, B_np)
    jax_criterion = jax_make_criterion()
    jax_optimizer = jax_make_optimizer(lr=0.01)

    jax_params = jax_train_model(
        jnp.array(X_np),
        jnp.array(y_np),
        jax_params,
        jax_optimizer,
        jax_criterion,
        num_epochs,
    )

    jax_w = float(np.array(jax_params["w"]).reshape(-1)[0])
    jax_b = float(np.array(jax_params["b"]).reshape(-1)[0])

    jax_pred_test = np.array(jax_model(jax_params, jnp.array(X_test_np)))

    _assert_allclose("convergence_w", pt_w, jax_w, tol)
    _assert_allclose("convergence_b", pt_b, jax_b, tol)
    _assert_allclose("convergence_predictions", pt_pred_test, jax_pred_test, tol)


def _run_test(fn, tol: float) -> bool:
    name = fn.__name__
    try:
        fn(tol=tol)
        print(f"PASS: {name} (tol={tol})")
        return True
    except Exception as e:
        print(f"FAIL: {name} (tol={tol}) -> {e}")
        return False


if __name__ == "__main__":
    ok = True
    ok &= _run_test(test_forward_pass, tol=1e-5)
    ok &= _run_test(test_loss, tol=1e-5)
    ok &= _run_test(test_train_step, tol=1e-5)
    ok &= _run_test(test_convergence, tol=1e-2)

    if ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")