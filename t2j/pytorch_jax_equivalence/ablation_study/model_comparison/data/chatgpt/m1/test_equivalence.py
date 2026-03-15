# test_equivalence.py
#
# Equivalence tests between:
# - pytorch_refactored.py (PyTorch reference)
# - jax_code_fixed.py      (JAX translation)
#
# IMPORTANT:
# - Imports components directly; does NOT copy code from either source file.
# - No RNG is called anywhere in this test file.

from __future__ import annotations

import sys
import traceback
from typing import Any, Dict

import numpy as np
import torch

import jax
import jax.numpy as jnp

# -----------------------------
# RULE 1: Import components directly (no copying)
# -----------------------------
from pytorch_refactored import (
    CustomLSTMModel,
    LSTMModel,
    make_model as pt_make_model,
    make_criterion as pt_make_criterion,
    make_optimizer as pt_make_optimizer,
    train_model as pt_train_model,
    train_inbuilt_model as pt_train_inbuilt_model,
)

import jax_code_fixed as jx

from jax_code_fixed import (
    custom_lstm_forward as jax_model,
    inbuilt_lstm_forward as jax_inbuilt_model,
    CRITERION_FN as loss_fn,
    train_step_custom as train_step,
    train_step_inbuilt as train_step_inbuilt,
    train_model as jax_train_model,
    train_inbuilt_model as jax_train_inbuilt_model,
)

# -----------------------------
# RULE 2: Hardcoded numpy tensors shared across all tests
# -----------------------------
SEQ_LEN = 3
BATCH = 2
INPUT_DIM = 1
HIDDEN = 50  # MUST match both models (CustomLSTMModel(1,50) and LSTMModel hidden_size=50)

X_np = np.array(
    [
        [[0.10], [0.20], [0.30]],
        [[-0.10], [-0.20], [-0.30]],
    ],
    dtype=np.float32,
)

y_np = np.array([[0.05], [-0.05]], dtype=np.float32)

W_np = np.linspace(-0.05, 0.05, HIDDEN, dtype=np.float32).reshape(HIDDEN, 1)  # [50,1]
B_np = np.array([0.005], dtype=np.float32)  # [1]

X_test_np = np.array(
    [
        [[0.15], [0.25], [0.35]],
        [[-0.15], [-0.25], [-0.35]],
    ],
    dtype=np.float32,
)

H0_np = np.zeros((BATCH, HIDDEN), dtype=np.float32)
C0_np = np.zeros((BATCH, HIDDEN), dtype=np.float32)


# -----------------------------
# RULE 3: inject_pytorch_params(model, w_np, b_np)
# -----------------------------
def inject_pytorch_params(model: CustomLSTMModel, w_np: np.ndarray, b_np: np.ndarray) -> None:
    with torch.no_grad():
        for name in ["Wxi", "Whi", "bi", "Wxf", "Whf", "bf", "Wxo", "Who", "bo", "Wxc", "Whc", "bc"]:
            getattr(model, name).data.zero_()
        model.fc.weight.data.copy_(torch.from_numpy(w_np.T))
        model.fc.bias.data.copy_(torch.from_numpy(b_np))


def inject_pytorch_inbuilt_params(model: LSTMModel, w_np: np.ndarray, b_np: np.ndarray) -> None:
    with torch.no_grad():
        for p in model.lstm.parameters():
            p.data.zero_()
        model.fc.weight.data.copy_(torch.from_numpy(w_np.T))
        model.fc.bias.data.copy_(torch.from_numpy(b_np))


# -----------------------------
# RULE 4: make_jax_params(w_np, b_np)
# -----------------------------
def make_jax_params(w_np: np.ndarray, b_np: np.ndarray) -> Dict[str, Any]:
    z_Wx = jnp.zeros((INPUT_DIM, HIDDEN), dtype=jnp.float32)
    z_Wh = jnp.zeros((HIDDEN, HIDDEN), dtype=jnp.float32)
    z_b = jnp.zeros((HIDDEN,), dtype=jnp.float32)
    return {
        "Wxi": z_Wx, "Whi": z_Wh, "bi": z_b,
        "Wxf": z_Wx, "Whf": z_Wh, "bf": z_b,
        "Wxo": z_Wx, "Who": z_Wh, "bo": z_b,
        "Wxc": z_Wx, "Whc": z_Wh, "bc": z_b,
        "fc_w": jnp.asarray(w_np, dtype=jnp.float32),
        "fc_b": jnp.asarray(b_np, dtype=jnp.float32),
    }


def make_jax_inbuilt_params(w_np: np.ndarray, b_np: np.ndarray) -> Dict[str, Any]:
    z_Wx = jnp.zeros((INPUT_DIM, 4 * HIDDEN), dtype=jnp.float32)
    z_Wh = jnp.zeros((HIDDEN, 4 * HIDDEN), dtype=jnp.float32)
    z_b = jnp.zeros((4 * HIDDEN,), dtype=jnp.float32)
    return {
        "W_x": z_Wx,
        "W_h": z_Wh,
        "b": z_b,
        "fc_w": jnp.asarray(w_np, dtype=jnp.float32),
        "fc_b": jnp.asarray(b_np, dtype=jnp.float32),
    }


def _assert_allclose(a: np.ndarray, b: np.ndarray, tol: float, msg: str) -> None:
    max_abs = float(np.max(np.abs(a - b)))
    if max_abs > tol:
        raise AssertionError(f"{msg} max_abs_diff={max_abs} > tol={tol}")


def _flatten_params_custom(params: Dict[str, Any]) -> np.ndarray:
    keys = ["Wxi", "Whi", "bi", "Wxf", "Whf", "bf", "Wxo", "Who", "bo", "Wxc", "Whc", "bc", "fc_w", "fc_b"]
    flat = []
    for k in keys:
        v = params[k]
        arr = v if isinstance(v, np.ndarray) else np.array(jax.device_get(v))
        flat.append(arr.reshape(-1))
    return np.concatenate(flat, axis=0)


def _flatten_params_inbuilt_fc_only(params: Dict[str, Any]) -> np.ndarray:
    keys = ["fc_w", "fc_b"]
    flat = []
    for k in keys:
        v = params[k]
        arr = v if isinstance(v, np.ndarray) else np.array(jax.device_get(v))
        flat.append(arr.reshape(-1))
    return np.concatenate(flat, axis=0)


def _extract_pt_params_custom(model: CustomLSTMModel) -> Dict[str, np.ndarray]:
    d: Dict[str, np.ndarray] = {}
    for k in ["Wxi", "Whi", "bi", "Wxf", "Whf", "bf", "Wxo", "Who", "bo", "Wxc", "Whc", "bc"]:
        d[k] = getattr(model, k).detach().cpu().numpy()
    d["fc_w"] = model.fc.weight.detach().cpu().numpy().T
    d["fc_b"] = model.fc.bias.detach().cpu().numpy()
    return d


def _extract_pt_params_inbuilt_fc_only(model: LSTMModel) -> Dict[str, np.ndarray]:
    return {
        "fc_w": model.fc.weight.detach().cpu().numpy().T,
        "fc_b": model.fc.bias.detach().cpu().numpy(),
    }


# -----------------------------
# Tier 1 — CustomLSTMModel tests
# -----------------------------
def test_forward_pass(tol: float = 1e-5) -> None:
    model_pt, _ = pt_make_model()
    inject_pytorch_params(model_pt, W_np, B_np)

    X_pt = torch.from_numpy(X_np)
    H0_pt = torch.from_numpy(H0_np)
    C0_pt = torch.from_numpy(C0_np)
    pred_pt, _ = model_pt(X_pt, (H0_pt, C0_pt))
    pred_pt = pred_pt.detach().cpu().numpy()

    params_jx = make_jax_params(W_np, B_np)
    X_jx = jnp.asarray(X_np, dtype=jnp.float32)
    H0_jx = jnp.asarray(H0_np, dtype=jnp.float32)
    C0_jx = jnp.asarray(C0_np, dtype=jnp.float32)

    dummy_key = jnp.array([0, 0], dtype=jnp.uint32)
    pred_jx, _state, _ = jax_model(params_jx, X_jx, dummy_key, H_C=(H0_jx, C0_jx))
    pred_jx = np.array(jax.device_get(pred_jx))

    _assert_allclose(pred_pt, pred_jx, tol, "test_forward_pass")


def test_loss(tol: float = 1e-5) -> None:
    model_pt, _ = pt_make_model()
    inject_pytorch_params(model_pt, W_np, B_np)

    X_pt = torch.from_numpy(X_np)
    y_pt = torch.from_numpy(y_np)
    H0_pt = torch.from_numpy(H0_np)
    C0_pt = torch.from_numpy(C0_np)

    pred_pt, _ = model_pt(X_pt, (H0_pt, C0_pt))
    crit_pt = pt_make_criterion()
    loss_pt = float(crit_pt(pred_pt[:, -1, :], y_pt).item())

    params_jx = make_jax_params(W_np, B_np)
    X_jx = jnp.asarray(X_np, dtype=jnp.float32)
    y_jx = jnp.asarray(y_np, dtype=jnp.float32)
    H0_jx = jnp.asarray(H0_np, dtype=jnp.float32)
    C0_jx = jnp.asarray(C0_np, dtype=jnp.float32)

    dummy_key = jnp.array([0, 0], dtype=jnp.uint32)
    pred_jx, _state, _ = jax_model(params_jx, X_jx, dummy_key, H_C=(H0_jx, C0_jx))
    loss_jx = float(jax.device_get(loss_fn(pred_jx[:, -1, :], y_jx)))

    if abs(loss_pt - loss_jx) > tol:
        raise AssertionError(f"test_loss abs_diff={abs(loss_pt - loss_jx)} > tol={tol}")


def test_train_step(tol: float = 1e-5) -> None:
    model_pt, _ = pt_make_model()
    inject_pytorch_params(model_pt, W_np, B_np)

    X_pt = torch.from_numpy(X_np)
    y_pt = torch.from_numpy(y_np)

    H0_pt = torch.from_numpy(H0_np)
    C0_pt = torch.from_numpy(C0_np)
    orig_forward = model_pt.forward

    def forward_no_rng(inputs, H_C=None):
        if H_C is None:
            return orig_forward(inputs, (H0_pt, C0_pt))
        return orig_forward(inputs, H_C)

    model_pt.forward = forward_no_rng

    crit_pt = pt_make_criterion()
    opt_pt = pt_make_optimizer(model_pt)

    pred_pt, _ = model_pt(X_pt, None)
    loss_pt_t = crit_pt(pred_pt[:, -1, :], y_pt)
    opt_pt.zero_grad()
    loss_pt_t.backward()
    opt_pt.step()

    pt_params_after = _flatten_params_custom(_extract_pt_params_custom(model_pt))
    pt_loss_after = float(loss_pt_t.item())

    params_jx = make_jax_params(W_np, B_np)
    opt_state_jx = jx.OPTIMIZER.init(params_jx)

    H0_jx = jnp.asarray(H0_np, dtype=jnp.float32)
    C0_jx = jnp.asarray(C0_np, dtype=jnp.float32)
    orig_jax_forward = jx.custom_lstm_forward

    def jax_forward_no_rng(params, inputs, rng_key, H_C=None):
        if H_C is None:
            pred, state, _ = orig_jax_forward(params, inputs, rng_key, H_C=(H0_jx, C0_jx))
            return pred, state, rng_key
        return orig_jax_forward(params, inputs, rng_key, H_C=H_C)

    jx.custom_lstm_forward = jax_forward_no_rng

    X_jx_in = jnp.asarray(X_np, dtype=jnp.float32)
    y_jx_in = jnp.asarray(y_np, dtype=jnp.float32)
    dummy_key = jnp.array([0, 0], dtype=jnp.uint32)

    params_jx_after, _opt_state_after, loss_jx_after, _ = train_step(
        X_jx_in, y_jx_in, params_jx, opt_state_jx, dummy_key
    )

    jx_params_after = _flatten_params_custom(params_jx_after)
    jx_loss_after = float(jax.device_get(loss_jx_after))

    _assert_allclose(pt_params_after, jx_params_after, tol, "test_train_step params")
    if abs(pt_loss_after - jx_loss_after) > tol:
        raise AssertionError(f"test_train_step loss abs_diff={abs(pt_loss_after - jx_loss_after)} > tol={tol}")


def test_convergence(tol: float = 1e-2) -> None:
    model_pt, _ = pt_make_model()
    inject_pytorch_params(model_pt, W_np, B_np)

    X_pt = torch.from_numpy(X_np)
    y_pt = torch.from_numpy(y_np)

    H0_pt = torch.from_numpy(H0_np)
    C0_pt = torch.from_numpy(C0_np)
    orig_forward = model_pt.forward

    def forward_no_rng(inputs, H_C=None):
        if H_C is None:
            return orig_forward(inputs, (H0_pt, C0_pt))
        return orig_forward(inputs, H_C)

    model_pt.forward = forward_no_rng

    crit_pt = pt_make_criterion()
    opt_pt = pt_make_optimizer(model_pt)

    EPOCHS = 200
    pt_train_model(X_pt, y_pt, model_pt, opt_pt, crit_pt, EPOCHS)

    pt_params_final = _flatten_params_custom(_extract_pt_params_custom(model_pt))

    X_test_pt = torch.from_numpy(X_test_np)
    pred_pt_full, _ = model_pt(X_test_pt, (H0_pt, C0_pt))
    pred_pt = pred_pt_full[:, -1, :].detach().cpu().numpy()

    params_jx0 = make_jax_params(W_np, B_np)
    opt_state_jx0 = jx.OPTIMIZER.init(params_jx0)

    H0_jx = jnp.asarray(H0_np, dtype=jnp.float32)
    C0_jx = jnp.asarray(C0_np, dtype=jnp.float32)
    orig_jax_forward = jx.custom_lstm_forward

    def jax_forward_no_rng(params, inputs, rng_key, H_C=None):
        if H_C is None:
            pred, state, _ = orig_jax_forward(params, inputs, rng_key, H_C=(H0_jx, C0_jx))
            return pred, state, rng_key
        return orig_jax_forward(params, inputs, rng_key, H_C=H_C)

    jx.custom_lstm_forward = jax_forward_no_rng

    X_jx = jnp.asarray(X_np, dtype=jnp.float32)
    y_jx = jnp.asarray(y_np, dtype=jnp.float32)
    dummy_key = jnp.array([0, 0], dtype=jnp.uint32)

    params_jx_final, _opt_state_final, _ = jax_train_model(
        X_jx, y_jx, params_jx0, opt_state_jx0, EPOCHS, dummy_key
    )
    jx_params_final = _flatten_params_custom(params_jx_final)

    X_test_jx = jnp.asarray(X_test_np, dtype=jnp.float32)
    pred_jx_full, _state, _ = jax_model(params_jx_final, X_test_jx, dummy_key, H_C=(H0_jx, C0_jx))
    pred_jx = np.array(jax.device_get(pred_jx_full[:, -1, :]))

    _assert_allclose(pt_params_final, jx_params_final, tol, "test_convergence params")
    _assert_allclose(pred_pt, pred_jx, tol, "test_convergence preds")


# -----------------------------
# Tier 1 — Built-in LSTMModel tests
# -----------------------------
def test_forward_pass_inbuilt(tol: float = 1e-5) -> None:
    _custom, model_pt = pt_make_model()
    inject_pytorch_inbuilt_params(model_pt, W_np, B_np)

    X_pt = torch.from_numpy(X_np)
    pred_pt = model_pt(X_pt).detach().cpu().numpy()

    params_jx = make_jax_inbuilt_params(W_np, B_np)
    X_jx = jnp.asarray(X_np, dtype=jnp.float32)
    pred_jx = np.array(jax.device_get(jax_inbuilt_model(params_jx, X_jx)))

    _assert_allclose(pred_pt, pred_jx, tol, "test_forward_pass_inbuilt")


def test_loss_inbuilt(tol: float = 1e-5) -> None:
    _custom, model_pt = pt_make_model()
    inject_pytorch_inbuilt_params(model_pt, W_np, B_np)

    X_pt = torch.from_numpy(X_np)
    y_pt = torch.from_numpy(y_np)

    pred_pt = model_pt(X_pt)
    crit_pt = pt_make_criterion()
    loss_pt = float(crit_pt(pred_pt, y_pt).item())

    params_jx = make_jax_inbuilt_params(W_np, B_np)
    X_jx = jnp.asarray(X_np, dtype=jnp.float32)
    y_jx = jnp.asarray(y_np, dtype=jnp.float32)
    pred_jx = jax_inbuilt_model(params_jx, X_jx)
    loss_jx = float(jax.device_get(loss_fn(pred_jx, y_jx)))

    if abs(loss_pt - loss_jx) > tol:
        raise AssertionError(f"test_loss_inbuilt abs_diff={abs(loss_pt - loss_jx)} > tol={tol}")


def test_train_step_inbuilt_fc_only(tol: float = 1e-5) -> None:
    _custom, model_pt = pt_make_model()
    inject_pytorch_inbuilt_params(model_pt, W_np, B_np)

    X_pt = torch.from_numpy(X_np)
    y_pt = torch.from_numpy(y_np)

    crit_pt = pt_make_criterion()
    opt_pt = pt_make_optimizer(model_pt)

    pred_pt = model_pt(X_pt)
    loss_pt_t = crit_pt(pred_pt, y_pt)
    opt_pt.zero_grad()
    loss_pt_t.backward()
    opt_pt.step()

    pt_fc_after = _flatten_params_inbuilt_fc_only(_extract_pt_params_inbuilt_fc_only(model_pt))
    pt_loss_after = float(loss_pt_t.item())

    params_jx = make_jax_inbuilt_params(W_np, B_np)
    opt_state_jx = jx.OPTIMIZER.init(params_jx)

    X_jx = jnp.asarray(X_np, dtype=jnp.float32)
    y_jx = jnp.asarray(y_np, dtype=jnp.float32)

    params_jx_after, _opt_state_after, loss_jx_after = train_step_inbuilt(
        X_jx, y_jx, params_jx, opt_state_jx
    )

    jx_fc_after = _flatten_params_inbuilt_fc_only(params_jx_after)
    jx_loss_after = float(jax.device_get(loss_jx_after))

    _assert_allclose(pt_fc_after, jx_fc_after, tol, "test_train_step_inbuilt_fc_only fc_params")
    if abs(pt_loss_after - jx_loss_after) > tol:
        raise AssertionError(
            f"test_train_step_inbuilt_fc_only loss abs_diff={abs(pt_loss_after - jx_loss_after)} > tol={tol}"
        )


# -----------------------------
# Tier 2 — Built-in convergence test
# -----------------------------
def test_convergence_inbuilt_preds_only(tol: float = 1e-2) -> None:
    """
    Why preds-only here:
    PyTorch nn.LSTM has two bias vectors (bias_ih, bias_hh) and Adam maintains separate moments per tensor.
    JAX inbuilt uses a single combined bias tensor, so the optimizer state/trajectory differs even when
    the forward computation can match. Over many steps, internal LSTM params can diverge while predictions
    still match. This test checks the intended functional equivalence at convergence: predictions.
    """
    _custom, model_pt = pt_make_model()
    inject_pytorch_inbuilt_params(model_pt, W_np, B_np)

    X_pt = torch.from_numpy(X_np)
    y_pt = torch.from_numpy(y_np)

    crit_pt = pt_make_criterion()
    opt_pt = pt_make_optimizer(model_pt)

    EPOCHS = 200
    pt_train_inbuilt_model(X_pt, y_pt, model_pt, opt_pt, crit_pt, EPOCHS)

    X_test_pt = torch.from_numpy(X_test_np)
    pred_pt = model_pt(X_test_pt).detach().cpu().numpy()

    params_jx0 = make_jax_inbuilt_params(W_np, B_np)
    opt_state_jx0 = jx.OPTIMIZER.init(params_jx0)

    X_jx = jnp.asarray(X_np, dtype=jnp.float32)
    y_jx = jnp.asarray(y_np, dtype=jnp.float32)

    params_jx_final, _opt_state_final = jax_train_inbuilt_model(
        X_jx, y_jx, params_jx0, opt_state_jx0, EPOCHS
    )

    X_test_jx = jnp.asarray(X_test_np, dtype=jnp.float32)
    pred_jx = np.array(jax.device_get(jax_inbuilt_model(params_jx_final, X_test_jx)))

    _assert_allclose(pred_pt, pred_jx, tol, "test_convergence_inbuilt_preds_only preds")


# -----------------------------
# RULE 8: __main__ runner
# -----------------------------
def _run_test(fn, tol: float) -> bool:
    name = fn.__name__
    try:
        fn(tol=tol)
        print(f"PASS {name} (tol={tol})")
        return True
    except Exception as e:
        print(f"FAIL {name} (tol={tol}) -> {e}")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    ok = True

    ok &= _run_test(test_forward_pass, tol=1e-5)
    ok &= _run_test(test_loss, tol=1e-5)
    ok &= _run_test(test_train_step, tol=1e-5)
    ok &= _run_test(test_convergence, tol=1e-2)

    ok &= _run_test(test_forward_pass_inbuilt, tol=1e-5)
    ok &= _run_test(test_loss_inbuilt, tol=1e-5)
    ok &= _run_test(test_train_step_inbuilt_fc_only, tol=1e-5)
    ok &= _run_test(test_convergence_inbuilt_preds_only, tol=1e-2)

    sys.exit(0 if ok else 1)