"""
test_equivalence.py

Cross-framework equivalence tests between pytorch_refactored.py and
jax_code_fixed.py for both the Custom LSTM and Inbuilt LSTM models.
All tests use hardcoded tensors — no RNG anywhere.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import torch
import jax.numpy as jnp
import optax

# ---------------------------------------------------------------------------
# Imports from pytorch_refactored.py (actual function names)
# ---------------------------------------------------------------------------
from pytorch_refactored import (
    CustomLSTMModel,
    LSTMModel,
    make_model_custom,
    make_model_inbuilt,
    make_criterion,
    make_optimizer_custom,
    make_optimizer_inbuilt,
    train_model_custom as pt_train_model_custom,
    train_model_inbuilt as pt_train_model_inbuilt,
)

# ---------------------------------------------------------------------------
# Imports from jax_code_fixed.py (actual function names)
# ---------------------------------------------------------------------------
from jax_code_fixed import (
    custom_lstm_forward as jax_custom_forward,
    inbuilt_lstm_forward as jax_inbuilt_forward,
    mse_loss_custom as jax_loss_custom,
    mse_loss_inbuilt as jax_loss_inbuilt,
    train_step_custom as jax_train_step_custom,
    train_step_inbuilt as jax_train_step_inbuilt,
    train_model_custom as jax_train_model_custom,
    train_model_inbuilt as jax_train_model_inbuilt,
)

# ---------------------------------------------------------------------------
# Hardcoded numpy tensors shared across ALL tests — no RNG anywhere.
#
# Dimensions: batch=4, seq_len=3, input_dim=1, hidden_units=50.
# Both make_model_custom() and make_model_inbuilt() hardcode hidden=50,
# so all test tensors must match this size.
# ---------------------------------------------------------------------------
HIDDEN = 50
SEQ_LEN = 3
BATCH = 4

# Input sequences: (batch, seq_len, 1)
X_np = np.array([
    [[0.1], [0.2], [0.3]],
    [[0.4], [0.5], [0.6]],
    [[0.7], [0.8], [0.9]],
    [[1.0], [1.1], [1.2]],
], dtype=np.float64)

# Targets: (batch, 1)
y_np = np.array([[0.4], [0.7], [1.0], [1.3]], dtype=np.float64)

# Initial hidden and cell states: (batch, hidden) — deterministic, no RNG
H_np = np.full((BATCH, HIDDEN), 0.1, dtype=np.float64)
C_np = np.full((BATCH, HIDDEN), 0.05, dtype=np.float64)

# Custom LSTM gate weights: Wx (1, hidden), Wh (hidden, hidden), b (hidden,)
Wxi_np = np.full((1, HIDDEN), 0.1, dtype=np.float64)
Whi_np = np.full((HIDDEN, HIDDEN), 0.05, dtype=np.float64)
bi_np  = np.zeros(HIDDEN, dtype=np.float64)

Wxf_np = np.full((1, HIDDEN), 0.1, dtype=np.float64)
Whf_np = np.full((HIDDEN, HIDDEN), 0.05, dtype=np.float64)
bf_np  = np.zeros(HIDDEN, dtype=np.float64)

Wxo_np = np.full((1, HIDDEN), 0.1, dtype=np.float64)
Who_np = np.full((HIDDEN, HIDDEN), 0.05, dtype=np.float64)
bo_np  = np.zeros(HIDDEN, dtype=np.float64)

Wxc_np = np.full((1, HIDDEN), 0.1, dtype=np.float64)
Whc_np = np.full((HIDDEN, HIDDEN), 0.05, dtype=np.float64)
bc_np  = np.zeros(HIDDEN, dtype=np.float64)

# FC layer: JAX stores as (hidden, 1), PyTorch stores as (1, hidden)
fc_w_np = np.full((HIDDEN, 1), 0.2, dtype=np.float64)
fc_b_np = np.array([0.01], dtype=np.float64)

# Inbuilt LSTM: concatenated gate weights in PyTorch's [i, f, g, o] order
weight_ih_np = np.full((4 * HIDDEN, 1), 0.1, dtype=np.float64)
weight_hh_np = np.full((4 * HIDDEN, HIDDEN), 0.05, dtype=np.float64)
bias_ih_np   = np.zeros(4 * HIDDEN, dtype=np.float64)
bias_hh_np   = np.zeros(4 * HIDDEN, dtype=np.float64)

inbuilt_fc_w_np = np.full((HIDDEN, 1), 0.2, dtype=np.float64)
inbuilt_fc_b_np = np.array([0.01], dtype=np.float64)

# Test input for inference
X_test_np = np.array([
    [[0.2], [0.3], [0.4]],
    [[0.5], [0.6], [0.7]],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Helper: inject params into PyTorch models (bypasses random init)
# ---------------------------------------------------------------------------
def inject_custom_pytorch_params(model):
    """Inject all hardcoded params into PyTorch CustomLSTMModel."""
    model.Wxi.data = torch.from_numpy(Wxi_np.copy())
    model.Whi.data = torch.from_numpy(Whi_np.copy())
    model.bi.data  = torch.from_numpy(bi_np.copy())
    model.Wxf.data = torch.from_numpy(Wxf_np.copy())
    model.Whf.data = torch.from_numpy(Whf_np.copy())
    model.bf.data  = torch.from_numpy(bf_np.copy())
    model.Wxo.data = torch.from_numpy(Wxo_np.copy())
    model.Who.data = torch.from_numpy(Who_np.copy())
    model.bo.data  = torch.from_numpy(bo_np.copy())
    model.Wxc.data = torch.from_numpy(Wxc_np.copy())
    model.Whc.data = torch.from_numpy(Whc_np.copy())
    model.bc.data  = torch.from_numpy(bc_np.copy())
    # PyTorch fc: weight is (out, in) = (1, hidden)
    model.fc.weight.data = torch.from_numpy(fc_w_np.T.copy())
    model.fc.bias.data   = torch.from_numpy(fc_b_np.copy())


def inject_inbuilt_pytorch_params(model):
    """Inject all hardcoded params into PyTorch LSTMModel."""
    model.lstm.weight_ih_l0.data = torch.from_numpy(weight_ih_np.copy())
    model.lstm.weight_hh_l0.data = torch.from_numpy(weight_hh_np.copy())
    model.lstm.bias_ih_l0.data   = torch.from_numpy(bias_ih_np.copy())
    model.lstm.bias_hh_l0.data   = torch.from_numpy(bias_hh_np.copy())
    model.fc.weight.data = torch.from_numpy(inbuilt_fc_w_np.T.copy())
    model.fc.bias.data   = torch.from_numpy(inbuilt_fc_b_np.copy())


# ---------------------------------------------------------------------------
# Helper: build JAX params dicts from the same numpy arrays
# ---------------------------------------------------------------------------
def make_jax_custom_params():
    """Construct the JAX params dict for the custom LSTM."""
    return {
        'Wxi': jnp.array(Wxi_np), 'Whi': jnp.array(Whi_np), 'bi': jnp.array(bi_np),
        'Wxf': jnp.array(Wxf_np), 'Whf': jnp.array(Whf_np), 'bf': jnp.array(bf_np),
        'Wxo': jnp.array(Wxo_np), 'Who': jnp.array(Who_np), 'bo': jnp.array(bo_np),
        'Wxc': jnp.array(Wxc_np), 'Whc': jnp.array(Whc_np), 'bc': jnp.array(bc_np),
        'fc_w': jnp.array(fc_w_np),
        'fc_b': jnp.array(fc_b_np),
    }


def make_jax_inbuilt_params():
    """Construct the JAX params dict for the inbuilt-equivalent LSTM."""
    return {
        'weight_ih': jnp.array(weight_ih_np),
        'weight_hh': jnp.array(weight_hh_np),
        'bias_ih':   jnp.array(bias_ih_np),
        'bias_hh':   jnp.array(bias_hh_np),
        'fc_w': jnp.array(inbuilt_fc_w_np),
        'fc_b': jnp.array(inbuilt_fc_b_np),
    }


# Float32 variants for convergence tests (Tier 2 uses wider tolerance)
def inject_inbuilt_pytorch_params_f32(model):
    """Inject float32 params into PyTorch LSTMModel."""
    model.lstm.weight_ih_l0.data = torch.from_numpy(weight_ih_np.astype(np.float32))
    model.lstm.weight_hh_l0.data = torch.from_numpy(weight_hh_np.astype(np.float32))
    model.lstm.bias_ih_l0.data   = torch.from_numpy(bias_ih_np.astype(np.float32))
    model.lstm.bias_hh_l0.data   = torch.from_numpy(bias_hh_np.astype(np.float32))
    model.fc.weight.data = torch.from_numpy(inbuilt_fc_w_np.T.astype(np.float32))
    model.fc.bias.data   = torch.from_numpy(inbuilt_fc_b_np.astype(np.float32))


def make_jax_inbuilt_params_f32():
    """Construct float32 JAX params dict for the inbuilt-equivalent LSTM."""
    return {
        'weight_ih': jnp.array(weight_ih_np, dtype=jnp.float32),
        'weight_hh': jnp.array(weight_hh_np, dtype=jnp.float32),
        'bias_ih':   jnp.array(bias_ih_np, dtype=jnp.float32),
        'bias_hh':   jnp.array(bias_hh_np, dtype=jnp.float32),
        'fc_w': jnp.array(inbuilt_fc_w_np, dtype=jnp.float32),
        'fc_b': jnp.array(inbuilt_fc_b_np, dtype=jnp.float32),
    }


# ===================================================================
# Tier 1 — Unit equivalence tests  (tolerance 1e-5)
# ===================================================================
UNIT_TOL = 1e-5


def test_custom_forward_pass():
    """Custom LSTM: identical X, params, and H/C state → same output.

    Both frameworks receive the same deterministic H/C tuple, so no RNG
    is needed for the forward pass.
    """
    H_C_pt = (torch.from_numpy(H_np.copy()), torch.from_numpy(C_np.copy()))
    H_C_jax = (jnp.array(H_np), jnp.array(C_np))

    # PyTorch
    pt_model = make_model_custom()
    pt_model.double()
    inject_custom_pytorch_params(pt_model)
    with torch.no_grad():
        pt_pred, _ = pt_model(torch.from_numpy(X_np), H_C_pt)
    pt_out = pt_pred[:, -1, :].detach().numpy()

    # JAX — pass H_C directly so no RNG key is consumed
    jax_params = make_jax_custom_params()
    jax_pred, _ = jax_custom_forward(jax_params, jnp.array(X_np), H_C_jax, None)
    jax_out = np.array(jax_pred[:, -1, :])

    diff = np.max(np.abs(pt_out - jax_out))
    passed = diff < UNIT_TOL
    return passed, diff


def test_inbuilt_forward_pass():
    """Inbuilt LSTM: identical X and params → same output."""
    # PyTorch
    pt_model = make_model_inbuilt()
    pt_model.double()
    inject_inbuilt_pytorch_params(pt_model)
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(X_np)).detach().numpy()

    # JAX
    jax_params = make_jax_inbuilt_params()
    jax_out = np.array(jax_inbuilt_forward(jax_params, jnp.array(X_np)))

    diff = np.max(np.abs(pt_out - jax_out))
    passed = diff < UNIT_TOL
    return passed, diff


def test_custom_loss():
    """Custom LSTM: identical inputs and H/C state → same MSE loss."""
    H_C_pt = (torch.from_numpy(H_np.copy()), torch.from_numpy(C_np.copy()))
    H_C_jax = (jnp.array(H_np), jnp.array(C_np))

    # PyTorch
    pt_model = make_model_custom()
    pt_model.double()
    inject_custom_pytorch_params(pt_model)
    criterion = make_criterion()
    with torch.no_grad():
        pt_pred, _ = pt_model(torch.from_numpy(X_np), H_C_pt)
        pt_loss = criterion(pt_pred[:, -1, :], torch.from_numpy(y_np)).item()

    # JAX — compute forward + loss manually with H_C to avoid needing a key
    jax_params = make_jax_custom_params()
    jax_pred, _ = jax_custom_forward(jax_params, jnp.array(X_np), H_C_jax, None)
    jax_loss = float(jnp.mean((jax_pred[:, -1, :] - jnp.array(y_np)) ** 2))

    diff = abs(pt_loss - jax_loss)
    passed = diff < UNIT_TOL
    return passed, diff


def test_inbuilt_loss():
    """Inbuilt LSTM: identical inputs → same MSE loss."""
    # PyTorch
    pt_model = make_model_inbuilt()
    pt_model.double()
    inject_inbuilt_pytorch_params(pt_model)
    criterion = make_criterion()
    with torch.no_grad():
        pt_pred = pt_model(torch.from_numpy(X_np))
        pt_loss = criterion(pt_pred, torch.from_numpy(y_np)).item()

    # JAX
    jax_params = make_jax_inbuilt_params()
    jax_loss = float(jax_loss_inbuilt(jax_params, jnp.array(X_np), jnp.array(y_np)))

    diff = abs(pt_loss - jax_loss)
    passed = diff < UNIT_TOL
    return passed, diff


def test_inbuilt_train_step():
    """Inbuilt LSTM: one Adam step from identical state → same loss and params.

    Note: The custom LSTM train_step requires RNG for H/C init (H_C=None
    is hardcoded in mse_loss_custom), so we test the train_step only for
    the inbuilt model where the forward pass is fully deterministic.
    """
    lr = 0.01

    # --- PyTorch ---
    pt_model = make_model_inbuilt()
    pt_model.double()
    inject_inbuilt_pytorch_params(pt_model)
    criterion = make_criterion()
    optimizer = make_optimizer_inbuilt(pt_model)

    pt_pred = pt_model(torch.from_numpy(X_np))
    pt_loss = criterion(pt_pred, torch.from_numpy(y_np))
    optimizer.zero_grad()
    pt_loss.backward()
    optimizer.step()

    pt_loss_val = pt_loss.item()
    pt_w_ih = pt_model.lstm.weight_ih_l0.data.detach().numpy().copy()
    pt_fc_w = pt_model.fc.weight.data.detach().numpy().copy()

    # --- JAX ---
    jax_params = make_jax_inbuilt_params()
    jax_optimizer = optax.adam(learning_rate=lr)
    jax_opt_state = jax_optimizer.init(jax_params)

    jax_new_params, _, jax_loss_val = jax_train_step_inbuilt(
        jax_params, jax_opt_state, jnp.array(X_np), jnp.array(y_np),
        jax_optimizer.update,
    )

    jax_loss_scalar = float(jax_loss_val)
    jax_w_ih = np.array(jax_new_params['weight_ih'])
    # JAX fc_w is (hidden, 1), PyTorch fc.weight is (1, hidden)
    jax_fc_w = np.array(jax_new_params['fc_w']).T

    loss_diff = abs(pt_loss_val - jax_loss_scalar)
    w_ih_diff = np.max(np.abs(pt_w_ih - jax_w_ih))
    fc_w_diff = np.max(np.abs(pt_fc_w - jax_fc_w))
    max_diff = max(loss_diff, w_ih_diff, fc_w_diff)
    passed = max_diff < UNIT_TOL
    return passed, max_diff


def test_custom_train_step():
    """Custom LSTM: one Adam step from identical state → same loss and params.

    Since train_step_custom bakes in H_C=None (requiring RNG), we compute
    one manual gradient step in both frameworks using identical deterministic
    H/C. This tests the same gradient + optimizer logic without needing RNG.
    """
    lr = 0.01
    H_C_pt = (torch.from_numpy(H_np.copy()), torch.from_numpy(C_np.copy()))
    H_C_jax = (jnp.array(H_np), jnp.array(C_np))

    # --- PyTorch: one Adam step ---
    pt_model = make_model_custom()
    pt_model.double()
    inject_custom_pytorch_params(pt_model)
    criterion = make_criterion()
    optimizer = make_optimizer_custom(pt_model)

    pt_pred, _ = pt_model(torch.from_numpy(X_np), H_C_pt)
    pt_loss = criterion(pt_pred[:, -1, :], torch.from_numpy(y_np))
    optimizer.zero_grad()
    pt_loss.backward()
    optimizer.step()

    pt_loss_val = pt_loss.item()
    pt_Wxi = pt_model.Wxi.data.detach().numpy().copy()
    pt_fc_w = pt_model.fc.weight.data.detach().numpy().copy()

    # --- JAX: one manual Adam step with H/C injected ---
    jax_params = make_jax_custom_params()
    jax_optimizer = optax.adam(learning_rate=lr)
    jax_opt_state = jax_optimizer.init(jax_params)

    def custom_loss_with_hc(params, X_seq, y_seq):
        pred, _ = jax_custom_forward(params, X_seq, H_C_jax, None)
        return jnp.mean((pred[:, -1, :] - y_seq) ** 2)

    jax_loss_val, grads = jax.value_and_grad(custom_loss_with_hc)(
        jax_params, jnp.array(X_np), jnp.array(y_np)
    )
    updates, _ = jax_optimizer.update(grads, jax_opt_state, jax_params)
    jax_new_params = optax.apply_updates(jax_params, updates)

    jax_loss_scalar = float(jax_loss_val)
    jax_Wxi = np.array(jax_new_params['Wxi'])
    # JAX fc_w is (hidden, 1), PyTorch fc.weight is (1, hidden)
    jax_fc_w = np.array(jax_new_params['fc_w']).T

    loss_diff = abs(pt_loss_val - jax_loss_scalar)
    Wxi_diff = np.max(np.abs(pt_Wxi - jax_Wxi))
    fc_w_diff = np.max(np.abs(pt_fc_w - jax_fc_w))
    max_diff = max(loss_diff, Wxi_diff, fc_w_diff)
    passed = max_diff < UNIT_TOL
    return passed, max_diff


# ===================================================================
# Tier 2 — Convergence equivalence test  (tolerance 1e-2)
# ===================================================================
CONV_TOL = 1e-2
CONV_EPOCHS = 200


def test_inbuilt_convergence():
    """Inbuilt LSTM: full training from identical start → predictions agree.

    Calls pt_train_model_inbuilt() and jax_train_model_inbuilt() directly.
    Uses float32 since convergence tolerance is 1e-2 and float32 drift
    over many epochs is expected.
    """
    lr = 0.01
    X_f32 = X_np.astype(np.float32)
    y_f32 = y_np.astype(np.float32)
    X_test_f32 = X_test_np.astype(np.float32)

    # --- PyTorch ---
    pt_model = make_model_inbuilt()
    inject_inbuilt_pytorch_params_f32(pt_model)
    criterion = make_criterion()
    optimizer = make_optimizer_inbuilt(pt_model)

    pt_train_model_inbuilt(
        torch.from_numpy(X_f32), torch.from_numpy(y_f32),
        pt_model, optimizer, criterion, num_epochs=CONV_EPOCHS,
    )

    with torch.no_grad():
        pt_preds = pt_model(torch.from_numpy(X_test_f32)).detach().numpy()

    # --- JAX ---
    jax_params = make_jax_inbuilt_params_f32()
    jax_optimizer = optax.adam(learning_rate=lr)
    jax_opt_state = jax_optimizer.init(jax_params)

    jax_params, _ = jax_train_model_inbuilt(
        jnp.array(X_f32), jnp.array(y_f32),
        jax_params, jax_opt_state,
        jax_optimizer.update, num_epochs=CONV_EPOCHS,
    )

    jax_preds = np.array(jax_inbuilt_forward(jax_params, jnp.array(X_test_f32)))

    pred_diff = np.max(np.abs(pt_preds - jax_preds))
    passed = pred_diff < CONV_TOL
    return passed, pred_diff


# ===================================================================
# __main__ runner
# ===================================================================
if __name__ == "__main__":
    tests = [
        # Tier 1 — Unit equivalence (tolerance 1e-5)
        ("test_custom_forward_pass",   test_custom_forward_pass,   UNIT_TOL),
        ("test_inbuilt_forward_pass",  test_inbuilt_forward_pass,  UNIT_TOL),
        ("test_custom_loss",           test_custom_loss,           UNIT_TOL),
        ("test_inbuilt_loss",          test_inbuilt_loss,          UNIT_TOL),
        ("test_custom_train_step",     test_custom_train_step,     UNIT_TOL),
        ("test_inbuilt_train_step",    test_inbuilt_train_step,    UNIT_TOL),
        # Tier 2 — Convergence (tolerance 1e-2)
        # Note: test_custom_convergence is omitted because the custom LSTM
        # training loop resets H_C=None each epoch, causing torch.randn /
        # jax.random.normal to generate different H/C values across
        # frameworks (different PRNG algorithms). This makes the gradient
        # trajectories diverge from epoch 1 in a way that is not a
        # translation bug. The custom model's math is fully verified by
        # the Tier 1 forward, loss, and train_step tests with injected H/C.
        ("test_inbuilt_convergence",   test_inbuilt_convergence,   CONV_TOL),
    ]

    all_passed = True
    print("=" * 70)
    print("Cross-framework equivalence tests: PyTorch <-> JAX (LSTM)")
    print("=" * 70)

    for name, fn, tol in tests:
        passed, diff = fn()
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name:<30s}  max_diff={diff:.2e}  tol={tol:.0e}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)