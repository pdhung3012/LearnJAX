import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import unfreeze, freeze
import shutil, os, inspect

# Clear __pycache__ to avoid stale bytecode
for d in ['.', os.path.dirname(os.path.abspath(__file__))]:
    cache = os.path.join(d, '__pycache__')
    if os.path.exists(cache):
        shutil.rmtree(cache)

from pytorch_refactored import (
    CustomLSTMModel as PtCustomModel,
    LSTMModel as PtBuiltinModel,
    train_model as pt_train_model
)

from jax_code_fixed import (
    CustomLSTMModel as JaxCustomModel,
    LSTMModel as JaxBuiltinModel,
)

# Verify imports
src = inspect.getsource(JaxCustomModel)
assert 'lax.scan' not in src, "ERROR: jax_code_fixed.py still uses lax.scan!"
assert 'for t in range' in src, "ERROR: jax_code_fixed.py missing explicit loop!"
print("[OK] jax_code_fixed.py verified: uses explicit loop (no lax.scan)")

# Hardcoded numpy tensors
BATCH, SEQ, DIM, HIDDEN = 2, 5, 1, 50
X_np = np.linspace(-1, 1, BATCH * SEQ * DIM).reshape(BATCH, SEQ, DIM).astype(np.float32)
y_np = np.array([[0.5], [0.8]], dtype=np.float32)

W_np = np.ones((DIM, HIDDEN), dtype=np.float32) * 0.1
H_np = np.ones((HIDDEN, HIDDEN), dtype=np.float32) * 0.05
B_np = np.zeros((HIDDEN,), dtype=np.float32)
W_fc_np = np.ones((HIDDEN, 1), dtype=np.float32) * 0.2
B_fc_np = np.array([0.1], dtype=np.float32)
LR = 0.01


def inject_pytorch_params(model, is_custom=True):
    with torch.no_grad():
        if is_custom:
            for gate in ['i', 'f', 'o', 'c']:
                getattr(model, f'Wx{gate}').data = torch.from_numpy(W_np.copy())
                getattr(model, f'Wh{gate}').data = torch.from_numpy(H_np.copy())
                getattr(model, f'b{gate}').data = torch.from_numpy(B_np.copy())
        else:
            # PyTorch nn.LSTM gate order in stacked weights: [i, f, g, o]
            # weight_ih shape: (4*hidden, input) = (200, 1)
            # weight_hh shape: (4*hidden, hidden) = (200, 50)
            model.lstm.weight_ih_l0.data = torch.from_numpy(np.tile(W_np.T, (4, 1)))
            model.lstm.weight_hh_l0.data = torch.from_numpy(np.tile(H_np.T, (4, 1)))
            # PyTorch has TWO bias vectors: bias_ih and bias_hh
            # gate_value = W_ih @ x + b_ih + W_hh @ h + b_hh
            model.lstm.bias_ih_l0.data = torch.from_numpy(np.tile(B_np, (4,)))
            model.lstm.bias_hh_l0.data = torch.zeros(4 * HIDDEN)
        model.fc.weight.data = torch.from_numpy(W_fc_np.T.copy())
        model.fc.bias.data = torch.from_numpy(B_fc_np.copy())


def make_jax_params(model, is_custom=True):
    init_key = jax.random.PRNGKey(0)
    if is_custom:
        variables = model.init(init_key, jnp.array(X_np), init_key)
    else:
        variables = model.init(init_key, jnp.array(X_np))
    params = unfreeze(variables['params'])

    if is_custom:
        for gate in ['i', 'f', 'o', 'c']:
            params[f'Wx{gate}'] = jnp.array(W_np)
            params[f'Wh{gate}'] = jnp.array(H_np)
            params[f'b{gate}'] = jnp.array(B_np)
        params['Dense_0'] = {'kernel': jnp.array(W_fc_np), 'bias': jnp.array(B_fc_np)}
    else:
        # JAX LSTMModel has per-gate Dense layers with BOTH input and hidden biases:
        #   gate = dense_ix(x) + dense_hx(h)
        #        = (kernel_i @ x + bias_i) + (kernel_h @ h + bias_h)
        #
        # This matches PyTorch: gate = W_ih @ x + b_ih + W_hh @ h + b_hh
        #
        # PyTorch injection: bias_ih = tile(B_np, 4), bias_hh = zeros(200)
        # So for each gate: b_ih_gate = B_np, b_hh_gate = zeros
        #
        # JAX mapping:
        #   input Dense (ii, ig, if, io): bias = B_np  (matches bias_ih per gate)
        #   hidden Dense (hi, hg, hf, ho): bias = zeros (matches bias_hh per gate)
        #
        # Gate name mapping (PyTorch -> JAX):
        #   i (input)     -> ii/hi
        #   f (forget)    -> if/hf
        #   g (candidate) -> ig/hg
        #   o (output)    -> io/ho

        B_zeros = np.zeros((HIDDEN,), dtype=np.float32)

        # Input-to-hidden gates: kernel=W_np, bias=B_np (matching PyTorch bias_ih)
        for gate_name in ['ii', 'ig', 'if', 'io']:
            params[gate_name] = {
                'kernel': jnp.array(W_np),
                'bias': jnp.array(B_np),
            }

        # Hidden-to-hidden gates: kernel=H_np, bias=zeros (matching PyTorch bias_hh)
        for gate_name in ['hi', 'hg', 'hf', 'ho']:
            params[gate_name] = {
                'kernel': jnp.array(H_np),
                'bias': jnp.array(B_zeros),
            }

        params['Dense_0'] = {'kernel': jnp.array(W_fc_np), 'bias': jnp.array(B_fc_np)}

    return freeze(params)


# =====================================================================
# Tests
# =====================================================================

def test_forward_pass(mode="custom"):
    is_custom = (mode == "custom")
    pt_m = PtCustomModel(DIM, HIDDEN) if is_custom else PtBuiltinModel()
    jax_m = JaxCustomModel(DIM, HIDDEN) if is_custom else JaxBuiltinModel()
    inject_pytorch_params(pt_m, is_custom)
    jax_p = make_jax_params(jax_m, is_custom)

    pt_m.eval()
    with torch.no_grad():
        if is_custom:
            pt_out = pt_m(torch.from_numpy(X_np),
                          (torch.zeros(BATCH, HIDDEN), torch.zeros(BATCH, HIDDEN)))[0][:, -1, :]
        else:
            pt_out = pt_m(torch.from_numpy(X_np))

    if is_custom:
        zero_HC = (jnp.zeros((BATCH, HIDDEN)), jnp.zeros((BATCH, HIDDEN)))
        jax_out = jax_m.apply({'params': jax_p}, jnp.array(X_np),
                              jax.random.PRNGKey(0), H_C=zero_HC)[0][:, -1, :]
    else:
        jax_out = jax_m.apply({'params': jax_p}, jnp.array(X_np))

    diff = np.abs(pt_out.detach().numpy() - np.array(jax_out)).max()
    return diff < 1e-5, diff


def test_loss(mode="custom"):
    is_custom = (mode == "custom")
    pt_m = PtCustomModel(DIM, HIDDEN) if is_custom else PtBuiltinModel()
    inject_pytorch_params(pt_m, is_custom)

    with torch.no_grad():
        if is_custom:
            pt_out = pt_m(torch.from_numpy(X_np),
                          (torch.zeros(BATCH, HIDDEN), torch.zeros(BATCH, HIDDEN)))[0][:, -1, :]
        else:
            pt_out = pt_m(torch.from_numpy(X_np))
        pt_loss = nn.MSELoss()(pt_out, torch.from_numpy(y_np))

    jax_m = JaxCustomModel(DIM, HIDDEN) if is_custom else JaxBuiltinModel()
    jax_p = make_jax_params(jax_m, is_custom)

    if is_custom:
        H_C = (jnp.zeros((BATCH, HIDDEN)), jnp.zeros((BATCH, HIDDEN)))
        pred, _ = jax_m.apply({'params': jax_p}, jnp.array(X_np),
                              jax.random.PRNGKey(0), H_C=H_C)
        jax_loss = jnp.mean((pred[:, -1, :] - jnp.array(y_np)) ** 2)
    else:
        pred = jax_m.apply({'params': jax_p}, jnp.array(X_np))
        jax_loss = jnp.mean((pred - jnp.array(y_np)) ** 2)

    diff = np.abs(pt_loss.item() - float(jax_loss))
    return diff < 1e-5, diff


def test_train_step(mode="custom"):
    is_custom = (mode == "custom")

    # --- PyTorch: forward + backward + manual SGD ---
    pt_m = PtCustomModel(DIM, HIDDEN) if is_custom else PtBuiltinModel()
    inject_pytorch_params(pt_m, is_custom)

    pt_m.train()
    if is_custom:
        pt_out = pt_m(torch.from_numpy(X_np),
                       (torch.zeros(BATCH, HIDDEN), torch.zeros(BATCH, HIDDEN)))[0][:, -1, :]
    else:
        pt_out = pt_m(torch.from_numpy(X_np))

    pt_loss = nn.MSELoss()(pt_out, torch.from_numpy(y_np))
    pt_loss.backward()

    with torch.no_grad():
        for p in pt_m.parameters():
            p.data -= LR * p.grad

    pt_m.eval()
    with torch.no_grad():
        if is_custom:
            pt_out2 = pt_m(torch.from_numpy(X_np),
                           (torch.zeros(BATCH, HIDDEN), torch.zeros(BATCH, HIDDEN)))[0][:, -1, :]
        else:
            pt_out2 = pt_m(torch.from_numpy(X_np))

    # --- JAX: forward + grad + manual SGD ---
    jax_m = JaxCustomModel(DIM, HIDDEN) if is_custom else JaxBuiltinModel()
    jax_p = make_jax_params(jax_m, is_custom)

    if is_custom:
        def loss_fn(p):
            H_C = (jnp.zeros((BATCH, HIDDEN)), jnp.zeros((BATCH, HIDDEN)))
            pred, _ = jax_m.apply({'params': p}, jnp.array(X_np),
                                  jax.random.PRNGKey(0), H_C=H_C)
            return jnp.mean((pred[:, -1, :] - jnp.array(y_np)) ** 2)
    else:
        def loss_fn(p):
            pred = jax_m.apply({'params': p}, jnp.array(X_np))
            return jnp.mean((pred - jnp.array(y_np)) ** 2)

    jax_loss, grads = jax.value_and_grad(loss_fn)(jax_p)
    new_jax_p = jax.tree.map(lambda p, g: p - LR * g, jax_p, grads)

    if is_custom:
        zero_HC = (jnp.zeros((BATCH, HIDDEN)), jnp.zeros((BATCH, HIDDEN)))
        jax_out2 = jax_m.apply({'params': new_jax_p}, jnp.array(X_np),
                               jax.random.PRNGKey(0), H_C=zero_HC)[0][:, -1, :]
    else:
        jax_out2 = jax_m.apply({'params': new_jax_p}, jnp.array(X_np))

    loss_diff = np.abs(pt_loss.item() - float(jax_loss))
    output_diff = np.abs(pt_out2.detach().numpy() - np.array(jax_out2)).max()
    diff = max(loss_diff, output_diff)
    return diff < 1e-5, diff


def test_convergence(mode="custom"):
    is_custom = (mode == "custom")
    num_epochs = 10

    # --- PyTorch ---
    pt_m = PtCustomModel(DIM, HIDDEN) if is_custom else PtBuiltinModel()
    inject_pytorch_params(pt_m, is_custom)

    X_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(y_np)
    h0 = torch.zeros(BATCH, HIDDEN)
    c0 = torch.zeros(BATCH, HIDDEN)

    for epoch in range(num_epochs):
        pt_m.train()
        if is_custom:
            pt_out = pt_m(X_t, (h0, c0))[0][:, -1, :]
        else:
            pt_out = pt_m(X_t)
        loss = nn.MSELoss()(pt_out, y_t)

        for p in pt_m.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()

        with torch.no_grad():
            for p in pt_m.parameters():
                p.data -= LR * p.grad

    pt_m.eval()
    with torch.no_grad():
        if is_custom:
            pt_final = pt_m(X_t, (h0, c0))[0][:, -1, :]
        else:
            pt_final = pt_m(X_t)

    # --- JAX ---
    jax_m = JaxCustomModel(DIM, HIDDEN) if is_custom else JaxBuiltinModel()
    jax_p = make_jax_params(jax_m, is_custom)

    X_j = jnp.array(X_np)
    y_j = jnp.array(y_np)

    if is_custom:
        def loss_fn(p):
            H_C = (jnp.zeros((BATCH, HIDDEN)), jnp.zeros((BATCH, HIDDEN)))
            pred, _ = jax_m.apply({'params': p}, X_j, jax.random.PRNGKey(0), H_C=H_C)
            return jnp.mean((pred[:, -1, :] - y_j) ** 2)
    else:
        def loss_fn(p):
            pred = jax_m.apply({'params': p}, X_j)
            return jnp.mean((pred - y_j) ** 2)

    for epoch in range(num_epochs):
        loss, grads = jax.value_and_grad(loss_fn)(jax_p)
        jax_p = jax.tree.map(lambda p, g: p - LR * g, jax_p, grads)

    if is_custom:
        zero_HC = (jnp.zeros((BATCH, HIDDEN)), jnp.zeros((BATCH, HIDDEN)))
        jax_final = jax_m.apply({'params': jax_p}, X_j,
                                jax.random.PRNGKey(0), H_C=zero_HC)[0][:, -1, :]
    else:
        jax_final = jax_m.apply({'params': jax_p}, X_j)

    diff = np.mean(np.abs(pt_final.detach().numpy() - np.array(jax_final)))
    return diff < 1e-2, diff


if __name__ == "__main__":
    for mode in ["custom", "builtin"]:
        print(f"\n--- Testing {mode.upper()} Model Equivalence ---")
        results = [
            ("Forward Pass", test_forward_pass(mode)),
            ("Loss Calculation", test_loss(mode)),
            ("Train Step", test_train_step(mode)),
            ("Convergence", test_convergence(mode))
        ]
        for name, (passed, err) in results:
            print(f"[{'PASS' if passed else 'FAIL'}] {name:20} | Error: {err:.2e}")
