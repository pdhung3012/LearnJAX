"""
Diagnostic 2: Compare updated parameter values after one SGD step.
Tests gradients, manual SGD (no optax), and optax SGD separately.
"""
import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import unfreeze, freeze

from pytorch_refactored import CustomLSTMModel as PtCustomModel
from jax_code_fixed import CustomLSTMModel as JaxCustomModel

BATCH, SEQ, DIM, HIDDEN = 2, 5, 1, 50
X_np = np.linspace(-1, 1, BATCH * SEQ * DIM).reshape(BATCH, SEQ, DIM).astype(np.float32)
y_np = np.array([[0.5], [0.8]], dtype=np.float32)
LR = 0.01

W_np = np.ones((DIM, HIDDEN), dtype=np.float32) * 0.1
H_np = np.ones((HIDDEN, HIDDEN), dtype=np.float32) * 0.05
B_np = np.zeros((HIDDEN,), dtype=np.float32)
W_fc_np = np.ones((HIDDEN, 1), dtype=np.float32) * 0.2
B_fc_np = np.array([0.1], dtype=np.float32)


def inject_pytorch_params(model):
    with torch.no_grad():
        for gate in ['i', 'f', 'o', 'c']:
            getattr(model, f'Wx{gate}').data = torch.from_numpy(W_np.copy())
            getattr(model, f'Wh{gate}').data = torch.from_numpy(H_np.copy())
            getattr(model, f'b{gate}').data = torch.from_numpy(B_np.copy())
        model.fc.weight.data = torch.from_numpy(W_fc_np.T.copy())
        model.fc.bias.data = torch.from_numpy(B_fc_np.copy())


def make_jax_params(model):
    init_key = jax.random.PRNGKey(0)
    variables = model.init(init_key, jnp.array(X_np), init_key)
    params = unfreeze(variables['params'])
    for gate in ['i', 'f', 'o', 'c']:
        params[f'Wx{gate}'] = jnp.array(W_np)
        params[f'Wh{gate}'] = jnp.array(H_np)
        params[f'b{gate}'] = jnp.array(B_np)
    params['Dense_0'] = {
        'kernel': jnp.array(W_fc_np),
        'bias': jnp.array(B_fc_np),
    }
    return freeze(params)


# ============================================================
print("STEP 1: PyTorch - compute gradients and update")
print("=" * 60)

pt_m = PtCustomModel(DIM, HIDDEN)
inject_pytorch_params(pt_m)
pt_opt = torch.optim.SGD(pt_m.parameters(), lr=LR)

pt_m.train()
pt_out = pt_m(torch.from_numpy(X_np), (torch.zeros(BATCH, HIDDEN), torch.zeros(BATCH, HIDDEN)))[0][:, -1, :]
pt_loss = nn.MSELoss()(pt_out, torch.from_numpy(y_np))
print(f"PT loss: {pt_loss.item():.10f}")

pt_opt.zero_grad()
pt_loss.backward()

# Grab gradients and pre-update values for Wxi
pt_wxi_before = pt_m.Wxi.data.clone().numpy()
pt_wxi_grad = pt_m.Wxi.grad.numpy().copy()
print(f"PT Wxi before: sum={pt_wxi_before.sum():.10f}")
print(f"PT Wxi grad:   sum={pt_wxi_grad.sum():.10f}")

pt_opt.step()

pt_wxi_after = pt_m.Wxi.data.numpy().copy()
print(f"PT Wxi after:  sum={pt_wxi_after.sum():.10f}")
print(f"PT Wxi expected (before - lr*grad): sum={(pt_wxi_before - LR * pt_wxi_grad).sum():.10f}")

# Verify PT SGD is doing what we think
pt_manual = pt_wxi_before - LR * pt_wxi_grad
pt_sgd_check = np.abs(pt_wxi_after - pt_manual).max()
print(f"PT SGD correctness check (should be ~0): {pt_sgd_check:.2e}")


# ============================================================
print("\nSTEP 2: JAX - compute gradients")
print("=" * 60)

jax_m = JaxCustomModel(DIM, HIDDEN)
jax_p = make_jax_params(jax_m)

def loss_fn(p):
    H_C = (jnp.zeros((BATCH, HIDDEN)), jnp.zeros((BATCH, HIDDEN)))
    pred, _ = jax_m.apply({'params': p}, jnp.array(X_np), jax.random.PRNGKey(0), H_C=H_C)
    return jnp.mean((pred[:, -1, :] - jnp.array(y_np)) ** 2)

jax_loss, grads = jax.value_and_grad(loss_fn)(jax_p)
grads_dict = unfreeze(grads)
jax_p_dict = unfreeze(jax_p)

print(f"JAX loss: {float(jax_loss):.10f}")
print(f"JAX Wxi before: sum={float(jax_p_dict['Wxi'].sum()):.10f}")
print(f"JAX Wxi grad:   sum={float(grads_dict['Wxi'].sum()):.10f}")


# ============================================================
print("\nSTEP 3: Manual SGD in JAX (no optax)")
print("=" * 60)

# Manually do: param = param - lr * grad
manual_new_params = jax.tree.map(lambda p, g: p - LR * g, jax_p, grads)
manual_p_dict = unfreeze(manual_new_params)

jax_wxi_manual = np.array(manual_p_dict['Wxi'])
print(f"JAX Wxi after (manual SGD): sum={jax_wxi_manual.sum():.10f}")
print(f"Manual SGD diff vs PT: {np.abs(pt_wxi_after - jax_wxi_manual).max():.2e}")


# ============================================================
print("\nSTEP 4: optax SGD")
print("=" * 60)

jax_opt = optax.sgd(LR)
opt_state = jax_opt.init(jax_p)
updates, new_opt_state = jax_opt.update(grads, opt_state)

# Check what optax produces as updates
updates_dict = unfreeze(updates)
print(f"optax update for Wxi: sum={float(np.array(updates_dict['Wxi']).sum()):.10f}")
print(f"Expected update (-lr * grad): sum={float((-LR * np.array(grads_dict['Wxi'])).sum()):.10f}")

new_params = optax.apply_updates(jax_p, updates)
new_p_dict = unfreeze(new_params)

jax_wxi_optax = np.array(new_p_dict['Wxi'])
print(f"JAX Wxi after (optax SGD): sum={jax_wxi_optax.sum():.10f}")
print(f"optax SGD diff vs PT: {np.abs(pt_wxi_after - jax_wxi_optax).max():.2e}")


# ============================================================
print("\nSTEP 5: Compare ALL updated params")
print("=" * 60)

# Compare all params after manual JAX SGD vs PyTorch SGD
param_pairs = [
    ('Wxi', 'Wxi', False), ('Whi', 'Whi', False), ('bi', 'bi', False),
    ('Wxf', 'Wxf', False), ('Whf', 'Whf', False), ('bf', 'bf', False),
    ('Wxo', 'Wxo', False), ('Who', 'Who', False), ('bo', 'bo', False),
    ('Wxc', 'Wxc', False), ('Whc', 'Whc', False), ('bc', 'bc', False),
]

for pt_name, jax_name, _ in param_pairs:
    pt_val = getattr(pt_m, pt_name).data.numpy()
    jax_val = np.array(manual_p_dict[jax_name])
    diff = np.abs(pt_val - jax_val).max()
    status = "OK" if diff < 1e-5 else "MISMATCH"
    print(f"  {pt_name:10s} diff={diff:.2e} [{status}]")

# FC layer
pt_fc_w = pt_m.fc.weight.data.numpy()  # (1, 50)
jax_fc_w = np.array(manual_p_dict['Dense_0']['kernel']).T  # transpose to match
diff_fc_w = np.abs(pt_fc_w - jax_fc_w).max()
print(f"  {'fc.weight':10s} diff={diff_fc_w:.2e}")

pt_fc_b = pt_m.fc.bias.data.numpy()
jax_fc_b = np.array(manual_p_dict['Dense_0']['bias'])
diff_fc_b = np.abs(pt_fc_b - jax_fc_b).max()
print(f"  {'fc.bias':10s} diff={diff_fc_b:.2e}")


# ============================================================
print("\nSTEP 6: Forward pass with updated params")
print("=" * 60)

pt_m.eval()
with torch.no_grad():
    pt_out2 = pt_m(torch.from_numpy(X_np),
                    (torch.zeros(BATCH, HIDDEN), torch.zeros(BATCH, HIDDEN)))[0][:, -1, :]

zero_HC = (jnp.zeros((BATCH, HIDDEN)), jnp.zeros((BATCH, HIDDEN)))
jax_out_manual = jax_m.apply({'params': manual_new_params}, jnp.array(X_np),
                              jax.random.PRNGKey(0), H_C=zero_HC)[0][:, -1, :]
jax_out_optax = jax_m.apply({'params': new_params}, jnp.array(X_np),
                             jax.random.PRNGKey(0), H_C=zero_HC)[0][:, -1, :]

print(f"PT output after step:          {pt_out2.numpy().flatten()}")
print(f"JAX output (manual SGD):       {np.array(jax_out_manual).flatten()}")
print(f"JAX output (optax SGD):        {np.array(jax_out_optax).flatten()}")
print(f"Diff (PT vs JAX manual):       {np.abs(pt_out2.numpy() - np.array(jax_out_manual)).max():.2e}")
print(f"Diff (PT vs JAX optax):        {np.abs(pt_out2.numpy() - np.array(jax_out_optax)).max():.2e}")
print(f"Diff (JAX manual vs optax):    {np.abs(np.array(jax_out_manual) - np.array(jax_out_optax)).max():.2e}")
