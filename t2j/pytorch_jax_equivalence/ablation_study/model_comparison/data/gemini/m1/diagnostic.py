"""
Diagnostic script to pinpoint where PyTorch and JAX diverge in train step.
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


print("=" * 60)
print("STEP 0: Verify parameters match after injection")
print("=" * 60)

pt_m = PtCustomModel(DIM, HIDDEN)
inject_pytorch_params(pt_m)

jax_m = JaxCustomModel(DIM, HIDDEN)
jax_p = make_jax_params(jax_m)
jax_p_dict = unfreeze(jax_p)

# Compare each parameter
param_map = {
    'Wxi': ('Wxi', False),
    'Whi': ('Whi', False),
    'bi': ('bi', False),
    'Wxf': ('Wxf', False),
    'Whf': ('Whf', False),
    'bf': ('bf', False),
    'Wxo': ('Wxo', False),
    'Who': ('Who', False),
    'bo': ('bo', False),
    'Wxc': ('Wxc', False),
    'Whc': ('Whc', False),
    'bc': ('bc', False),
    'fc.weight': ('Dense_0/kernel', True),  # needs transpose
    'fc.bias': ('Dense_0/bias', False),
}

all_match = True
for pt_name, (jax_path, needs_transpose) in param_map.items():
    # Get PyTorch param
    parts = pt_name.split('.')
    pt_val = pt_m
    for part in parts:
        pt_val = getattr(pt_val, part)
    pt_np = pt_val.data.numpy()

    # Get JAX param
    jax_parts = jax_path.split('/')
    jax_val = jax_p_dict
    for part in jax_parts:
        jax_val = jax_val[part]
    jax_np = np.array(jax_val)

    if needs_transpose:
        jax_np = jax_np.T

    diff = np.abs(pt_np - jax_np).max()
    status = "OK" if diff < 1e-7 else "MISMATCH"
    if diff >= 1e-7:
        all_match = False
    print(f"  {pt_name:15s} vs {jax_path:20s} | pt_shape={pt_np.shape} jax_shape={jax_np.shape if not needs_transpose else '(transposed)'} | diff={diff:.2e} [{status}]")

print(f"\nAll params match: {all_match}")

print("\n" + "=" * 60)
print("STEP 1: Compare forward pass (no grad)")
print("=" * 60)

pt_m.eval()
with torch.no_grad():
    h0 = torch.zeros(BATCH, HIDDEN)
    c0 = torch.zeros(BATCH, HIDDEN)
    pt_pred_full, (pt_H, pt_C) = pt_m(torch.from_numpy(X_np), (h0, c0))
    pt_pred = pt_pred_full[:, -1, :]

zero_HC = (jnp.zeros((BATCH, HIDDEN)), jnp.zeros((BATCH, HIDDEN)))
jax_pred_full, (jax_H, jax_C) = jax_m.apply({'params': jax_p}, jnp.array(X_np), jax.random.PRNGKey(0), H_C=zero_HC)
jax_pred = jax_pred_full[:, -1, :]

print(f"PT pred:  {pt_pred.numpy().flatten()}")
print(f"JAX pred: {np.array(jax_pred).flatten()}")
print(f"Pred diff: {np.abs(pt_pred.numpy() - np.array(jax_pred)).max():.2e}")

# Compare hidden states at each timestep
print("\nCompare hidden state H at each timestep:")
pt_m2 = PtCustomModel(DIM, HIDDEN)
inject_pytorch_params(pt_m2)
pt_m2.eval()

# Manual step-by-step for PyTorch
with torch.no_grad():
    H_pt = torch.zeros(BATCH, HIDDEN)
    C_pt = torch.zeros(BATCH, HIDDEN)
    for t in range(SEQ):
        X_t = torch.from_numpy(X_np[:, t, :])
        I_t = torch.sigmoid(X_t @ pt_m2.Wxi + H_pt @ pt_m2.Whi + pt_m2.bi)
        F_t = torch.sigmoid(X_t @ pt_m2.Wxf + H_pt @ pt_m2.Whf + pt_m2.bf)
        O_t = torch.sigmoid(X_t @ pt_m2.Wxo + H_pt @ pt_m2.Who + pt_m2.bo)
        C_tilde = torch.tanh(X_t @ pt_m2.Wxc + H_pt @ pt_m2.Whc + pt_m2.bc)
        C_pt = F_t * C_pt + I_t * C_tilde
        H_pt = O_t * torch.tanh(C_pt)
        print(f"  t={t}: PT H_sum={H_pt.sum().item():.8f}")

print("\n" + "=" * 60)
print("STEP 2: Forward with grad (train mode)")
print("=" * 60)

# Fresh models
pt_m3 = PtCustomModel(DIM, HIDDEN)
inject_pytorch_params(pt_m3)
pt_m3.train()

pt_res = pt_m3(torch.from_numpy(X_np), (torch.zeros(BATCH, HIDDEN), torch.zeros(BATCH, HIDDEN)))
pt_pred3 = pt_res[0][:, -1, :]
pt_loss3 = nn.MSELoss()(pt_pred3, torch.from_numpy(y_np))

jax_m2 = JaxCustomModel(DIM, HIDDEN)
jax_p2 = make_jax_params(jax_m2)

def loss_fn(p):
    H_C = (jnp.zeros((BATCH, HIDDEN)), jnp.zeros((BATCH, HIDDEN)))
    pred, _ = jax_m2.apply({'params': p}, jnp.array(X_np), jax.random.PRNGKey(0), H_C=H_C)
    return jnp.mean((pred[:, -1, :] - jnp.array(y_np)) ** 2), pred[:, -1, :]

(jax_loss2, jax_pred2), grads = jax.value_and_grad(loss_fn, has_aux=True)(jax_p2)

print(f"PT pred (train):  {pt_pred3.detach().numpy().flatten()}")
print(f"JAX pred (grad):  {np.array(jax_pred2).flatten()}")
print(f"Pred diff: {np.abs(pt_pred3.detach().numpy() - np.array(jax_pred2)).max():.2e}")
print(f"PT loss:  {pt_loss3.item():.10f}")
print(f"JAX loss: {float(jax_loss2):.10f}")
print(f"Loss diff: {abs(pt_loss3.item() - float(jax_loss2)):.2e}")

# Compare gradients
print("\nGradient comparison:")
pt_loss3.backward()

grad_pairs = [
    ('Wxi', 'Wxi', False),
    ('Whi', 'Whi', False),
    ('bi', 'bi', False),
    ('Wxf', 'Wxf', False),
    ('Whf', 'Whf', False),
    ('bf', 'bf', False),
    ('Wxo', 'Wxo', False),
    ('Who', 'Who', False),
    ('bo', 'bo', False),
    ('Wxc', 'Wxc', False),
    ('Whc', 'Whc', False),
    ('bc', 'bc', False),
]

grads_dict = unfreeze(grads)

for pt_name, jax_name, transpose in grad_pairs:
    pt_grad = getattr(pt_m3, pt_name).grad.numpy()
    jax_grad = np.array(grads_dict[jax_name])
    if transpose:
        jax_grad = jax_grad.T
    diff = np.abs(pt_grad - jax_grad).max()
    print(f"  {pt_name:10s} grad_diff={diff:.2e}  pt_norm={np.linalg.norm(pt_grad):.6f}  jax_norm={np.linalg.norm(jax_grad):.6f}")

# FC layer
pt_fc_w_grad = pt_m3.fc.weight.grad.numpy()
jax_fc_w_grad = np.array(grads_dict['Dense_0']['kernel']).T
diff_fc_w = np.abs(pt_fc_w_grad - jax_fc_w_grad).max()
print(f"  {'fc.weight':10s} grad_diff={diff_fc_w:.2e}")

pt_fc_b_grad = pt_m3.fc.bias.grad.numpy()
jax_fc_b_grad = np.array(grads_dict['Dense_0']['bias'])
diff_fc_b = np.abs(pt_fc_b_grad - jax_fc_b_grad).max()
print(f"  {'fc.bias':10s} grad_diff={diff_fc_b:.2e}")