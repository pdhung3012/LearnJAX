import numpy as np
import torch
import jax.numpy as jnp
import optax

from pytorch_refactored import (
    DNNModel,
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
# X has 2 features; y ≈ x0 + 2*x1
X_np = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0], [5.0, 3.0], [4.0, 5.0],
                 [6.0, 1.0], [1.0, 6.0], [7.0, 2.0], [3.0, 7.0], [8.0, 4.0]],
                dtype=np.float32)
y_np = np.array([[5.1], [5.2], [10.1], [11.0], [14.2],
                 [8.0], [13.1], [11.1], [17.0], [16.2]], dtype=np.float32)
# Layer 1: Linear(2, 10)
W1_np = np.full((2, 10), 0.1, dtype=np.float32)
B1_np = np.full((10,), 0.01, dtype=np.float32)
# Layer 2: Linear(10, 1)
W2_np = np.full((10, 1), 0.1, dtype=np.float32)
B2_np = np.array([0.01], dtype=np.float32)
X_test_np = np.array([[4.0, 3.0], [7.0, 8.0]], dtype=np.float32)


def inject_pytorch_params(model, w1_np, b1_np, w2_np, b2_np):
    """Inject hardcoded params into PyTorch DNN model."""
    with torch.no_grad():
        model.fc1.weight.data = torch.tensor(w1_np.T)  # PT stores (out, in)
        model.fc1.bias.data = torch.tensor(b1_np)
        model.fc2.weight.data = torch.tensor(w2_np.T)
        model.fc2.bias.data = torch.tensor(b2_np)


def make_jax_params(w1_np, b1_np, w2_np, b2_np):
    """Construct JAX params dict from the same numpy arrays."""
    return {
        'w1': jnp.array(w1_np), 'b1': jnp.array(b1_np),
        'w2': jnp.array(w2_np), 'b2': jnp.array(b2_np),
    }


# ---------- Tier 1 — tolerance 1e-5 ----------

def test_forward_pass():
    model = make_model()
    inject_pytorch_params(model, W1_np, B1_np, W2_np, B2_np)
    with torch.no_grad():
        pt_out = model(torch.tensor(X_test_np)).numpy()

    jax_params = make_jax_params(W1_np, B1_np, W2_np, B2_np)
    jax_out = np.array(jax_model(jax_params, jnp.array(X_test_np)))

    diff = float(np.max(np.abs(pt_out - jax_out)))
    passed = diff < 1e-5
    print(f"test_forward_pass: {'PASS' if passed else 'FAIL'} "
          f"(max diff: {diff:.2e}, tol: 1e-5)")
    return passed


def test_loss():
    model = make_model()
    inject_pytorch_params(model, W1_np, B1_np, W2_np, B2_np)
    criterion = make_criterion()
    with torch.no_grad():
        pt_pred = model(torch.tensor(X_np))
        pt_loss = criterion(pt_pred, torch.tensor(y_np)).item()

    jax_params = make_jax_params(W1_np, B1_np, W2_np, B2_np)
    jax_loss = float(loss_fn(jax_params, jnp.array(X_np), jnp.array(y_np)))

    diff = abs(pt_loss - jax_loss)
    passed = diff < 1e-5
    print(f"test_loss: {'PASS' if passed else 'FAIL'} "
          f"(PT: {pt_loss:.6f}, JAX: {jax_loss:.6f}, diff: {diff:.2e}, tol: 1e-5)")
    return passed


def test_train_step():
    lr = 0.01

    # PyTorch — one Adam step
    model = make_model()
    inject_pytorch_params(model, W1_np, B1_np, W2_np, B2_np)
    optimizer = make_optimizer(model)
    criterion = make_criterion()
    pred = model(torch.tensor(X_np))
    loss = criterion(pred, torch.tensor(y_np))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pt_loss = loss.item()
    pt_w1 = model.fc1.weight.data.numpy()
    pt_b1 = model.fc1.bias.data.numpy()
    pt_w2 = model.fc2.weight.data.numpy()
    pt_b2 = model.fc2.bias.data.numpy()

    # JAX — one optax Adam step
    jax_params = make_jax_params(W1_np, B1_np, W2_np, B2_np)
    optimizer_jax = optax.adam(lr)
    opt_state = optimizer_jax.init(jax_params)
    new_params, opt_state, jax_loss_val = train_step(
        jax_params, jnp.array(X_np), jnp.array(y_np), opt_state, optimizer_jax)
    jax_loss = float(jax_loss_val)
    jax_w1 = np.array(new_params['w1'])
    jax_b1 = np.array(new_params['b1'])
    jax_w2 = np.array(new_params['w2'])
    jax_b2 = np.array(new_params['b2'])

    loss_diff = abs(pt_loss - jax_loss)
    w1_diff = float(np.max(np.abs(pt_w1 - jax_w1.T)))
    b1_diff = float(np.max(np.abs(pt_b1 - jax_b1)))
    w2_diff = float(np.max(np.abs(pt_w2 - jax_w2.T)))
    b2_diff = float(np.max(np.abs(pt_b2 - jax_b2)))
    max_param_diff = max(w1_diff, b1_diff, w2_diff, b2_diff)

    passed = loss_diff < 1e-5 and max_param_diff < 1e-5
    print(f"test_train_step: {'PASS' if passed else 'FAIL'} "
          f"(loss diff: {loss_diff:.2e}, max param diff: {max_param_diff:.2e}, tol: 1e-5)")
    return passed


# ---------- Tier 2 — tolerance 1e-2 ----------

def test_convergence():
    num_epochs = 200
    lr = 0.01

    # PyTorch
    model = make_model()
    inject_pytorch_params(model, W1_np, B1_np, W2_np, B2_np)
    optimizer = make_optimizer(model)
    criterion = make_criterion()
    pt_train_model(torch.tensor(X_np), torch.tensor(y_np),
                   model, optimizer, criterion, num_epochs)
    pt_w1 = model.fc1.weight.data.numpy()
    pt_b1 = model.fc1.bias.data.numpy()

    # JAX
    jax_params = make_jax_params(W1_np, B1_np, W2_np, B2_np)
    optimizer_jax = optax.adam(lr)
    opt_state = optimizer_jax.init(jax_params)
    jax_final, _ = jax_train_model(jnp.array(X_np), jnp.array(y_np),
                                   jax_params, num_epochs,
                                   optimizer=optimizer_jax, opt_state=opt_state)
    jax_w1 = np.array(jax_final['w1'])
    jax_b1 = np.array(jax_final['b1'])

    # Compare first-layer params as representative
    w1_diff = float(np.max(np.abs(pt_w1 - jax_w1.T)))
    b1_diff = float(np.max(np.abs(pt_b1 - jax_b1)))
    max_diff = max(w1_diff, b1_diff)

    passed = max_diff < 1e-2
    print(f"test_convergence: {'PASS' if passed else 'FAIL'} "
          f"(max param diff: {max_diff:.2e}, tol: 1e-2)")
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
