# test_equivalence.py
import numpy as np
import torch
import jax.numpy as jnp

from pytorch_refactored import SimpleModel, make_model, make_criterion, make_optimizer
from pytorch_refactored import train_model as pt_train_model

from jax_code_fixed import model_apply as jax_model
from jax_code_fixed import loss_fn
from jax_code_fixed import train_step
from jax_code_fixed import train_model as jax_train_model


X_np = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(100, 1)
y_np = (3.0 * X_np + 2.0).astype(np.float32)
W_np = np.array([[0.25]], dtype=np.float32)
B_np = np.array([0.10], dtype=np.float32)
X_test_np = np.array([[0.5], [1.0], [1.5]], dtype=np.float32)


def inject_pytorch_params(model: SimpleModel, w_np: np.ndarray, b_np: np.ndarray) -> None:
    with torch.no_grad():
        model.fc.weight.data.copy_(torch.from_numpy(w_np).to(model.fc.weight.dtype))
        model.fc.bias.data.copy_(torch.from_numpy(b_np).to(model.fc.bias.dtype))


def make_jax_params(w_np: np.ndarray, b_np: np.ndarray):
    return {
        "w": jnp.array(w_np, dtype=jnp.float32),
        "b": jnp.array(b_np, dtype=jnp.float32),
    }


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _assert_close(name: str, a: np.ndarray, b: np.ndarray, tol: float) -> None:
    diff = _max_abs_diff(a, b)
    if diff > tol:
        raise AssertionError(f"{name} failed: max_abs_diff={diff} > tol={tol}\nA={a}\nB={b}")


def test_forward_pass(tol: float = 1e-5) -> None:
    pt_model = make_model()
    inject_pytorch_params(pt_model, W_np, B_np)

    X_t = torch.from_numpy(X_np)
    with torch.no_grad():
        pt_out = pt_model(X_t).detach().cpu().numpy()

    jax_params = make_jax_params(W_np, B_np)
    jax_out = np.array(jax_model(jax_params, jnp.array(X_np, dtype=jnp.float32)))

    _assert_close("forward_pass", pt_out, jax_out, tol)


def test_loss(tol: float = 1e-5) -> None:
    pt_model = make_model()
    inject_pytorch_params(pt_model, W_np, B_np)
    criterion = make_criterion()

    X_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(y_np)

    with torch.no_grad():
        pt_loss = float(criterion(pt_model(X_t), y_t).item())

    jax_params = make_jax_params(W_np, B_np)
    jax_loss = float(loss_fn(jax_params, jnp.array(X_np, dtype=jnp.float32), jnp.array(y_np, dtype=jnp.float32)))

    _assert_close("loss", np.array([pt_loss], dtype=np.float32), np.array([jax_loss], dtype=np.float32), tol)


def test_train_step(tol: float = 1e-5) -> None:
    lr = 0.01

    pt_model = make_model()
    inject_pytorch_params(pt_model, W_np, B_np)
    criterion = make_criterion()
    optimizer = make_optimizer(pt_model)

    X_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(y_np)

    optimizer.zero_grad()
    pt_preds = pt_model(X_t)
    pt_loss_t = criterion(pt_preds, y_t)
    pt_loss = float(pt_loss_t.item())
    pt_loss_t.backward()
    optimizer.step()

    pt_w_new = pt_model.fc.weight.detach().cpu().numpy()
    pt_b_new = pt_model.fc.bias.detach().cpu().numpy()

    jax_params = make_jax_params(W_np, B_np)
    jax_new_params, jax_loss = train_step(
        jax_params,
        jnp.array(X_np, dtype=jnp.float32),
        jnp.array(y_np, dtype=jnp.float32),
        lr,
    )

    jax_w_new = np.array(jax_new_params["w"])
    jax_b_new = np.array(jax_new_params["b"])
    jax_loss = float(jax_loss)

    _assert_close("train_step_loss", np.array([pt_loss], dtype=np.float32), np.array([jax_loss], dtype=np.float32), tol)
    _assert_close("train_step_w", pt_w_new, jax_w_new, tol)
    _assert_close("train_step_b", pt_b_new, jax_b_new, tol)


def test_convergence(tol: float = 1e-2) -> None:
    num_epochs = 100
    lr = 0.01

    pt_model = make_model()
    inject_pytorch_params(pt_model, W_np, B_np)
    criterion = make_criterion()
    optimizer = make_optimizer(pt_model)

    X_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(y_np)

    pt_train_model(X_t, y_t, pt_model, optimizer, criterion, num_epochs)

    pt_w = pt_model.fc.weight.detach().cpu().numpy()
    pt_b = pt_model.fc.bias.detach().cpu().numpy()

    jax_params = make_jax_params(W_np, B_np)
    jax_params = jax_train_model(
        jnp.array(X_np, dtype=jnp.float32),
        jnp.array(y_np, dtype=jnp.float32),
        jax_params,
        num_epochs,
        lr,
    )

    jax_w = np.array(jax_params["w"])
    jax_b = np.array(jax_params["b"])

    _assert_close("convergence_w", pt_w, jax_w, tol)
    _assert_close("convergence_b", pt_b, jax_b, tol)

    with torch.no_grad():
        pt_preds = pt_model(torch.from_numpy(X_test_np)).detach().cpu().numpy()

    jax_preds = np.array(jax_model(jax_params, jnp.array(X_test_np, dtype=jnp.float32)))
    _assert_close("convergence_predictions", pt_preds, jax_preds, tol)


def _run_test(fn, tol: float) -> bool:
    try:
        fn(tol=tol)
        print(f"PASS {fn.__name__} (tol={tol})")
        return True
    except Exception as e:
        print(f"FAIL: {fn.__name__} (tol={tol}) -> {e}")
        return False


if __name__ == "__main__":
    ok = True
    ok &= _run_test(test_forward_pass, 1e-5)
    ok &= _run_test(test_loss, 1e-5)
    ok &= _run_test(test_train_step, 1e-5)
    ok &= _run_test(test_convergence, 1e-2)
    raise SystemExit(0 if ok else 1)