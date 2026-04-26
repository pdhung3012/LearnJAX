"""m1 equivalence test: custom hand-rolled LSTM step with shared weights.

Since PyTorch and JAX have different RNGs and the original re-samples (H, C)
inside forward, we test only the deterministic per-step gate computation:
given the same Wxi/Whi/bi/... and same H, C, X_t, both produce identical
new H and C.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close


def lstm_step_pt(W, X_t, H_prev, C_prev):
    I = torch.sigmoid(X_t @ W["Wxi"] + H_prev @ W["Whi"] + W["bi"])
    F_ = torch.sigmoid(X_t @ W["Wxf"] + H_prev @ W["Whf"] + W["bf"])
    O = torch.sigmoid(X_t @ W["Wxo"] + H_prev @ W["Who"] + W["bo"])
    C_tilde = torch.tanh(X_t @ W["Wxc"] + H_prev @ W["Whc"] + W["bc"])
    C = F_ * C_prev + I * C_tilde
    H = O * torch.tanh(C)
    return H, C


def lstm_step_jx(W, X_t, H_prev, C_prev):
    import jax
    I = jax.nn.sigmoid(X_t @ W["Wxi"] + H_prev @ W["Whi"] + W["bi"])
    F_ = jax.nn.sigmoid(X_t @ W["Wxf"] + H_prev @ W["Whf"] + W["bf"])
    O = jax.nn.sigmoid(X_t @ W["Wxo"] + H_prev @ W["Who"] + W["bo"])
    C_tilde = jnp.tanh(X_t @ W["Wxc"] + H_prev @ W["Whc"] + W["bc"])
    C = F_ * C_prev + I * C_tilde
    H = O * jnp.tanh(C)
    return H, C


def main():
    rng = np.random.default_rng(0)
    in_dim, hidden = 3, 4
    keys = ["Wxi", "Whi", "bi", "Wxf", "Whf", "bf",
            "Wxo", "Who", "bo", "Wxc", "Whc", "bc"]
    shapes = {
        "Wxi": (in_dim, hidden), "Whi": (hidden, hidden), "bi": (hidden,),
        "Wxf": (in_dim, hidden), "Whf": (hidden, hidden), "bf": (hidden,),
        "Wxo": (in_dim, hidden), "Who": (hidden, hidden), "bo": (hidden,),
        "Wxc": (in_dim, hidden), "Whc": (hidden, hidden), "bc": (hidden,),
    }
    W_np = {k: rng.standard_normal(shapes[k]).astype(np.float32) * 0.3 for k in keys}
    W_pt = {k: torch.from_numpy(v) for k, v in W_np.items()}
    W_jx = {k: jnp.asarray(v) for k, v in W_np.items()}

    X_t = rng.standard_normal((2, in_dim)).astype(np.float32)
    H = rng.standard_normal((2, hidden)).astype(np.float32)
    C = rng.standard_normal((2, hidden)).astype(np.float32)

    H_pt, C_pt = lstm_step_pt(W_pt, torch.from_numpy(X_t),
                              torch.from_numpy(H), torch.from_numpy(C))
    H_jx, C_jx = lstm_step_jx(W_jx, jnp.asarray(X_t), jnp.asarray(H), jnp.asarray(C))
    assert_close(H_pt.numpy(), np.asarray(H_jx), atol=1e-5, name="H_after_step")
    assert_close(C_pt.numpy(), np.asarray(C_jx), atol=1e-5, name="C_after_step")
    print("[m1] PASS")


if __name__ == "__main__":
    main()
