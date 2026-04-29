"""Tests gradient through `torch.where` when one branch contains non-finite
values that aren't selected. Subtle: the unselected branch is still
EVALUATED for the gradient computation, so its non-finite values can
poison the gradient even though the forward output is finite.

The fix in both PT and JAX: ensure the unselected branch is finite-safe
in the function being differentiated. For this case we test the
"obviously safe" pattern where the unselected branch IS finite.
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "x":      rng.standard_normal((6,)).astype(np.float32),
        "thresh": np.array(0.0, dtype=np.float32),
    }


def compute(inputs):
    x = torch.tensor(inputs["x"], requires_grad=True)
    thresh = float(inputs["thresh"])
    # Finite-safe: both branches are finite for any input.
    safe = torch.where(x > thresh, x ** 2, x)
    loss = safe.sum()
    loss.backward()
    return {"grad": x.grad.numpy()}
