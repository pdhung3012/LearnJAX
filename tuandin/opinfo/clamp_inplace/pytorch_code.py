"""Tests `Tensor.clamp_(min, max)` — in-place clamp."""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "x": (rng.standard_normal((4, 8)) * 3).astype(np.float32),
        "lo": np.array(-1.0, dtype=np.float32),
        "hi": np.array(1.0, dtype=np.float32),
    }


def compute(inputs):
    x = torch.from_numpy(inputs["x"]).clone()
    x.clamp_(min=float(inputs["lo"]), max=float(inputs["hi"]))
    out = x ** 2     # downstream use
    return {"out": out.numpy()}
