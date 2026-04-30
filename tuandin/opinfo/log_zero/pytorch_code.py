"""Tests `torch.log(0)` — both PT and JAX return -inf in float32."""
import numpy as np
import torch


def make_inputs():
    return {"x": np.array([0.0, 1.0, np.e, 0.5], dtype=np.float32)}


def compute(inputs):
    x = torch.from_numpy(inputs["x"])
    return {"out": torch.log(x).numpy()}
