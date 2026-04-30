"""Tests bool * float — bool acts as 0/1 mask. The dtype promotion result
should be float32 in both PT and JAX (default). Common LLM bug: keeping
bool in a multiplication, expecting it to act as a logical AND.
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "b": (rng.standard_normal((4, 5)) > 0),
        "f": rng.standard_normal((4, 5)).astype(np.float32),
    }


def compute(inputs):
    b = torch.from_numpy(inputs["b"])
    f = torch.from_numpy(inputs["f"])
    out = b * f
    return {"out": out.numpy()}
