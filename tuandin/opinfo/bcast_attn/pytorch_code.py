"""Tests an attention-style broadcast: (B, 1, S, D) * (B, H, 1, D) ->
broadcasted to (B, H, S, D). This is the per-head Q-replicated multiply
common in MHA when Q is single-head and K/V are multi-head.
"""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    return {
        "q": rng.standard_normal((2, 1, 5, 8)).astype(np.float32),
        "k": rng.standard_normal((2, 4, 1, 8)).astype(np.float32),
    }


def compute(inputs):
    q = torch.from_numpy(inputs["q"])
    k = torch.from_numpy(inputs["k"])
    out = q * k    # broadcast to (2, 4, 5, 8)
    return {"out": out.numpy()}
