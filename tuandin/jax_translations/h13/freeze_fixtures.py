"""Freeze fixtures for h13: flash-attention output matches vanilla SDPA.

PyTorch reference is computed via vanilla attention in CPU PyTorch (the Triton
kernel needs CUDA). Outputs: O (attention output) and L (row logsumexp).
"""
import math
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    B, N_q, N_k, D = 1, 32, 64, 128
    return {
        "Q": rng.standard_normal((B, N_q, D)).astype(np.float32),
        "K": rng.standard_normal((B, N_k, D)).astype(np.float32),
        "V": rng.standard_normal((B, N_k, D)).astype(np.float32),
    }


def pytorch_reference(inputs):
    Q = torch.from_numpy(inputs["Q"]); K = torch.from_numpy(inputs["K"]); V = torch.from_numpy(inputs["V"])
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    scores = (Q @ K.transpose(-2, -1)) * scale
    O = torch.softmax(scores, dim=-1) @ V
    L = torch.logsumexp(scores, dim=-1)
    return {"O": O.numpy(), "L": L.numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("h13: fixtures written")
