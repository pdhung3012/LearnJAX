"""Freeze fixtures for h3: LayerNorm + FFN(Linear -> ReLU -> Linear) sub-block.

We don't test full MultiheadAttention (requires copying torch's specific
parameter layout into Flax MHA). LayerNorm + FFN are the sub-components that
also live in the encoder layer and are easy to test bit-for-bit.
"""
import numpy as np
import torch
import torch.nn as nn


def make_inputs():
    rng = np.random.default_rng(0)
    B, S, D, FF = 2, 5, 16, 64
    return {
        "x":     rng.standard_normal((B, S, D)).astype(np.float32),
        "gamma": rng.uniform(0.5, 1.5, (D,)).astype(np.float32),
        "beta":  (rng.standard_normal((D,)) * 0.1).astype(np.float32),
        "W1":    (rng.standard_normal((FF, D)) * 0.1).astype(np.float32),
        "b1":    (rng.standard_normal((FF,)) * 0.1).astype(np.float32),
        "W2":    (rng.standard_normal((D, FF)) * 0.1).astype(np.float32),
        "b2":    (rng.standard_normal((D,)) * 0.1).astype(np.float32),
    }


def pytorch_reference(inputs):
    x = torch.from_numpy(inputs["x"])
    D = x.shape[-1]
    ln = nn.LayerNorm(D)
    with torch.no_grad():
        ln.weight.copy_(torch.from_numpy(inputs["gamma"]))
        ln.bias.copy_(torch.from_numpy(inputs["beta"]))
    ln_out = ln(x).detach()
    W1 = torch.from_numpy(inputs["W1"]); b1 = torch.from_numpy(inputs["b1"])
    W2 = torch.from_numpy(inputs["W2"]); b2 = torch.from_numpy(inputs["b2"])
    ffn = (torch.relu(x @ W1.T + b1) @ W2.T + b2).detach()
    return {"layer_norm": ln_out.numpy(), "ffn": ffn.numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("h3: fixtures written")
