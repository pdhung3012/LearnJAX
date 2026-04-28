"""Freeze fixtures for e8: RMSNorm with caller-supplied scale."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from pytorch_code import RMSNorm


def make_inputs():
    rng = np.random.default_rng(0)
    dim = 8
    return {
        "x":     (rng.standard_normal((4, dim)) * 2).astype(np.float32),
        "scale": rng.uniform(0.5, 1.5, (dim,)).astype(np.float32),
        "eps":   np.array(1e-8, dtype=np.float32),
    }


def pytorch_reference(inputs):
    dim = inputs["x"].shape[-1]
    m = RMSNorm(dim=dim, eps=float(inputs["eps"]))
    with torch.no_grad():
        m.scale.copy_(torch.from_numpy(inputs["scale"]))
    return {"output": m(torch.from_numpy(inputs["x"])).detach().numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("e8: fixtures written")
