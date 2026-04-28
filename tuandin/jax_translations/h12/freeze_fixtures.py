"""Freeze fixtures for h12: SmolLM building blocks (RMSNorm, SwiGLU MLP, rotate_half).

Full LM forward isn't tested (would require porting checkpoint into Flax param
tree). Each block is a small, deterministic spot-check.
"""
import numpy as np
import torch
import torch.nn as nn


def make_inputs():
    rng = np.random.default_rng(0)
    H, I = 32, 64
    return {
        "x":      rng.standard_normal((2, 3, H)).astype(np.float32),
        "weight": rng.uniform(0.5, 1.5, (H,)).astype(np.float32),  # RMSNorm scale
        "W_gate": (rng.standard_normal((I, H)) * 0.1).astype(np.float32),
        "W_up":   (rng.standard_normal((I, H)) * 0.1).astype(np.float32),
        "W_down": (rng.standard_normal((H, I)) * 0.1).astype(np.float32),
        "z":      rng.standard_normal((2, 3, H)).astype(np.float32),  # rotate_half input
        "eps":    np.array(1e-5, dtype=np.float32),
    }


def pytorch_reference(inputs):
    H = inputs["x"].shape[-1]; I = inputs["W_gate"].shape[0]
    eps = float(inputs["eps"])

    # RMSNorm
    var = torch.from_numpy(inputs["x"]).pow(2).mean(-1, keepdim=True)
    rms = torch.from_numpy(inputs["x"]) * torch.rsqrt(var + eps)
    rms_out = (torch.from_numpy(inputs["weight"]) * rms).numpy()

    # SwiGLU MLP
    W_gate = torch.from_numpy(inputs["W_gate"])
    W_up = torch.from_numpy(inputs["W_up"])
    W_down = torch.from_numpy(inputs["W_down"])
    x = torch.from_numpy(inputs["x"])
    swish = torch.nn.SiLU()(x @ W_gate.T)
    mlp_out = (swish * (x @ W_up.T)) @ W_down.T

    # rotate_half
    z = torch.from_numpy(inputs["z"])
    half = z.shape[-1] // 2
    rh = torch.cat([-z[..., half:], z[..., :half]], dim=-1)

    return {
        "rms_norm":    rms_out,
        "swiglu_mlp":  mlp_out.numpy(),
        "rotate_half": rh.numpy(),
    }


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("h12: fixtures written")
