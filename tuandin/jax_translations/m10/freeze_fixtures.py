"""Freeze fixtures for m10: RoPE applied to (q, k) given matching cos/sin tables.

We use the (B, S, H, D) layout (HF / Llama style) and pass cos/sin shaped
(1, S, 1, D) so the contract is layout-independent.
"""
import numpy as np
import torch


def rotate_half_pt(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def make_inputs():
    rng = np.random.default_rng(0)
    B, S, H, D = 2, 8, 4, 16
    base = 10000.0
    inv_freq = 1.0 / (base ** (np.arange(0, D, 2, dtype=np.float32) / D))
    t = np.arange(S, dtype=np.float32)
    freqs = np.einsum("i,j->ij", t, inv_freq)         # (S, D/2)
    emb = np.concatenate([freqs, freqs], axis=-1)     # (S, D)
    cos = np.cos(emb)[None, :, None, :].astype(np.float32)  # (1, S, 1, D)
    sin = np.sin(emb)[None, :, None, :].astype(np.float32)
    return {
        "q":   rng.standard_normal((B, S, H, D)).astype(np.float32),
        "k":   rng.standard_normal((B, S, H, D)).astype(np.float32),
        "cos": cos,
        "sin": sin,
    }


def pytorch_reference(inputs):
    q = torch.from_numpy(inputs["q"]); k = torch.from_numpy(inputs["k"])
    cos = torch.from_numpy(inputs["cos"]); sin = torch.from_numpy(inputs["sin"])
    q_rot = q * cos + rotate_half_pt(q) * sin
    k_rot = k * cos + rotate_half_pt(k) * sin
    return {"q_rot": q_rot.numpy(), "k_rot": k_rot.numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("m10: fixtures written")
