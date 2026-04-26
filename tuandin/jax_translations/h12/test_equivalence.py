"""h12 equivalence test: SmolLM components — RMSNorm, MLP, RoPE — match
PyTorch reference. We don't compare the full 30-layer LLM forward (expensive
and weight transfer is involved); we test the building blocks.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close

sys.path.insert(0, str(Path(__file__).parent))
from pytorch_code import RMSNorm as RMSNorm_pt, MLP as MLP_pt, rotate_half as rh_pt, apply_rotary_pos_emb as apply_pt
from jax_code import RMSNorm as RMSNorm_jx, MLP as MLP_jx, rotate_half as rh_jx, apply_rotary_pos_emb as apply_jx


def main():
    rng = np.random.default_rng(0)
    H, I = 32, 64

    # ---- RMSNorm ----
    weight = rng.uniform(0.5, 1.5, (H,)).astype(np.float32)
    x = rng.standard_normal((2, 3, H)).astype(np.float32)
    pt = RMSNorm_pt(H, eps=1e-5)
    with torch.no_grad():
        pt.weight.copy_(torch.from_numpy(weight))
    out_pt = pt(torch.from_numpy(x)).detach().numpy()
    jx = RMSNorm_jx(H, eps=1e-5)
    out_jx = np.asarray(jx.apply({"params": {"weight": jnp.asarray(weight)}}, jnp.asarray(x)))
    assert_close(out_pt, out_jx, atol=1e-5, name="RMSNorm")

    # ---- MLP (SwiGLU) ----
    W_gate = rng.standard_normal((I, H)).astype(np.float32) * 0.1
    W_up   = rng.standard_normal((I, H)).astype(np.float32) * 0.1
    W_down = rng.standard_normal((H, I)).astype(np.float32) * 0.1
    pt = MLP_pt(H, I)
    with torch.no_grad():
        pt.W_gate.weight.copy_(torch.from_numpy(W_gate))
        pt.W_up.weight.copy_(torch.from_numpy(W_up))
        pt.W_down.weight.copy_(torch.from_numpy(W_down))
    out_pt = pt(torch.from_numpy(x)).detach().numpy()
    jx = MLP_jx(H, I)
    params = {"params": {
        "W_gate": {"kernel": jnp.asarray(W_gate.T)},
        "W_up":   {"kernel": jnp.asarray(W_up.T)},
        "W_down": {"kernel": jnp.asarray(W_down.T)},
    }}
    out_jx = np.asarray(jx.apply(params, jnp.asarray(x)))
    assert_close(out_pt, out_jx, atol=1e-5, name="SwiGLU_MLP")

    # ---- rotate_half ----
    z = rng.standard_normal((2, 3, H)).astype(np.float32)
    out_pt = rh_pt(torch.from_numpy(z)).numpy()
    out_jx = np.asarray(rh_jx(jnp.asarray(z)))
    assert_close(out_pt, out_jx, atol=1e-7, name="rotate_half")
    print("[h12] PASS")


if __name__ == "__main__":
    main()
