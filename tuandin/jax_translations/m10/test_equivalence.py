"""m10 equivalence test: Rotary positional embeddings.

We test the elementwise rotation given identical (q, k) and matched cos/sin
tables. The PyTorch and JAX versions use different layouts for cos/sin
(`(S,1,1,D)` vs `(1,S,1,D)`); we materialize the *same* underlying values and
broadcast them appropriately so the rotated outputs are bit-equivalent.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close

sys.path.insert(0, str(Path(__file__).parent))
from pytorch_code import rotate_half as rot_half_pt, apply_rotary_pos_emb as apply_pt
from jax_code import rotate_half as rot_half_jx, apply_rotary_pos_emb as apply_jx, make_rotary


def main():
    rng = np.random.default_rng(0)
    seq_len, batch, n_heads, head_dim = 8, 2, 4, 16

    # PyTorch layout: (S, B, H, D); JAX layout: (B, S, H, D).
    q_sbhd = rng.standard_normal((seq_len, batch, n_heads, head_dim)).astype(np.float32)
    k_sbhd = rng.standard_normal((seq_len, batch, n_heads, head_dim)).astype(np.float32)
    q_bshd = np.transpose(q_sbhd, (1, 0, 2, 3)).copy()
    k_bshd = np.transpose(k_sbhd, (1, 0, 2, 3)).copy()

    # Build cos/sin in JAX (returns shape (1, S, 1, D)).
    cos_jx, sin_jx = make_rotary(head_dim, seq_len)
    # Build matching PyTorch cos/sin (shape (S, 1, 1, D)).
    cos_pt = torch.from_numpy(np.transpose(np.asarray(cos_jx), (1, 0, 2, 3)))  # (S,1,1,D)
    sin_pt = torch.from_numpy(np.transpose(np.asarray(sin_jx), (1, 0, 2, 3)))

    q_rot_pt, k_rot_pt = apply_pt(torch.from_numpy(q_sbhd), torch.from_numpy(k_sbhd),
                                   cos_pt, sin_pt)
    q_rot_jx, k_rot_jx = apply_jx(jnp.asarray(q_bshd), jnp.asarray(k_bshd),
                                   cos_jx, sin_jx)

    # Re-align JAX (B,S,H,D) -> (S,B,H,D) for comparison.
    q_jx_sbhd = np.transpose(np.asarray(q_rot_jx), (1, 0, 2, 3))
    k_jx_sbhd = np.transpose(np.asarray(k_rot_jx), (1, 0, 2, 3))
    assert_close(q_rot_pt.numpy(), q_jx_sbhd, atol=1e-6, name="q_rot")
    assert_close(k_rot_pt.numpy(), k_jx_sbhd, atol=1e-6, name="k_rot")
    print("[m10] PASS")


if __name__ == "__main__":
    main()
