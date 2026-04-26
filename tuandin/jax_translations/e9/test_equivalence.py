"""e9 equivalence test: sinusoidal positional embedding tables match exactly."""
import sys
from pathlib import Path
import numpy as np
import torch
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close

sys.path.insert(0, str(Path(__file__).parent))
from pytorch_code import SinusoidalPositionalEmbedding as PE_pt
from jax_code import make_sinusoidal_pe, sinusoidal_pe_forward


def main():
    max_seq_len, d_model = 100, 64
    pe_pt = PE_pt(max_seq_len=max_seq_len, d_model=d_model)
    pe_jax = make_sinusoidal_pe(max_seq_len, d_model)

    # PE table itself.
    assert_close(pe_pt.pe.numpy(), np.asarray(pe_jax), atol=1e-6, name="pe_table")

    # forward(x) for an input of seq_len=50.
    seq_len = 50
    dummy_pt = torch.zeros(1, seq_len, d_model)
    dummy_jx = jnp.zeros((1, seq_len, d_model))
    out_pt = pe_pt(dummy_pt).numpy()
    out_jax = np.asarray(sinusoidal_pe_forward(pe_jax, dummy_jx))
    assert_close(out_pt, out_jax, atol=1e-6, name="forward")
    print("[e9] PASS")


if __name__ == "__main__":
    main()
