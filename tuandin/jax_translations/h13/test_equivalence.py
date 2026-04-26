"""h13 equivalence test: streaming-softmax flash attention vs vanilla SDPA.

The PyTorch Triton kernel needs CUDA, so we can't run it on this CPU. We
compare the JAX streaming flash implementation against vanilla attention
(both in JAX) with the same Q/K/V — they must agree to within float32
softmax precision.
"""
import math
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close

sys.path.insert(0, str(Path(__file__).parent))
from jax_code import flash_attention_forward


def main():
    rng = np.random.default_rng(0)
    B, N_q, N_k, D = 1, 32, 64, 128
    Q = rng.standard_normal((B, N_q, D)).astype(np.float32)
    K = rng.standard_normal((B, N_k, D)).astype(np.float32)
    V = rng.standard_normal((B, N_k, D)).astype(np.float32)
    Qj, Kj, Vj = jnp.asarray(Q), jnp.asarray(K), jnp.asarray(V)

    O_flash, L_flash = jax.jit(flash_attention_forward)(Qj, Kj, Vj)

    scale = 1.0 / math.sqrt(D)
    scores = (Qj @ jnp.swapaxes(Kj, -2, -1)) * scale
    O_ref = jax.nn.softmax(scores, axis=-1) @ Vj
    L_ref = jax.scipy.special.logsumexp(scores, axis=-1)

    assert_close(np.asarray(O_flash), np.asarray(O_ref), atol=1e-3, rtol=1e-3,
                 name="flash_O_vs_vanilla_O")
    assert_close(np.asarray(L_flash), np.asarray(L_ref), atol=1e-3, rtol=1e-3,
                 name="flash_L_vs_vanilla_L")
    print("[h13] PASS")


if __name__ == "__main__":
    main()
