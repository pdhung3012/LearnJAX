"""m5 equivalence test: SimpleRNN forward step with shared weights.

Tests that the elementwise tanh recurrence (h_t = tanh(W_ih x + b_ih + W_hh h + b_hh))
matches between PyTorch nn.RNN and Flax SimpleCell.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _test_utils import assert_close


def main():
    rng = np.random.default_rng(0)
    in_dim, hidden = 1, 8
    W_ih = rng.standard_normal((hidden, in_dim)).astype(np.float32) * 0.3
    b_ih = rng.standard_normal((hidden,)).astype(np.float32) * 0.1
    W_hh = rng.standard_normal((hidden, hidden)).astype(np.float32) * 0.3
    b_hh = rng.standard_normal((hidden,)).astype(np.float32) * 0.1
    x = rng.standard_normal((1, 1, in_dim)).astype(np.float32)
    h0 = rng.standard_normal((1, 1, hidden)).astype(np.float32)

    rnn = nn.RNN(in_dim, hidden, num_layers=1, batch_first=True, nonlinearity="tanh")
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(torch.from_numpy(W_ih))
        rnn.bias_ih_l0.copy_(torch.from_numpy(b_ih))
        rnn.weight_hh_l0.copy_(torch.from_numpy(W_hh))
        rnn.bias_hh_l0.copy_(torch.from_numpy(b_hh))
    out_pt, _ = rnn(torch.from_numpy(x), torch.from_numpy(h0))

    # Hand-roll the JAX equivalent using the same formula.
    h = jnp.asarray(h0[0])
    out_jx_steps = []
    for t in range(x.shape[1]):
        h = jnp.tanh(jnp.asarray(x[:, t, :]) @ jnp.asarray(W_ih.T) + jnp.asarray(b_ih)
                     + h @ jnp.asarray(W_hh.T) + jnp.asarray(b_hh))
        out_jx_steps.append(h)
    out_jx = jnp.stack(out_jx_steps, axis=1)
    assert_close(out_pt.detach().numpy(), np.asarray(out_jx), atol=1e-5,
                 name="rnn_step_forward")
    print("[m5] PASS")


if __name__ == "__main__":
    main()
