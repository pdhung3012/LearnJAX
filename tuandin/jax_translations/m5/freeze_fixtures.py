"""Freeze fixtures for m5: SimpleRNN single step (h_t = tanh(W_ih x + b_ih + W_hh h + b_hh))."""
import numpy as np
import torch
import torch.nn as nn


def make_inputs():
    rng = np.random.default_rng(0)
    in_dim, hidden = 1, 8
    return {
        "W_ih": (rng.standard_normal((hidden, in_dim)) * 0.3).astype(np.float32),
        "b_ih": (rng.standard_normal((hidden,)) * 0.1).astype(np.float32),
        "W_hh": (rng.standard_normal((hidden, hidden)) * 0.3).astype(np.float32),
        "b_hh": (rng.standard_normal((hidden,)) * 0.1).astype(np.float32),
        "x":    rng.standard_normal((1, 1, in_dim)).astype(np.float32),  # (B, T=1, in_dim)
        "h0":   rng.standard_normal((1, 1, hidden)).astype(np.float32),  # (num_layers, B, hidden)
    }


def pytorch_reference(inputs):
    in_dim = inputs["x"].shape[-1]
    hidden = inputs["W_ih"].shape[0]
    rnn = nn.RNN(in_dim, hidden, num_layers=1, batch_first=True, nonlinearity="tanh")
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(torch.from_numpy(inputs["W_ih"]))
        rnn.bias_ih_l0.copy_(torch.from_numpy(inputs["b_ih"]))
        rnn.weight_hh_l0.copy_(torch.from_numpy(inputs["W_hh"]))
        rnn.bias_hh_l0.copy_(torch.from_numpy(inputs["b_hh"]))
    out, _ = rnn(torch.from_numpy(inputs["x"]), torch.from_numpy(inputs["h0"]))
    return {"out": out.detach().numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("m5: fixtures written")
