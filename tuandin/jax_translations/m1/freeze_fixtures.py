"""Freeze fixtures for m1: custom LSTM single step with shared weights."""
import numpy as np
import torch


def make_inputs():
    rng = np.random.default_rng(0)
    in_dim, hidden, batch = 3, 4, 2
    keys = ["Wxi", "Whi", "bi", "Wxf", "Whf", "bf",
            "Wxo", "Who", "bo", "Wxc", "Whc", "bc"]
    shapes = {
        "Wxi": (in_dim, hidden), "Whi": (hidden, hidden), "bi": (hidden,),
        "Wxf": (in_dim, hidden), "Whf": (hidden, hidden), "bf": (hidden,),
        "Wxo": (in_dim, hidden), "Who": (hidden, hidden), "bo": (hidden,),
        "Wxc": (in_dim, hidden), "Whc": (hidden, hidden), "bc": (hidden,),
    }
    out = {k: (rng.standard_normal(shapes[k]) * 0.3).astype(np.float32) for k in keys}
    out["X_t"] = rng.standard_normal((batch, in_dim)).astype(np.float32)
    out["H_prev"] = rng.standard_normal((batch, hidden)).astype(np.float32)
    out["C_prev"] = rng.standard_normal((batch, hidden)).astype(np.float32)
    return out


def pytorch_reference(inputs):
    W = {k: torch.from_numpy(v) for k, v in inputs.items()
         if k.startswith(("Wx", "Wh", "b"))}
    X_t = torch.from_numpy(inputs["X_t"])
    H_prev = torch.from_numpy(inputs["H_prev"])
    C_prev = torch.from_numpy(inputs["C_prev"])
    I = torch.sigmoid(X_t @ W["Wxi"] + H_prev @ W["Whi"] + W["bi"])
    F_ = torch.sigmoid(X_t @ W["Wxf"] + H_prev @ W["Whf"] + W["bf"])
    O = torch.sigmoid(X_t @ W["Wxo"] + H_prev @ W["Who"] + W["bo"])
    C_tilde = torch.tanh(X_t @ W["Wxc"] + H_prev @ W["Whc"] + W["bc"])
    C = F_ * C_prev + I * C_tilde
    H = O * torch.tanh(C)
    return {"H": H.numpy(), "C": C.numpy()}


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("m1: fixtures written")
