"""Freeze fixtures for m11: masked-mean + cosine-similarity formulas.

We do NOT freeze SmolLM2-135M's outputs (270 MB download). The contract is
the formula: given last hidden states + attention mask, compute the
sentence embedding (masked mean), then the cosine similarity vs a keyword
embedding.
"""
import numpy as np
import torch
import torch.nn.functional as F


def make_inputs():
    rng = np.random.default_rng(0)
    B, S, H = 4, 7, 16
    return {
        "last_hidden":     rng.standard_normal((B, S, H)).astype(np.float32),
        "attention_mask": (rng.standard_normal((B, S)) > -0.5).astype(np.int32),
        "keyword_embed":  rng.standard_normal((1, H)).astype(np.float32),
    }


def pytorch_reference(inputs):
    last = torch.from_numpy(inputs["last_hidden"])
    mask = torch.from_numpy(inputs["attention_mask"])
    kw = torch.from_numpy(inputs["keyword_embed"])
    expanded = mask.unsqueeze(-1).float()
    summed = (last * expanded).sum(dim=1)
    counts = expanded.sum(dim=1).clamp(min=1e-9)
    sentence_embed = summed / counts
    cos_sim = F.cosine_similarity(sentence_embed, kw)
    return {
        "sentence_embed": sentence_embed.numpy(),
        "cos_sim":        cos_sim.numpy(),
    }


if __name__ == "__main__":
    inputs = make_inputs()
    np.savez("inputs.npz", **inputs)
    np.savez("expected.npz", **pytorch_reference(inputs))
    print("m11: fixtures written")
