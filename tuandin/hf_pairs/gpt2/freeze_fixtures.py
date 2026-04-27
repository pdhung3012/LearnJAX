"""Freeze fixtures for the GPT-2 case."""
from pathlib import Path
import numpy as np
import torch
from transformers import GPT2Config, GPT2Model

HERE = Path(__file__).parent


def make_inputs():
    return {
        "input_ids":      np.array([[1, 5, 9, 13, 17, 21, 25, 29]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1]],     dtype=np.int64),
    }


def build_pt_model(seed: int = 0) -> GPT2Model:
    config = GPT2Config(
        vocab_size=100, n_positions=32, n_embd=64,
        n_layer=2, n_head=4, n_inner=128,
    )
    torch.manual_seed(seed)
    model = GPT2Model(config)
    model.eval()
    return model


def main():
    inputs = make_inputs()
    model = build_pt_model()
    pt_dir = HERE / "pt_weights"
    model.save_pretrained(pt_dir)

    with torch.no_grad():
        out = model(
            input_ids=torch.from_numpy(inputs["input_ids"]),
            attention_mask=torch.from_numpy(inputs["attention_mask"]),
        )
    expected = {"last_hidden_state": out.last_hidden_state.numpy()}

    np.savez(HERE / "inputs.npz", **inputs)
    np.savez(HERE / "expected.npz", **expected)
    print("gpt2: fixtures written")


if __name__ == "__main__":
    main()
