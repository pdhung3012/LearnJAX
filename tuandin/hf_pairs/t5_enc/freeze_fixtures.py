"""Freeze fixtures for the T5 encoder case."""
from pathlib import Path
import numpy as np
import torch
from transformers import T5Config, T5EncoderModel

HERE = Path(__file__).parent


def make_inputs():
    return {
        "input_ids":      np.array([[1, 5, 9, 13, 17, 21, 25, 29]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1]],     dtype=np.int64),
    }


def build_pt_model(seed: int = 0) -> T5EncoderModel:
    config = T5Config(
        vocab_size=100, d_model=64, d_ff=128,
        num_layers=2, num_heads=4, d_kv=16,
    )
    torch.manual_seed(seed)
    model = T5EncoderModel(config)
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
    print("t5_enc: fixtures written")


if __name__ == "__main__":
    main()
