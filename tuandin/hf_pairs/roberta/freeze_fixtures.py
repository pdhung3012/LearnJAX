"""Freeze fixtures for the RoBERTa case."""
from pathlib import Path
import numpy as np
import torch
from transformers import RobertaConfig, RobertaModel

HERE = Path(__file__).parent


def make_inputs():
    return {
        "input_ids":      np.array([[0, 5, 9, 13, 17, 21, 25, 2]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1]],     dtype=np.int64),
    }


def build_pt_model(seed: int = 0) -> RobertaModel:
    config = RobertaConfig(
        vocab_size=100, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=128,
        max_position_embeddings=32, type_vocab_size=1,
        pad_token_id=1, bos_token_id=0, eos_token_id=2,
    )
    torch.manual_seed(seed)
    model = RobertaModel(config)
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
    print("roberta: fixtures written")


if __name__ == "__main__":
    main()
