"""Freeze fixtures for the ViT case."""
from pathlib import Path
import numpy as np
import torch
from transformers import ViTConfig, ViTModel

HERE = Path(__file__).parent


def make_inputs():
    rng = np.random.default_rng(42)
    # PyTorch convention: (B, C, H, W). Flax ViT *also* expects NCHW per
    # transformers' implementation (it transposes internally). So we stick
    # with NCHW for the contract.
    pixel_values = (rng.uniform(0, 1, (1, 3, 32, 32)) * 2 - 1).astype(np.float32)
    return {"pixel_values": pixel_values}


def build_pt_model(seed: int = 0) -> ViTModel:
    config = ViTConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=128, image_size=32, patch_size=8, num_channels=3,
    )
    torch.manual_seed(seed)
    model = ViTModel(config)
    model.eval()
    return model


def main():
    inputs = make_inputs()
    model = build_pt_model()
    pt_dir = HERE / "pt_weights"
    model.save_pretrained(pt_dir)

    with torch.no_grad():
        out = model(pixel_values=torch.from_numpy(inputs["pixel_values"]))
    expected = {"last_hidden_state": out.last_hidden_state.numpy()}

    np.savez(HERE / "inputs.npz", **inputs)
    np.savez(HERE / "expected.npz", **expected)
    print("vit: fixtures written")


if __name__ == "__main__":
    main()
