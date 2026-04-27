"""HuggingFace ViTModel — small config, deterministic init, single forward.

Vision transformer with learned positional embeddings on patch tokens.
Uses pixel_values input instead of token ids.
"""
import torch
from transformers import ViTConfig, ViTModel


def main():
    config = ViTConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        image_size=32,   # tiny image
        patch_size=8,    # 16 patches + cls = 17 tokens
        num_channels=3,
    )
    torch.manual_seed(0)
    model = ViTModel(config)
    model.eval()

    # Deterministic pixel input, NCHW, normalised to [-1, 1].
    rng = torch.Generator().manual_seed(42)
    pixel_values = torch.rand((1, 3, 32, 32), generator=rng) * 2 - 1

    with torch.no_grad():
        out = model(pixel_values=pixel_values)
    print("last_hidden_state shape:", tuple(out.last_hidden_state.shape))
    print("checksum:", float(out.last_hidden_state.sum()))


if __name__ == "__main__":
    main()
