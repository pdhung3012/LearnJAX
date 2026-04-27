"""HuggingFace T5EncoderModel — small config, deterministic init, single forward.

We use the encoder only to keep the case small and avoid cross-attention
(decoder is exercised separately in a future tier 2.1 case).
"""
import torch
from transformers import T5Config, T5EncoderModel


def main():
    config = T5Config(
        vocab_size=100,
        d_model=64,
        d_ff=128,
        num_layers=2,
        num_heads=4,
        d_kv=16,  # head_dim
    )
    torch.manual_seed(0)
    model = T5EncoderModel(config)
    model.eval()

    input_ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    print("last_hidden_state shape:", tuple(out.last_hidden_state.shape))
    print("checksum:", float(out.last_hidden_state.sum()))


if __name__ == "__main__":
    main()
