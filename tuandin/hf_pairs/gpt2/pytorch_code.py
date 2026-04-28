"""HuggingFace GPT2Model — small config, deterministic init, single forward.

Decoder-only transformer with tied input/output embeddings.
"""
import torch
from transformers import GPT2Config, GPT2Model


def main():
    config = GPT2Config(
        vocab_size=100,
        n_positions=32,
        n_embd=64,
        n_layer=2,
        n_head=4,
        n_inner=128,
    )
    torch.manual_seed(0)
    model = GPT2Model(config)
    model.eval()

    input_ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    print("last_hidden_state shape:", tuple(out.last_hidden_state.shape))
    print("checksum:", float(out.last_hidden_state.sum()))


if __name__ == "__main__":
    main()
