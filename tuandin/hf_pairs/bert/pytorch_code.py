"""HuggingFace BertModel — small config, deterministic init, single forward.

This script is the *input* the cheap LLM is asked to translate to JAX/Flax.
It builds a tiny BERT, sets a torch seed, runs forward on canonical input ids,
and prints the last hidden state's shape and a checksum.
"""
import torch
from transformers import BertConfig, BertModel


def main():
    config = BertConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=32,
    )
    torch.manual_seed(0)
    model = BertModel(config)
    model.eval()

    input_ids = torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29]])
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = out.last_hidden_state
    print("last_hidden_state shape:", tuple(last_hidden.shape))
    print("checksum:", float(last_hidden.sum()))


if __name__ == "__main__":
    main()
