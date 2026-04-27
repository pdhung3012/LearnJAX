"""HuggingFace RobertaModel — small config, deterministic init, single forward.

RoBERTa differs from BERT in: GPT-2-style BPE, no token_type_ids, position
ids start at pad_token_id+1=2 (the source of many translation bugs).
"""
import torch
from transformers import RobertaConfig, RobertaModel


def main():
    config = RobertaConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=32,
        type_vocab_size=1,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    model = RobertaModel(config)
    model.eval()

    input_ids = torch.tensor([[0, 5, 9, 13, 17, 21, 25, 2]])  # BOS ... EOS
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    print("last_hidden_state shape:", tuple(out.last_hidden_state.shape))
    print("checksum:", float(out.last_hidden_state.sum()))


if __name__ == "__main__":
    main()
