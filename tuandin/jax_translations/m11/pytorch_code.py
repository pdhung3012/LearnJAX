"""Extract sentence embeddings from a pre-trained causal LM (SmolLM2-135M).

Source: TorchLeet llm/Create-Embeddings-out-of-an-LLM/embeddings-q2.ipynb.

The notebook bug — `dataset['full'][:1000]` returns a dict-of-lists, not a list
of strings — is fixed here so the script actually runs (we extract the
'text' column and take the first 10 reviews).
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    # 10 example reviews instead of pulling the dataset, to keep this self-contained.
    reviews = [
        "I love this product! Great quality and fast shipping.",
        "Terrible product, broke after one use. Do not buy.",
        "Pretty good for the price. Would recommend to friends.",
        "Average quality, nothing special.",
        "Excellent quality! Worth every penny.",
        "Worst purchase I've ever made.",
        "Very satisfied with this purchase.",
        "Quality is okay but the packaging was damaged.",
        "Amazing! Exceeded my expectations.",
        "Mediocre product, would not buy again.",
    ]

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    model.eval()

    device = "cpu"
    model.to(device)

    encodings = tokenizer(reviews, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1]                     # (B, S, H)
    mask = attention_mask.unsqueeze(-1).float()                 # (B, S, 1)
    summed = (last_hidden * mask).sum(dim=1)                    # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-9)                    # (B, 1)
    sentence_embeddings = summed / counts                       # (B, H)
    print("sentence_embeddings shape:", sentence_embeddings.shape)

    keyword = "quality"
    kw_enc = tokenizer(keyword, return_tensors="pt")
    with torch.no_grad():
        kw_out = model(kw_enc["input_ids"].to(device),
                       attention_mask=kw_enc["attention_mask"].to(device),
                       output_hidden_states=True)
    kw_h = kw_out.hidden_states[-1]
    kw_m = kw_enc["attention_mask"].unsqueeze(-1).float().to(device)
    kw_emb = (kw_h * kw_m).sum(dim=1) / kw_m.sum(dim=1).clamp(min=1e-9)

    sims = F.cosine_similarity(sentence_embeddings, kw_emb)
    for i, (r, s) in enumerate(zip(reviews, sims)):
        print(f"#{i + 1} sim('{keyword}') = {s.item():+.4f}: {r}")


if __name__ == "__main__":
    main()
