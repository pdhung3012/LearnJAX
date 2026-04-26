"""JAX translation of m11: extract sentence embeddings from SmolLM2-135M.

Faithful to PyTorch:
- Same model, same tokenizer, same algorithm:
    forward with output_hidden_states=True
    take last hidden state
    mean over non-padding tokens
    cosine similarity vs a keyword embedding
- Uses HuggingFace `transformers`' Flax port (`FlaxAutoModelForCausalLM`).
  At time of writing, `HuggingFaceTB/SmolLM2-135M` does not ship Flax weights
  on the Hub; we load PyTorch weights and convert with `from_pt=True`. That
  conversion is one-time so its cost doesn't enter the steady-state runtime.

Speed:
- For inference on a small batch on CPU, JAX is comparable to or slightly
  faster than PyTorch once compiled (the `model.__call__` is jit'd by
  transformers' Flax glue).
- The `from_pt=True` weight conversion adds ~1-2 s on the first call; that's a
  cold-start cost, not a per-step cost.
"""
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM


def cosine_similarity(a, b, eps=1e-8):
    a_norm = a / jnp.maximum(jnp.linalg.norm(a, axis=-1, keepdims=True), eps)
    b_norm = b / jnp.maximum(jnp.linalg.norm(b, axis=-1, keepdims=True), eps)
    return jnp.sum(a_norm * b_norm, axis=-1)


def main():
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
    model = FlaxAutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", from_pt=True
    )

    encodings = tokenizer(reviews, return_tensors="jax", padding=True, truncation=True)
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    last_hidden = out.hidden_states[-1]                          # (B, S, H)
    mask = attention_mask[..., None].astype(jnp.float32)         # (B, S, 1)
    summed = jnp.sum(last_hidden * mask, axis=1)                 # (B, H)
    counts = jnp.maximum(jnp.sum(mask, axis=1), 1e-9)            # (B, 1)
    sentence_embeddings = summed / counts
    print("sentence_embeddings shape:", sentence_embeddings.shape)

    keyword = "quality"
    kw_enc = tokenizer(keyword, return_tensors="jax")
    kw_out = model(input_ids=kw_enc["input_ids"],
                   attention_mask=kw_enc["attention_mask"],
                   output_hidden_states=True)
    kw_h = kw_out.hidden_states[-1]
    kw_m = kw_enc["attention_mask"][..., None].astype(jnp.float32)
    kw_emb = jnp.sum(kw_h * kw_m, axis=1) / jnp.maximum(jnp.sum(kw_m, axis=1), 1e-9)

    sims = cosine_similarity(sentence_embeddings, kw_emb)
    for i, (r, s) in enumerate(zip(reviews, sims)):
        print(f"#{i + 1} sim('{keyword}') = {float(s):+.4f}: {r}")


if __name__ == "__main__":
    main()
