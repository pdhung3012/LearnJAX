# Tier 2 dataset: HuggingFace PT↔Flax model pairs

This directory extends the evaluation suite beyond the 30 TorchLeet cases in
`../jax_translations/` with **module-level translations** drawn from the
HuggingFace `transformers` library, where every paired (PyTorch, Flax)
implementation is gold-standard expert work.

## Why this tier

Each PT↔Flax pair gives us three valuable things at once:

1. **Realistic, idiom-heavy translation pairs** — every PR that landed a
   `Flax<Model>` started from the PT version and went through expert review.
2. **Real fix-trajectory data** — the Git history of those PRs records how
   reviewers caught bugs (wrong axis on softmax, wrong layout on conv,
   off-by-one positional encoding) that we can mine as training signal.
3. **A built-in weight transfer path** —
   `FlaxAuto<X>.from_pretrained(pt_dir, from_pt=True)` already implements the
   PT→Flax weight conversion, so we don't have to hand-port layouts.

## Contract (same as jax_translations/)

Every subdirectory holds:

- `pytorch_code.py` — short PyTorch script: builds a small `<Model>Config`,
  instantiates the PT model with a fixed seed, runs forward on a canonical
  input batch.
- `freeze_fixtures.py` — runs the PyTorch reference, then:
  - saves the PT weights via `model.save_pretrained("pt_weights/")` so the
    Flax side can load them via `from_pt=True`,
  - saves `inputs.npz` (input_ids + any other tensors) and `expected.npz`
    (the PT forward output, e.g. last_hidden_state).
- `pt_weights/` — committed if small (~few MB), gitignored if large.
- `inputs.npz`, `expected.npz` — frozen ground truth as before.
- `jax_code.py` — exposes `compute(inputs: dict) -> dict`. Internally loads
  weights from `pt_weights/` via `Flax<Model>.from_pretrained(..., from_pt=True)`
  and runs the Flax forward.
- `test_equivalence.py` — generic contract test (same template as Tier 1).

The cheap LLM is given `pytorch_code.py` and the contract spec. It does
**not** see `freeze_fixtures.py`, `pt_weights/`, or `expected.npz`.

## Model selection (Tier 2.0 — initial 5)

CPU-friendly, broadly-supported Flax classes with stable PT↔Flax parity:

| Case dir | PT class | Flax class | Notes |
|---|---|---|---|
| `bert/`    | `BertModel`     | `FlaxBertModel`     | Encoder-only; the canonical example. |
| `gpt2/`    | `GPT2Model`     | `FlaxGPT2Model`     | Decoder-only; tied embeddings. |
| `t5_enc/`  | `T5EncoderModel`| `FlaxT5EncoderModel`| Encoder half of T5 to keep small (no decoder cross-attn). |
| `vit/`     | `ViTModel`      | `FlaxViTModel`      | Vision transformer; pixel inputs instead of token ids. |
| `roberta/` | `RobertaModel`  | `FlaxRobertaModel`  | BERT variant; checks RoBERTa-specific embedding logic. |

**Deliberately deferred** (require more work or are less stable in Flax):

- LLaMA / Mistral — Flax port less mature, GQA + RoPE complications.
- Wav2Vec2 / Whisper — audio inputs are awkward + larger.
- BART / mBART — encoder-decoder doubles the complexity for the same value.
- DistilBERT — closely tracks BERT, low marginal value at this stage.

## Dimensions kept small

Every case uses a tiny config so it runs in seconds on CPU and the
`pt_weights/` directory stays a few MB:

```
hidden_size = 64
num_hidden_layers = 2
num_attention_heads = 4
intermediate_size = 128
max_position_embeddings = 32      # most cases
vocab_size = 100                   # truncated from real vocab
```

Real-world correctness depends only on shape + arithmetic, not on
hyperparameters being production-scale.

## Process per case

1. Write `freeze_fixtures.py` with the small config + canonical inputs +
   PT model save.
2. Run it once → produces `pt_weights/`, `inputs.npz`, `expected.npz`.
3. Write `jax_code.py` that loads `pt_weights/` via `from_pt=True` and
   re-implements the forward. The pretrained-loading path makes weight
   transfer trivial.
4. Drop in the generic contract test from `../jax_translations/_contract_test_template.py`.
5. Verify the test passes (`expected.npz` matches Flax output within
   atol/rtol).
6. Commit.

## Open questions / risks

- **Tolerance.** PT↔Flax aren't bit-equivalent due to numerical differences
  (LayerNorm reductions, dropout being deterministic-off in eval, etc).
  Expected diffs around 1e-4 to 1e-5 in fp32. May need to relax `atol/rtol`
  per case.
- **Flax availability.** Each chosen model must have `Flax<X>Model` exposed
  in the installed `transformers` version. We verify in step 1 of each case.
- **Fix-trajectory mining (separate work).** Mining HF PRs to extract
  reviewer fixes is a *separate* dataset-construction task, not done here.
  This tier just builds the eval suite.

## Tier 2.1+ (future)

Once the initial 5 are validated:

- Add encoder-decoder (BART, T5 full) to exercise cross-attention.
- Add LLaMA / Mistral (deferred — would reuse our `h11`/`h12` GQA + RoPE work).
- Mine HF GitHub PR diffs to extract `(broken-Flax-init, reviewer-fix, merged)`
  trajectories — direct fine-tuning supervision.
