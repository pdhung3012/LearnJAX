# torch2jax baseline findings

## Headline

**pass@1 = 1 / 15 = 6.7%** on the in-scope `nn.Module` cases
(`hf_pairs/` × 10 + `kernelbench_cnn/` × 5).

Only `vgg_bn_small` passed (max abs diff 2.68e-7) — a vanilla
Conv-BN-ReLU-Pool-Linear stack using only ops `torch2jax` natively
supports.

## What this proves for the eval suite

1. **The harness works end-to-end.** A non-LLM tool can be run through
   the pipeline, format-adapted (one-time), and graded against
   `expected.npz` automatically. No manual judging needed.
2. **The eval is genuinely challenging** even for purpose-built
   off-the-shelf tools. We can defend against "couldn't an existing
   tool already do this?" — at least with `torch2jax`'s current
   coverage, the answer is "almost never."
3. **The format-vs-algorithmic split holds up empirically.** The
   adapter handled the format issue once for all 15 cases; every
   subsequent fix would be algorithmic. Since `torch2jax` is
   deterministic, no fix iteration is possible — pass@1 == final score.

## Failure taxonomy (for the paper)

The 14 failures cluster cleanly. **Every failure is a missing op or
attribute in `torch2jax`'s interception layer** — none is a numerical
inaccuracy:

| Failure category | Count | Example |
|---|---:|---|
| Missing op: `torch.matmul`, `torch.clamp`, `torch.nn.functional.silu`, `torch.nn.functional.embedding` | 5 | ViT (`matmul`), MobileNetV2 (`clamp`), EffNet (`silu`), BART/Mistral (`embedding`) |
| Missing Tensor attribute: `.device`, `.long`, `.unsqueeze` | 6 | BERT/GPT-2/DistilBERT/ALBERT (`.device`), RoBERTa (`.long`), Wav2Vec2 (`.unsqueeze`) |
| Missing keyword arg: `mean(dim=...)` (only positional axis supported) | 2 | simple_bn_block, resnet18_small |
| Other (unhashable Torchish, etc.) | 1 | T5 (uses `Torchish` as dict key) |

## Implication for the paper

This baseline gives us:
- **A non-trivial floor**: published, deterministic tool gets 6.7%.
- **A clear failure-taxonomy table** we can show alongside cheap-LLM
  baselines later.
- **Sample fix targets**: the missing ops above are exactly the kinds
  of "translation rule extensions" a fine-tuned cheap LLM should be
  able to learn — they're not fundamentally hard, just absent from the
  rule-based tool.

## Per-case artifacts (the dataset-construction precursor)

Every case writes to `runs/<tier>__<case>/`:

```
runs/<tier>__<case>/
├── pytorch_code.py        # input the tool sees (copy of case PT source)
├── gold_jax_code.py       # target translation (copy of our hand-written reference)
├── candidate_jax/NOTE.md  # torch2jax produces no source — note explains; for cheap-LLM
│                          #   runs this becomes the LLM's emitted jax_code.py
├── error.txt              # full Python traceback (or "PASS")
├── result.json            # status + max_diff + error_category + tool/version
└── fix_step_proposal.md   # one-line proposal for what an algorithmic fix would be
```

The whole tree is the precursor to fix-trajectory training data: the
**cheap-LLM phase will reuse the same layout**, replacing
`candidate_jax/NOTE.md` with the LLM's emitted source and adding a real
`fix_steps/` subdirectory recording the iterative repair. So this run
is doubly useful — it both establishes the deterministic floor AND
validates the dataset-recording structure.

## Provenance

See [CONTEXT.md](CONTEXT.md) for: who built `torch2jax`
(Samuel Ainsworth, github.com/samuela/torch2jax), what its design
goals are, why we picked it as the first baseline, and what it does
NOT do.

## Caveats

- **`torch2jax` is at v0.1.0** — early-stage. A more mature converter
  (e.g., `pytorch2jax`) might pass more cases. Worth running the same
  baseline against another tool for comparison. The architecture of
  this directory makes that mechanical: `cp -r baselines/t2j
  baselines/<other_tool>` and replace the import.
- **Scope is 15/75** because `opinfo` and `jax_translations` are
  pure-function (not `nn.Module`) cases. To extend, we'd add a separate
  pure-function adapter using `torch2jax.t2j_function` (different API).
- **Versioning matters.** torch2jax 0.1.0 results should be tagged with
  the package version in the paper. Future releases may change behavior.

## Suggested next steps

1. **Reproduce on `pytorch2jax`** (or any other off-the-shelf tool) to
   triangulate. ~2 hours of work given the existing scaffolding.
2. **Extend to opinfo/jax_translations** with a function-level adapter
   (~half a day).
3. **Use this run as the "0% LLM" baseline row** in the paper's
   headline table:
   - Row 1: `torch2jax` (6.7%)
   - Row 2: `pytorch2jax` (TBD)
   - Row 3: cheap-LLM zero-shot (TBD, e.g. Code-Llama-13B)
   - Row 4: cheap-LLM + fix-loop (TBD)
   - Row 5: cheap-LLM + ours (fine-tuned) (TBD, the contribution)
4. **The 14 failure cases become the "fix-step" training data** when
   we run the iteration loop with an expensive LLM. Each manual
   intervention to add e.g. a `torch.matmul` translation rule
   corresponds to ~1 algorithmic fix step.
