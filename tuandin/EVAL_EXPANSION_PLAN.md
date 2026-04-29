# Eval expansion plan: Phases 1 + 2 (OpInfo + KernelBench-L3 CNN)

This doc records the motivation and step-by-step process for the next two
expansions of the PyTorch→JAX translation eval suite, on top of the existing
40 cases (`jax_translations/` × 30 + `hf_pairs/` × 10).

## Current state and gaps

The existing 40 cases cover:

- Component-level pieces (linear regression, custom MLPs/RNNs, attention
  primitives) in `jax_translations/`.
- Whole-encoder/decoder transformer architectures (BERT/GPT-2/RoBERTa/T5/
  ViT/DistilBERT/ALBERT/BART/Mistral/Wav2Vec2) in `hf_pairs/`.

When we audit what a real PyTorch→JAX translation pipeline needs to handle,
two specific gaps stand out:

1. **Operator-level corner cases** — in-place mutations (`x.add_`,
   `relu_`), dtype promotion edges (`int * float`, `bool * f32`),
   advanced indexing (`scatter`/`gather`, fancy boolean masks), autograd
   quirks (gradients through `sort`/`topk`/`unique`), and numerical edges
   (`softmax` of all-(-inf), division by zero). These are where cheap LLMs
   most often fail in real-world PT code, but our 40 architectural cases
   barely exercise them.

2. **CNNs with BatchNorm running stats** — every BERT-family case uses
   LayerNorm; ViT/Wav2Vec2 use LayerNorm; T5/Mistral use RMSNorm. Not a
   single case in our suite exercises BatchNorm in eval mode (where the
   `running_mean`/`running_var` buffers are used instead of batch
   statistics). Real-world PyTorch CNNs (ResNet, MobileNet, EfficientNet,
   etc.) all hinge on this and the conversion to Flax requires explicit
   `use_running_average=True` in `flax.linen.BatchNorm`. The translation
   gotcha is severe and we don't currently test it.

## Why exactly these two — and not alternatives

We considered several other expansion directions before committing to OpInfo
+ KB-L3 CNN. Brief rationale for accepting/rejecting each:

| Direction | Decision | Why |
|---|---|---|
| **OpInfo (operator-level)** | ✅ Phase 1 | Tests bug classes that *dominate* real-world cheap-LLM failures (in-place, dtype, scatter). Mechanical to add (~30 cases / half day). Disjoint from existing architectural cases. |
| **KB-L3 CNN with BN** | ✅ Phase 2 | Closes the BatchNorm-running-stats gap. ~5 cases is enough — beyond that, additional architectures retread patterns we already cover. |
| KB-L1, KB-L2 (single ops, fused ops) | ❌ skip | Their "expected output" is a fast CUDA/Triton kernel, not a JAX rewrite. Wrong target for our eval. |
| KB-L3 non-CNN (transformers etc.) | ❌ skip | Already covered by `hf_pairs/`. ~70% overlap. |
| KB-L4 / TorchBench full models | ⏸ defer | High effort per case (~3-5 hours each); lower marginal value than fix-trajectory mining. Revisit after fine-tuning experiments. |
| TorchBench (~80 real PT models) | ⏸ defer | Same reasoning as KB-L4. Useful breadth claim later, not now. |
| HF GitHub PR-diff mining | ⏸ Phase 3 (separate) | This is *training-data* construction, not eval-suite construction. Belongs after Phases 1+2 are landed. |

The shortlist comes down to: **bug-class coverage (OpInfo) + the one
architecture family our 40 cases miss (BN-CNN)**. Together they round out
the eval to ~75 cases hitting every translation challenge a cheap LLM is
likely to face.

---

## Phase 1: `tuandin/opinfo/` — operator-level corner cases

### Motivation

When a cheap LLM translates PyTorch code, the failure modes I see most often
are not "wrong attention layout" — they're things like:

- `x.relu_()` → output corrupted because JAX has no in-place ops; the LLM
  emitted `jnp.maximum(x, 0)` but forgot to *reassign* the result.
- `(int_tensor * 1.0)` → JAX promotes differently than PyTorch in some
  edge cases (especially around bool and complex).
- `torch.scatter_add_(dim, index, src)` → translated to a wrong-shape
  `jax.numpy.scatter` call.
- `softmax` over a row that's entirely `-inf` → PyTorch returns NaN; JAX
  returns `1/N` (depending on platform). Real bug class.

Each of these is a one-line translation but they break fast. A 30-case
operator suite directly targets them.

### What we'll test (categories, ~30 cases total)

| # | Category | Sample cases |
|---|---|---|
| 1 | **In-place ops** (5 cases) | `add_`, `mul_`, `relu_`, `clamp_`, `index_copy_` |
| 2 | **Reductions w/ axis semantics** (5) | `argmax(dim=)`, `argmin`, `cummax`, `cumprod`, `logsumexp` |
| 3 | **Indexing / scatter / gather** (6) | `index_select`, `gather`, `scatter`, `scatter_add`, `masked_fill`, fancy bool indexing |
| 4 | **Dtype edges** (4) | `int * float`, `bool * f32`, `tensor / 0` (inf vs nan), `complex.abs()` |
| 5 | **Autograd corners** (4) | grad through `sort`, `topk`, `unique`, `where` w/ non-finite branch |
| 6 | **Numerics** (3) | `softmax(all -inf row)`, `log(0)`, `exp(very large)` |
| 7 | **Broadcasting edges** (3) | `(B, 1, S, D) * (B, H, 1, D)`, `tensor[None] + tensor[..., None]`, partial broadcast under `where` |

Cases are inspired by PyTorch's `OpInfo` test database
(`torch/testing/_internal/common_methods_invocations.py`) but **hand-written**
in the same `compute(inputs) -> dict` contract as the rest of the suite, so
the harness, runner, and fix-step machinery work without modification.

### Per-case structure (~5-15 LOC each)

```
opinfo/
├── PLAN.md                    # this doc, copy of relevant section
├── _test_utils.py             # symlinks/copy from jax_translations
├── _contract_test_template.py # same as elsewhere
├── run_tests.py
└── <op_name>/
    ├── pytorch_code.py        # one short PT function exercising the op
    ├── freeze_fixtures.py     # build canonical inputs, save expected
    ├── inputs.npz
    ├── expected.npz
    ├── jax_code.py            # candidate JAX translation (compute(inputs))
    └── test_equivalence.py    # generic contract check
```

Each `pytorch_code.py` is short — typically 10-20 LOC defining a single
`compute(inputs)` with the op under test plus enough surrounding tensor
manipulation to make the case non-trivial.

### Step-by-step execution

1. **Pick the 30 ops.** Draft a list per category using the table above as
   a starter. Cross-check against PyTorch's OpInfo to ensure we're hitting
   the same corner cases their internal tests emphasize.
2. **Set up `opinfo/`** with the same scaffolding as `jax_translations/`
   (`_test_utils.py`, `_contract_test_template.py`, `run_tests.py`,
   `PLAN.md`, `pyproject.toml` if needed — likely just the `t2j` env).
3. **For each op (~10-15 min/case mechanical):**
    1. Write `<op>/pytorch_code.py`: short PT function exercising the op
       on a deterministic input.
    2. Write `<op>/freeze_fixtures.py`: build inputs, run PT, save
       `inputs.npz` + `expected.npz`.
    3. Write `<op>/jax_code.py`: candidate JAX implementation. For Phase 1
       we ourselves write this as the "expert reference" since OpInfo cases
       don't have a HuggingFace-style ground truth model.
    4. Drop in `test_equivalence.py` (template copy).
    5. Run; expect bit-equivalence (atol 1e-5) for most cases. Document
       any cases where PT/JAX numerics legitimately differ (e.g. NaN
       handling) and choose tolerance.
4. **Run the full suite** with `run_tests.py`. Expect 30/30 PASS.
5. **Bug-injection probe:** flip one op per category to a known-wrong
   variant (e.g. `add_` → `sub_`, `gather(dim=1)` → `gather(dim=0)`) and
   confirm the harness catches it. Same probe we did for Tier 1.
6. **Commit + push** as a single tier.
7. **Update top-level docs:** add `opinfo/` row to the per-tier table in
   `SUMMARY.md` (in `jax_translations/`) and to `EVAL_PLAN.md`.

### Effort

~30 cases × ~10-15 min each = **~5-7 hours** total. Mechanical work — the
contract is established, the harness exists.

### Success criteria

- 30/30 pass on the first run (or quick fix-up if any case fails).
- Bug-injection probes catch all 7 categories' deliberate breaks.
- `run_tests.py opinfo/` runs in under a minute (no model downloads).
- Each `pytorch_code.py` is self-contained, no `transformers` import.

---

## Phase 2: `tuandin/kernelbench_cnn/` — BatchNorm + real CNN translation

### Motivation

BatchNorm is the single architectural pattern our 40 cases don't exercise.
The translation challenge is non-trivial:

- PyTorch `nn.BatchNorm2d` in `eval()` mode uses `running_mean` /
  `running_var` buffers (computed during training, frozen at inference).
- The state_dict contains these as buffers (not parameters), with names
  like `bn1.running_mean`, `bn1.running_var`, `bn1.num_batches_tracked`.
- The Flax equivalent (`flax.linen.BatchNorm`) splits parameters and
  batch_stats: parameters in `params['scale']`/`['bias']`, running stats
  in `batch_stats['mean']`/`['var']`. They're *separate* PyTrees.
- Calling `BatchNorm(use_running_average=True)` is required at inference
  to read from `batch_stats` instead of computing batch statistics. A
  cheap LLM that omits this flag will get *training-mode* normalization
  and produce wrong outputs.
- Loading PT weights also requires routing `running_mean` / `running_var`
  to the `batch_stats` collection, NOT the regular `params` tree.

This is exactly the kind of corner that breaks cheap LLMs in subtle ways
(numerically-different output that still has the right shape). Adding 5
cases that feature BN explicitly closes the gap.

### What we'll test (5 cases)

| # | Case | Architecture | What it adds |
|---|---|---|---|
| 1 | `simple_bn_block` | Conv-BN-ReLU x 2 + GAP + Linear | Smallest case isolating BN behavior. |
| 2 | `resnet18_small` | ResNet18-style stem + 2 BasicBlock layers | Residual connection + BN inside skip path. |
| 3 | `mobilenet_v2_small` | MBConv block (expand 1×1, depthwise 3×3, project 1×1) with BN after each, ReLU6 | Depthwise-separable conv + BN + ReLU6 (a clamp/activation we don't otherwise test). |
| 4 | `vgg_bn_small` | VGG-style: Conv-BN-ReLU x 4 + Linear classifier | Vanilla CNN-with-BN, the classic. |
| 5 | `effnet_mb_block` | MBConv with squeeze-and-excitation | SE-block (global pool → FC → sigmoid → channel-wise multiply) is its own translation challenge. |

Each is **small**: hidden channels 16-32, 2-4 conv layers, input 32×32×3.
Goal is to test the *patterns*, not production scale.

### Per-case structure (same symmetric design as `hf_pairs/`)

```
kernelbench_cnn/<case>/
├── pytorch_code.py        # Hand-written nn.Module, no torchvision import
├── freeze_fixtures.py     # Build PT model, save weights + run forward
├── inputs.npz             # Canonical pixel input
├── expected.npz           # PT forward output
├── pt_weights/            # safetensors + config.json
├── jax_code.py            # From-scratch Flax/jnp forward (compute(inputs))
└── test_equivalence.py    # Generic contract check
```

The architectures are simple enough to hand-implement (no need to import
`torchvision.models`). KernelBench L3's reference implementations are a
useful sanity check but we don't load them directly — the symmetric design
we adopted for `hf_pairs/` requires `pytorch_code.py` to spell out the
architecture.

### Step-by-step execution

For each of the 5 cases (~2-3 hours each):

1. **Write `<case>/pytorch_code.py`**: hand-written `nn.Module` defining
   the architecture. Use small dimensions so weights commit at <1 MB.
   Include `BatchNorm2d` with deterministic running-stats initialization
   (e.g., set running_mean to small random values, running_var to ~1.0).
2. **Run `freeze_fixtures.py`**:
    - Build the PT model with a fixed seed.
    - Manually populate `running_mean` / `running_var` (we can't rely on
      a training pass; just set them to reproducible values).
    - Call `model.eval()` to put BN in inference mode.
    - Save state_dict to `pt_weights/`, run forward, save outputs.
    - Optional sanity check against a torchvision equivalent if applicable
      (e.g. compare our `resnet18_small` block to `torchvision.models.resnet18`'s
      first 2 layers with matching weights).
3. **Write `<case>/jax_code.py`**: from-scratch Flax/jnp forward. The
   tricky parts:
    - Load BN buffers (`running_mean`, `running_var`) from the safetensors
      dict and route them through the BN formula manually:
      `(x - mean) / sqrt(var + eps) * scale + bias`
    - Or use `flax.linen.BatchNorm(use_running_average=True)` and pass
      `batch_stats` separately. The contract test already supports either
      since `compute()` is opaque.
4. **Drop in `test_equivalence.py`**, run.
5. **Iterate** until pass.
6. **Commit per case** (5 commits) or as a single batch.

### Effort

~5 cases × ~2-3 hours each = **~10-15 hours** total. Heavier than Phase 1
because each case is a real architecture, not a 10-line op test.

### Success criteria

- 5/5 pass with max abs diff < 5e-6.
- Each `pytorch_code.py` is fully spelled out (no `from torchvision import`,
  no `from transformers import`).
- The `simple_bn_block` case explicitly demonstrates the
  `use_running_average=True` requirement: bug-inject by removing the flag
  and confirm the harness catches the difference.
- BatchNorm-running-stats coverage explicit in the suite docs.

---

## Combined deliverables

After Phases 1 + 2:

- 75 total eval cases (40 existing + 30 OpInfo + 5 KB-L3 CNN).
- Coverage matrix:

| Pattern | jax_translations | hf_pairs | opinfo (new) | kernelbench_cnn (new) |
|---|---|---|---|---|
| Linear / MLP / RNN basics | ✅ | — | ✅ | — |
| Transformers (encoder/decoder) | partial | ✅ | — | — |
| Modern LLM (RoPE/GQA/RMSNorm) | h12 only | mistral | — | — |
| Convolutional layers | partial | vit, wav2vec2 | — | ✅ |
| **BatchNorm running stats** | ❌ | ❌ | ❌ | ✅ |
| **In-place ops** | ❌ | ❌ | ✅ | — |
| **Dtype promotion edges** | ❌ | ❌ | ✅ | — |
| **scatter/gather/index** | partial | partial | ✅ | — |
| **Autograd corners** | ❌ | ❌ | ✅ | — |
| Audio | — | wav2vec2 | — | — |

- Documentation: per-tier `PLAN.md` and updates to `SUMMARY.md` /
  `EVAL_PLAN.md` reflecting the expanded coverage.

## Sequencing

1. **Phase 1 (this week)**: build `opinfo/`. Land in one PR/commit batch.
2. **Phase 2 (next week)**: build `kernelbench_cnn/`. Land in one PR/commit
   batch (or 5 small commits, one per case).
3. **Phase 3 (after both)**: pivot to fix-trajectory mining (out of scope
   for this doc).

## Open questions / risks

- **Phase 1 numerics:** some operator edge cases have legitimately
  different PT/JAX behavior (e.g. NaN in `softmax(all -inf)`, signed-zero
  semantics). For each such case, decide tolerance + behavior assertion.
  Worst case: add a "known PT/JAX divergence" tag that the harness allows
  through with a logged note rather than failing.
- **Phase 2 BN init:** PT and JAX will produce identical BN forward only
  if `running_mean` / `running_var` / `eps` match exactly. Need to be
  careful about `momentum=0.1` vs `0.01` defaults and `track_running_stats=True`.
- **Phase 2 sanity check against torchvision:** optional. If we add it,
  we have to load torchvision's pretrained weights into our hand-written
  module — same weight-transfer machinery as `hf_pairs/m4` (the medical-
  imaging case in `jax_translations/`). Probably skip for v1; revisit if
  reviewers ask for "real" weights.
