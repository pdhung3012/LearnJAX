# Evaluation plan: separating format fixes from true fixes

This document records the evaluation methodology for the cheap-LLM → expensive-LLM
fine-tuning pipeline that uses this directory's 30 cases as a held-out test
suite.

## Why two metrics, not one

The headline number we want for the paper is **how much fine-tuning improves a
cheap LLM at PyTorch → JAX translation**. The natural metric is "fix steps
needed until the harness passes." But that single number conflates two
unrelated abilities:

- **Format / API conformance** — does the output define a callable
  `compute(inputs: dict) -> dict` with the right input/output keys?
- **Algorithmic correctness** — given that compute() is callable with the right
  signature, does it produce numerically correct outputs?

If we collapse them into one number, "fix steps reduced after fine-tuning"
will look better than the model actually got, because trivial format-only
fixes (renaming a function, wrapping output in a dict, fixing key spelling)
inflate the count without teaching the model anything about JAX vs PyTorch.
Reviewers will spot this.

## Two metrics, deliberately separated

| Metric | Definition | Measures |
|---|---|---|
| **API conformance rate** | Fraction of generated `jax_code.py` files that import without error AND expose a callable `compute(inputs)` AND, when called on `inputs.npz`, return a dict whose keys exactly match those of `expected.npz`. | Format / signature compliance only. |
| **Pass rate \| API conformance** | Of the API-conformant outputs, fraction that pass `assert_close` against `expected.npz` within tolerance. | Algorithmic correctness only. |
| **pass@1** (composite) | API-conformant ∧ correct, no fixes applied. The standard code-eval metric (Chen et al. 2021, HumanEval). What you'd actually deploy. | The headline number. |
| **Mean fix steps to pass (true fixes only)** | Among outputs that eventually reach correctness, mean number of fix iterations excluding the first iteration if it only fixed format. | Algorithmic improvement signal. |
| **Mean fix steps to pass (any fix)** | Same, including format-only fixes. Reported alongside for transparency. | Total operator effort. |

We always report **all five** numbers. The headline improvement claim is on
**pass@1** + **Mean fix steps (true fixes only)**.

**Naming note:** in iterative-fix variants (Self-Refine, Self-Debug, etc.) the
post-fix metric is sometimes written **pass@1 after N fixes** or **pass rate
after N fixes**. We adopt that convention; "fix step k" means "after k
expensive-LLM repair iterations."

## How a single fix step is classified

Every fix iteration produces a unified diff against the previous candidate.
Classify each diff as one of:

1. **Format-only** — changes are limited to:
   - Renaming the entry function to `compute`
   - Wrapping the bottom of the script in an `if __name__ == "__main__"` guard
   - Adding/removing a top-level `print` or `assert` statement
   - Converting a return value from a tuple/scalar/array to a dict with the
     expected keys
   - Renaming output dict keys (e.g. `"out"` → `"output"`) when the value is
     unchanged
   - Adding `np.asarray(...)` wrappers at the boundary
   - Pure import cleanup (e.g. removing unused imports)
2. **True (algorithmic) fix** — anything else: changing a math operator, a
   weight transpose, a layer order, a loss formula, a softmax axis, etc.
3. **Mixed** — diff contains both. Classify as **true fix** for the metric
   (don't let format changes hide an algorithmic edit).

The classification can be deterministic (regex-based) for category 1; for
ambiguous cases (rare), have the expensive LLM emit a one-line label
("format-only", "algorithmic", "mixed") alongside its diff. Spot-check 5-10%
manually for sanity.

## Why this matters operationally

A fine-tuned cheap model that emits `compute()` correctly but still gets
softmax axes wrong is much worse than one that emits malformed output but
gets the math right. The two-metric split makes that distinction visible.

It also affects which fix-trajectories we keep as training data:

- **(broken format → fixed format)** trajectories teach the model the
  contract. We keep them but downsample — too many will overfit the model on
  the API at the expense of the math.
- **(broken algorithm → fixed algorithm)** trajectories are the high-value
  data. We keep all of them.
- **(broken format AND algorithm → fixed both at once)** mixed trajectories
  we keep but flag as mixed; they're useful but harder to attribute credit
  during analysis.

## Recommended simplification for weak LLM baselines: skeleton prompting

Weak LLMs (small open-source models, ≤7B) tend to produce format errors at a
much higher rate than strong ones — sometimes 50%+ of all failures. If we
prompt them with a free-form spec, the fix-step machinery ends up dominated
by format-only fixes (rename `main` → `compute`, return dict instead of
tuple, etc.). The "fix steps reduced after fine-tuning" headline then
mostly reflects the model learning our output format, not better translation.

**Solution: HumanEval-style skeleton prompting.** Instead of asking the weak
LLM to produce a complete file, give it a stub to fill in:

```python
import numpy as np
import jax.numpy as jnp

def compute(inputs):
    """<docstring with exact input keys/shapes and output keys/shapes>"""
    # YOUR CODE HERE — translate the PyTorch reference below.
    ...

# Reference PyTorch code (do not modify):
# <contents of pytorch_code.py>
```

Effects:

- **Format conformance becomes near-trivial** (95-100%) regardless of model
  strength — any model that can do code completion can produce the body.
- The fix-step count then **cleanly reflects algorithmic quality** alone, so
  the format/true-fix split can be reported as a single-line sanity check
  rather than as parallel headline columns.
- Reviewer objection "is this just better instruction-following?" is
  defused — the format is given, not learned.

**Recommendation:**

- Use skeleton prompting for **all LLM baselines** (weak and strong) in the
  reported headline numbers. This matches HumanEval/MBPP/BigCodeBench
  convention.
- Keep the split classification machinery as a safety net for the residual
  format errors and as a single-line sanity check in the paper. The headline
  metric becomes single-axis ("Mean fix steps to pass").
- For the **format-knowledge ablation** (do models know JAX-style output
  conventions on their own?), run one small experiment: same model, with vs.
  without skeleton, on a subset of cases. Report format-error rate as a
  single sentence. This is enough to address the question without expanding
  the headline metric.

This trades a small amount of analyzability (we no longer measure
"format-only fixes" continuously across the main eval) for a much cleaner,
less-attackable headline result.

## Hybrid handling for non-LLM tools (e.g. `t2j`)

`t2j` is a deterministic Python library — it can't be prompted to follow the
`compute()` contract. We handle it asymmetrically:

1. **For LLM-based candidates** (Code-Llama, DeepSeek-Coder, Claude, etc.):
   prompt with the contract spec upfront. Format-only fix steps will be rare
   but possible.
2. **For deterministic tools** (`t2j`, `pytorch2jax`): wrap the tool's output
   in a thin manual `compute()` adapter that we write **once per tool**. The
   tool's output never sees the contract; the adapter handles it. We report
   `t2j` results as a **clearly separate baseline row** in the paper, with
   "format-only fix steps" set to 0 (the adapter handles format) and only
   true fix steps counted.

## Reporting template (paper-ready)

With skeleton prompting (recommended), API conformance is ≈100% for all LLM
baselines, so the table collapses to a simpler form:

```
                          pass@1   Mean fix    (Format-error
Method                             steps        rate, sanity)
─────────────────────────────────────────────────────────────
t2j (with adapter)        X.X%     N.N         0%   *
pytorch2jax (adapter)     X.X%     N.N         0%   *
Code-Llama-13B            X.X%     N.N         X%
DeepSeek-Coder-6.7B       X.X%     N.N         X%
DeepSeek-Coder-6.7B       X.X%     N.N         X%
  + ours (fine-tuned)
─────────────────────────────────────────────────────────────
Claude Opus (oracle)      X.X%     —           0%

* Adapter handles format; only true fix steps counted.
```

If skeleton prompting is *not* used (e.g. for a free-form generation
ablation), report the full split:

```
                        pass@1   API conf   Pass | API   Mean true   Mean any
Method                           rate       conf        fix steps   fix steps
─────────────────────────────────────────────────────────────────────────────
t2j (with adapter)      X.X%    100% *     X.X%       N.N         N.N
Code-Llama-13B          X.X%    X.X%       X.X%       N.N         N.N
DeepSeek-Coder-6.7B     X.X%    X.X%       X.X%       N.N         N.N
DeepSeek-Coder-6.7B     X.X%    X.X%       X.X%       N.N         N.N
  + ours (fine-tuned)
─────────────────────────────────────────────────────────────────────────────
Claude Opus (oracle)    X.X%    100%       X.X%       —           —

* `t2j` is given a manual format adapter, so its API conformance is
  trivially 100%. Reported for transparency; do not compare across rows on
  this column.
```

## Open questions

- **Tolerances per category.** Should we use looser `atol/rtol` for
  numerically sensitive cases (h13 streaming softmax, h12 SmolLM forward) and
  stricter elsewhere? Current default is 1e-5; h13 uses 1e-3. Worth
  documenting per-case in `SUMMARY.md`.
- **Cap on fix steps.** Currently proposed 5 rounds. Need a pilot to confirm
  this catches realistic distributions without truncating useful trajectories.
- **Held-out vs train split.** All 30 cases here are currently held-out. For
  paper, may want to release a separate "training" set of cases (could
  expand from TorchBench / HF Flax/PyTorch pairs) and keep these 30 as the
  reported eval suite.
