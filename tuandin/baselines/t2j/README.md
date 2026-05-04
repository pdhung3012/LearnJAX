# Baseline: `torch2jax` deterministic tool

This directory measures **pass@1 of the deterministic `torch2jax` tool**
across our eval suite, **without any LLM in the loop**. It serves three
purposes:

1. **Smoke-test the eval pipeline.** Before pointing expensive LLMs at
   the suite, we run a free deterministic baseline to confirm:
     - the harness can compare arbitrary `compute()` outputs to
       `expected.npz`,
     - `pt_weights/` weights load correctly into a tool-translated JAX,
     - the tolerance levels are reasonable.
2. **Establish a published baseline.** Anyone reading the paper can ask
   "couldn't an off-the-shelf tool already do this?" — `torch2jax`'s
   pass@1 number answers that.
3. **Make the format-vs-algorithmic split concrete.** `torch2jax` cannot
   follow instructions about output format, so we apply a manual
   one-time **format adapter** (see `adapter.py`) that wraps its output
   in our `compute(inputs: dict) -> dict` contract. Per the methodology
   in `EVAL_PLAN.md`, format-fix steps for deterministic tools are
   counted as **0** (handled by the adapter) and only **algorithmic**
   fixes count toward the headline `Mean fix steps` metric.

## What we measure

For each case that the tool can attempt:

| Metric | Definition |
|---|---|
| **applicable** | Does the case expose a callable `nn.Module` (via `build_pt_model()` in `pytorch_code.py`)? `opinfo` and parts of `jax_translations` are pure-function cases that `t2j_module` does not target. |
| **API conformance** | After the format adapter wraps the tool's output, is `compute(inputs) -> dict` callable and does it return the right keys? Should be 100% by construction (the adapter handles format). |
| **pass@1** | Does the wrapped output pass `test_equivalence.py` against `expected.npz` within atol=1e-5? |

We do **not** iterate fix steps for `torch2jax` — it's deterministic, can't
take feedback. Each case is a single measurement: pass or fail.

## Scope (initial run)

- **Tier 2 `hf_pairs/`** (10 cases) — every case ships `build_pt_model()`
  and committed `pt_weights/` so the baseline is plug-and-play.
- **Tier 4 `kernelbench_cnn/`** (5 cases) — same.
- Total: **15 cases** in scope for v1.
- `jax_translations/` (30) and `opinfo/` (30) deferred — they use a
  pure-function `compute(inputs)` style rather than `nn.Module`, so a
  different adapter (`t2j_function`) would be needed. We can add this in
  v2 if results justify it.

## Files

- `adapter.py` — the one-time format adapter. Takes a case directory,
  imports its `pytorch_code.build_pt_model`, runs `t2j_module`, and
  exposes a `compute(inputs)` matching the contract.
- `run_baseline.py` — driver that walks the in-scope cases, runs the
  adapter, calls each case's `test_equivalence.py` mechanism, and
  prints pass/fail + max abs diff.
- `results.md` — table of results (filled in by `run_baseline.py`).

## How to run

```bash
# Make sure torch2jax is installed in the t2j env:
/opt/miniconda3/envs/t2j/bin/pip install torch2jax==0.1.0

# Run baseline:
/opt/miniconda3/envs/t2j/bin/python /Users/tuandinh/repo/LearnJAX/tuandin/baselines/t2j/run_baseline.py
```
