# Context: why `torch2jax` as the first baseline

## What `torch2jax` is

- **Package:** `torch2jax` v0.1.0 on PyPI.
- **Author:** Samuel Ainsworth (skainsworth@gmail.com).
- **Repository:** https://github.com/samuela/torch2jax
- **Self-described purpose** (from package metadata): *"Run PyTorch in
  JAX. Mix-and-match PyTorch and JAX code with seamless, end-to-end
  autodiff, use JAX classics like `jit`, `grad`, and `vmap` on PyTorch
  code, and run PyTorch models on TPUs."*

## How `torch2jax` works (relevant to baseline framing)

Critically, **`torch2jax` is NOT a source-to-source translator.** It uses
*abstract interpretation* (a.k.a. tracing) at runtime: it intercepts
`torch.*` and `Tensor.*` ops as the PyTorch model executes, and
re-dispatches each one to a JAX equivalent. The user keeps writing
PyTorch code; behind the scenes the *execution* runs in JAX.

Quoting the package README:

> `torch2jax` uses abstract interpretation (aka tracing) to move JAX
> values through PyTorch code. As a result, you get a JAX-native
> computation graph that follows _exactly_ your PyTorch code, down to
> the last epsilon.

This has two consequences for our baseline:

1. **No `jax_code.py` source file is ever produced** — the "candidate
   translation" exists only as the runtime trace. When we say "the tool
   translated this case," we mean "the tool ran this case in JAX." So
   our per-case artifact directories include a `candidate_jax/NOTE.md`
   instead of a candidate source file (see `runs/<case>/`).
2. **A failure means a missing translation rule** — every error is
   `torch2jax` not knowing how to forward a particular `torch.*` call.
   Errors are categorical (the op simply isn't implemented), not
   numerical. This is exactly what we observe in our run: 14/14
   failures are `NotImplementedError` / `AttributeError` from missing
   ops or attributes.

## Why we picked `torch2jax` for the first baseline

1. **Most-credible deterministic option.** It is actively developed,
   reasonably documented, has working tracing of common ops (linear,
   layernorm, conv2d, batchnorm, attention, etc.), and is published
   under a single maintainer with a real GitHub repo. We considered
   `pytorch2jax` (a different package on PyPI by other authors) as an
   alternative and may add it as a parallel baseline; results would be
   complementary.
2. **Architecture-agnostic in principle.** The tracing approach in
   theory works on any `torch.nn.Module`, so the same adapter can run
   it across all our `nn.Module`-based eval cases without per-case
   surgery. This is cleaner than a source-translator that would have
   to handle `pytorch_code.py`'s file-level structure.
3. **Sets a defensible floor in the paper.** A reviewer asking
   "couldn't an off-the-shelf tool already do this?" gets a concrete
   answer: at v0.1.0, `torch2jax` passes 1 of 15 cases (6.7%). That
   number is reproducible by anyone with `pip install torch2jax==0.1.0`.

## What `torch2jax` is NOT and the limits of this baseline

- **Not a complete translator.** Coverage is partial; the 14 failures
  in our run reflect ops/attributes the project hasn't reached yet
  (most ML researchers' goals — Samuel's included — are typically
  proof-of-concept rather than total parity). The pass@1 number we
  report is *of this version of torch2jax*, not of "all
  rule-based translators in principle."
- **Not what a cheap LLM does.** A cheap LLM emits source code; a
  trained interceptor like `torch2jax` runs PyTorch code on JAX. The
  comparison is somewhat apples-to-oranges; we include it as a
  *non-LLM* floor, not as a direct competitor to the cheap-LLM rows
  we'll add next.
- **Not maintained for our use case.** Submitting bug reports for the
  missing ops would be the polite thing to do upstream — but doing so
  isn't part of this paper's contribution. We treat the v0.1.0 surface
  as a fixed point of comparison.

## Versioning

Pinned at **`torch2jax==0.1.0`**. If we re-run later, we should pin the
same version (or report results under both old and new). The
`run_baseline.py` line `pip install torch2jax==0.1.0` makes this
explicit.

## Pointers

- Source: `/opt/miniconda3/envs/t2j/lib/python3.10/site-packages/torch2jax/__init__.py`
  (1100+ LOC, single file; mostly per-op `@implements(torch.foo)`
  registrations).
- Run: `python tuandin/baselines/t2j/run_baseline.py`
- Headline: `tuandin/baselines/t2j/FINDINGS.md`.
- Per-case artifacts (after retrofit): `tuandin/baselines/t2j/runs/<case>/`.
