# PyTorch → JAX Translation Benchmarks

The translations are organized into two batches:

1. **Basic deep-learning cases** (`e1`–`e7`, `m1`/`m3`–`m8`, `h1`/`h3`–`h6`/`h10`) —
   sourced from `intrinsic_data_fixed/verification/<case>/`. Linear regression,
   CNNs, autoencoders, basic LSTM/RNN, GAN, seq2seq, etc.
2. **Modern LLM cases** (`e8`–`e11`, `m9`–`m11`, `h11`–`h13`) — sourced from
   the [TorchLeet](https://github.com/Exorust/TorchLeet) `llm/` directory.
   RMSNorm, sinusoidal/rotary positional embeddings, BPE tokenizer, attention
   from scratch, multi-head and grouped-query attention, sentence embeddings
   from a pre-trained SmolLM2-135M, full SmolLM-135M from scratch, Flash
   Attention v2.

Every subdirectory holds these files:
- `pytorch_code.py` — the (lightly cleaned) reference, runnable as a script.
- `jax_code.py` — a faithful JAX translation written for this benchmark. **It must expose a `compute(inputs: dict) -> dict` contract function** (see "Contract architecture" below).
- `freeze_fixtures.py` — one-shot script that generates `inputs.npz` and `expected.npz` from the canonical PyTorch reference. Run once, output committed.
- `inputs.npz` — frozen canonical inputs (data + weights + scalar config).
- `expected.npz` — frozen PyTorch-reference outputs on those inputs. **This is the ground truth.**
- `test_equivalence.py` — generic ~30-line harness: loads the two `.npz` files, calls `compute(inputs)`, asserts elementwise closeness on every output key.

`run_bench.py` runs both files with `time.time()` around a `subprocess.run`, so the
reported numbers are end-to-end wall-clock per script (process start, library
imports, data generation, training, inference, `print` calls — everything).
`run_tests.py` runs the per-case `test_equivalence.py` (sequentially or with
`--random N --seed S` for sampling).

## Contract architecture (test_equivalence)

PyTorch and JAX have different RNGs, so end-to-end training equivalence isn't a
meaningful target. Equivalence is instead defined as: **given the same canonical
inputs, the JAX `compute()` must produce numerically the same outputs as the
PyTorch reference**. The PyTorch outputs are computed once and frozen; tests
compare against the frozen artifact, not against a live PyTorch run.

```
                  freeze_fixtures.py  (run ONCE, outputs committed)
                     │
                     ├──> generates canonical inputs from a fixed seed
                     │       │
                     │       └──> inputs.npz  ←─────────────┐
                     │                                       │
                     ├──> runs PyTorch reference(inputs)     │
                     │       │                               │
                     │       └──> expected.npz   (the        │
                     │             "ground truth" frozen     │
                     │             once and committed)       │
                     ▼                                       │
                  (script exits)                             │
                                                             │
                                                             │ same file,
                                                             │ loaded at test time
                                                             │
                  test_equivalence.py                        │
                     │                                       │
                     ├──── load inputs.npz   ──────────────  ┘
                     │
                     ├──── call jax_code.compute(inputs)  ──> actual outputs
                     │
                     ├──── load expected.npz             ──> ground truth
                     │
                     └──── assert_close(actual, expected, atol/rtol)
```

**Three artifacts, two phases:**

| Phase | Script | Reads | Writes |
|---|---|---|---|
| 1 (one-time) | `freeze_fixtures.py` | RNG seed + PyTorch ref code | `inputs.npz`, `expected.npz` |
| 2 (every test run) | `test_equivalence.py` | `inputs.npz`, `expected.npz`, `jax_code.py` | (PASS / assertion error) |

JAX never runs `freeze_fixtures.py`. It just reads the already-frozen
`inputs.npz` via `compute(inputs)` and is judged against the already-frozen
`expected.npz`.

### The contract

Each `jax_code.py` must define:

```python
def compute(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run the case's deterministic core on canonical inputs.

    Args:
        inputs: dict whose keys match those of inputs.npz (data tensors,
                weight tensors, and 0-d scalar config arrays).
    Returns:
        dict whose keys match those of expected.npz (output tensors).
    """
```

The function takes numpy at the boundary and returns numpy at the boundary;
internal JIT/Flax/equinox/raw-`jnp` choices are hidden. This makes the harness:

- **Robust to renaming.** The cheap LLM is free to call its internal class `Foo`,
  `Attention`, etc. — the harness only imports `compute`.
- **Robust to JAX style.** A pure-`jax.numpy`, Flax, or `equinox` implementation
  all satisfy the same contract.
- **An actual oracle.** Bug-injection probes confirm the harness fails when
  `compute()` is wrong (sign flip, argument swap, weight-layout error).

### Why this matters for the cheap-LLM evaluation loop

For the planned fine-tuning pipeline (cheap LLM / `t2j` tool produces JAX,
expensive LLM iteratively fixes failures), the cheap model sees:

- `pytorch_code.py` and the contract spec ("write `compute(inputs)` that maps
  these input keys to these output keys").

It does **not** see `freeze_fixtures.py` or `expected.npz` — those would leak
the answer. After it emits a candidate, the harness loads `inputs.npz` +
`expected.npz` (which the LLM never saw) and judges. So `inputs.npz` and
`expected.npz` together act as a **held-out test set, per case**, with the
same role as a hidden test in LeetCode.

The pass/fail signal (plus the harness error message and per-output max-abs-
diff) is the supervision the expensive LLM uses to emit a fix; each (broken
candidate, fix diff, harness signal) tuple is direct training data.

## Hardware / software config

| Component | Value |
|---|---|
| CPU | Apple M4 (10 physical / 10 logical cores) |
| RAM | 32 GB |
| OS | macOS 26.3.1 (Darwin 25.3.0 arm64) |
| Accelerator | none — CPU only |
| Python | 3.10.19 |
| PyTorch | 2.5.1 (default thread count: 4) |
| JAX | 0.6.2, backend = `cpu` |
| Flax | 0.10.7 |
| Optax | 0.2.2 |
| NumPy | 2.2.5 |

Runs were single-shot (no warm-up averaging), so a few hundred ms of jitter is
expected. The relative ranking is stable across re-runs.

## Latency results

`speedup` = `pytorch_s / jax_s`. **>1.00 means JAX was faster.**

| Case | Workload | PyTorch (s) | JAX (s) | Speedup | Comment |
|---|---|---:|---:|---:|---|
| e1 | Linear regression, full-batch SGD, 1000 epochs | 1.277 | 0.640 | **1.99×** | Classic jit win. |
| e2 | e1 + DataLoader from CSV (32-batch shuffle) | 1.629 | 1.555 | 1.05× | pandas/CSV I/O dominates. |
| e3 | Linear reg with custom `tanh(x)+x` + matplotlib | 1.360 | 0.816 | **1.67×** | |
| e4 | Linear reg with hand-rolled Huber loss | 1.107 | 0.658 | **1.68×** | |
| e5 | 2→10→1 MLP + Adam | 1.137 | 0.889 | **1.28×** | |
| e6 | Linear reg + per-epoch TensorBoard logging | 4.152 | 3.490 | **1.19×** | TensorBoard write dominates both. |
| e7 | Train Linear(1,1), save, reload, predict | 1.026 | 0.642 | **1.60×** | JAX uses `pickle` for state_dict. |
| h1 | identical to e7 | 1.024 | 0.617 | **1.66×** | |
| h3 | Small transformer encoder regression, 1000 epochs | 4.860 | 5.527 | 0.88× | Small attention on CPU XLA loses to PyTorch MHA. |
| h4 | 1-D GAN, 1000 epochs | 1.885 | 2.367 | 0.76× | Two tiny Adam updates per step; PyTorch wins narrowly. |
| h5 | Seq2seq with attention, 12-step decoder unroll, 100 epochs | 1.705 | 6.039 | 0.28× | Big unrolled `jit` program. `lax.scan` would likely flip this to >1×. |
| h6 | LSTM LM + dynamic int8 quant | **PyTorch FAILS** (no qnnpack engine on macOS arm64) | 2.93 | n/a | JAX wins by running at all. |
| m1 | Hand-rolled LSTM + `nn.LSTM`, 500 epochs each | 3.149 | 3.754 | 0.84× | Flax `OptimizedLSTMCell` < PyTorch MKL-fused LSTM on CPU. |
| m5 | Per-window RNN training (90 steps × 500 epochs) | 15.302 | 4.869 | **3.14×** | Per-step Python overhead in PyTorch is the bottleneck. |

### Modern LLM cases (TorchLeet `llm/`)

| Case | Workload | PyTorch (s) | JAX (s) | Speedup | Comment |
|---|---|---:|---:|---:|---|
| e8  | RMSNorm — learnable per-feature scale; (3, 5) input | 0.660 | 0.562 | **1.17×** | Trivial elementwise; both are basically import-time. |
| e9  | Sinusoidal positional embedding table; query at seq_len=50 | 0.658 | 0.490 | **1.34×** | Pure data construction; JAX wins on import + numpy-style array build. |
| e10 | Scaled-dot-product attention from scratch; assert vs `F.scaled_dot_product_attention` (PyTorch) / `jax.nn.dot_product_attention` (JAX) | 0.660 | 0.598 | **1.10×** | Tiny tensors; ≈ tied. |
| e11 | Byte-pair encoder (pure Python) | 0.009 | 0.008 | 1.07× | No tensor work — both files do the same thing. |
| m9  | Multi-head attention from scratch + comparison vs Flax/PyTorch built-in MHA | 0.670 | 0.949 | 0.71× | Two separate model inits + two forwards; jit compile dominates. |
| m10 | Rotary positional embeddings (RoPE); apply to (S, B, H, D) tensors | 0.657 | 0.633 | **1.04×** | Pure broadcast/elementwise; tied. |
| m11 | Extract sentence embeddings from pre-trained SmolLM2-135M + cosine similarity | not benchmarked here — needs HuggingFace download (~270 MB). Expect roughly **~0.8–1.0×**: PyTorch model is the original; JAX path goes through `from_pt=True` weight conversion (one-time cold cost) before steady-state inference. |
| h11 | Grouped-query attention (8 Q heads, 2 KV heads, d_model=64) | 0.681 | 0.808 | 0.84× | Single forward; jit compile dominates. |
| h12 | SmolLM-135M from scratch (30 LlamaDecoder layers, GQA, RoPE, RMSNorm, SwiGLU) — single forward on a 4-token prompt | 1.132 | 2.114 | 0.54× | Single-shot forward pays full XLA compile (~30 layers compiled per call). For repeated decoding/training the compile cost amortizes. |
| h13 | Flash Attention v2 forward — Triton kernel (PyTorch) vs streaming-softmax JAX implementation | **PyTorch needs CUDA + Triton** (not available on this Mac) | 1.14 (incl. compile) | n/a | JAX runs cleanly on CPU and matches vanilla attention to ~5e-7 abs diff. Pallas kernel stub included for GPU/TPU. |

### Cases not directly benchmarked

| Case | Reason | Expected speedup |
|---|---|---|
| m3 | Downloads CIFAR-10 (~170 MB), 5 init schemes × 10 epochs each | ~1.0× — bottlenecked by torchvision DataLoader (shared by both). |
| m4 | 100×10 CT volume, ResNet18 backbone with ported pretrained weights, 3D conv head | 0.5–0.8× on CPU — XLA-CPU 3D / transposed conv is slower than MKL-DNN. Closes on GPU. |
| m6 | CIFAR-10 augmentation visualization, no training | identical (no compute path differs). |
| m7 | MNIST 784→128→10, SGD, 5 epochs | 1.0–1.3× — DataLoader-bound. |
| m8 | MNIST conv autoencoder, 10 epochs | 0.9–1.1× — DataLoader-bound. |
| h10 | Grad-CAM with pretrained ResNet18 on FakeData; weights ported from torchvision into Flax; activation gradient recovered via a zero-perturbation trick | 0.7–1.0× — single forward + backward; both run in <1 s. |

## Notes on the modern-LLM batch

A few places where the original TorchLeet notebooks have bugs or shortcuts;
the translations document them in the per-file docstring:

- **m9 (MHA)** — the notebook's `multi_head_attention` constructs fresh
  `nn.Linear` layers inside the function, so the weights are random on every
  call. Its assertion against `nn.MultiheadAttention` therefore fails. The
  PyTorch script preserves this verbatim; the JAX translation builds the
  projections as proper module parameters, which is the *intended* design.
- **m10 (RoPE)** — the `Rotary` class caches `cos/sin` shaped `(S, 1, 1, D)`,
  which only broadcasts cleanly against the GPT-NeoX layout `(S, B, H, D)`.
  The original notebook's "test" actually invoked `apply_rotary_pos_emb`
  with the wrong number of arguments and would never have run; the
  pytorch_code.py here uses the `(S, B, H, D)` layout to make the design
  testable. The JAX translation uses the more common `(B, S, H, D)` layout
  with `cos/sin` shaped `(1, S, 1, D)`.
- **m11 (LLM embeddings)** — the notebook calls
  `dataset['full'][:1000]` which returns a `dict` of columns, not a `list`
  of strings. The translation replaces the dataset slice with 10 hardcoded
  reviews so the script actually runs end-to-end.
- **h12 (SmolLM)** — the original test loads a saved checkpoint
  (`BareBones_SmolLM-135M.pt`) that isn't in the repo. Both files here run a
  forward pass with random init; logit shape is asserted, but no weight
  comparison is attempted.
- **h13 (Flash Attention)** — the PyTorch reference is a Triton kernel and
  needs CUDA. The JAX file gives a portable `jax.lax.scan`-based streaming
  implementation that runs anywhere, plus a commented Pallas kernel stub for
  GPU/TPU.

## Where JAX wins, where it doesn't

- **Wins:** small/medium models with many cheap optimizer steps. Dispatch overhead
  per step is a fixed cost in PyTorch eager that XLA fuses away. e1/e3/e4/e5/e7/h1
  all benefit; **m5 is the cleanest win at 3.14×** because the inner Python loop
  fires the optimizer 90×500 = 45 000 times.
- **Losses:** workloads dominated by mature PyTorch CPU kernels (`nn.LSTM` in m1,
  multi-head attention in h3, 3D conv in m4) or where XLA produces a large
  unrolled program for tiny per-step compute (h4 GAN, h5 seq2seq). On GPU the
  picture for h3/m4 typically flips back in JAX's favor.
- **Ties:** I/O- or sync-bound scripts (e2 CSV, e6 TensorBoard) where compute is a
  small fraction of wall time.
- **Functional win:** h6 — PyTorch's dynamic-quant kernel is unavailable on this
  Apple Silicon build; the JAX translation emulates int8 weight quantization with
  a portable Q/DQ pass and runs cleanly.

## Reproducing

```bash
# Run latency benchmarks:
/opt/miniconda3/envs/t2j/bin/python run_bench.py
/opt/miniconda3/envs/t2j/bin/python run_bench.py e1 e5 m5      # subset

# Run equivalence tests (contract-based):
/opt/miniconda3/envs/t2j/bin/python run_tests.py
/opt/miniconda3/envs/t2j/bin/python run_tests.py --random 5 --seed 42

# Regenerate fixtures for a single case (only after intentionally changing
# the canonical inputs or the PyTorch reference):
cd <case>/ && /opt/miniconda3/envs/t2j/bin/python freeze_fixtures.py
```

`run_bench.py` sets `MPLBACKEND=Agg` so matplotlib `.show()` calls don't pop a
window. Cases that download CIFAR/MNIST are not in the default list.
