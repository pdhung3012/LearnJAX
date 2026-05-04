# torch2jax baseline results

**pass@1 = 1/15 = 6.7%** (deterministic torch2jax v0.1.0 with one-time format adapter; no LLM iteration).

Atol/rtol = 1e-5. See [CONTEXT.md](CONTEXT.md) for tool provenance, [FINDINGS.md](FINDINGS.md) for paper-facing analysis, and `runs/<tier>__<case>/` for per-case artifacts (input PT, gold JAX, error log, fix proposal).

| Case | Status | error_category | max_diff |
|---|---|---|---:|
| `hf_pairs/bert` | RUNTIME_ERROR | `missing_attr:device` | — |
| `hf_pairs/gpt2` | RUNTIME_ERROR | `missing_attr:device` | — |
| `hf_pairs/roberta` | RUNTIME_ERROR | `missing_attr:long` | — |
| `hf_pairs/distilbert` | RUNTIME_ERROR | `missing_attr:device` | — |
| `hf_pairs/albert` | RUNTIME_ERROR | `missing_attr:device` | — |
| `hf_pairs/t5_enc` | RUNTIME_ERROR | `unhashable` | — |
| `hf_pairs/vit` | RUNTIME_ERROR | `missing_op:torch.matmul` | — |
| `hf_pairs/bart_enc` | RUNTIME_ERROR | `missing_op:torch.nn.functional.embedding` | — |
| `hf_pairs/mistral` | RUNTIME_ERROR | `missing_op:torch.nn.functional.embedding` | — |
| `hf_pairs/wav2vec2` | RUNTIME_ERROR | `missing_attr:unsqueeze` | — |
| `kernelbench_cnn/simple_bn_block` | RUNTIME_ERROR | `missing_kwarg:dim` | — |
| `kernelbench_cnn/resnet18_small` | RUNTIME_ERROR | `missing_kwarg:dim` | — |
| `kernelbench_cnn/vgg_bn_small` | PASS | `—` | 2.682e-07 |
| `kernelbench_cnn/mobilenet_v2_small` | RUNTIME_ERROR | `missing_op:torch.clamp` | — |
| `kernelbench_cnn/effnet_mb_block` | RUNTIME_ERROR | `missing_op:torch.nn.functional.silu` | — |
