# g_fs_3

Benchmarking of large language models on code-generation tasks using the **TorchLeet** dataset. This repository contains model outputs, fix logs, and manually annotated JAX conversions.

## Repository Structure

```
models/
├── Codegemma-7b-it/
├── Mistral-7b/
```

Each model directory contains:

- **fix_logs/** — Logs of fixes applied during evaluation.
- **outputs/torchleet/** — Results organized by difficulty level (`basic`, `easy`, `medium`, `hard`).

### Outputs by difficulty

Within each difficulty folder:

| Content | Description |
|--------|-------------|
| **JAX files** | Manually annotated, fully working JAX implementations. |
| **\*.jsonl** | Raw model outputs in JSON Lines format. |
| **raw_outputs/** | Unprocessed outputs from the LLMs in Python format. |

## Datasets

Benchmarks are run on **TorchLeet**, used here to evaluate and compare code-generation performance across the included models.

## Agents

### Setup

Set up the Python environment and build the sandbox image:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker build -t jax-sandbox:latest -f agent/docker/Dockerfile.jax .
```

### Model Clients

All three backends share the same interface — swap the backend string and the rest of your code stays the same.

#### GGUF Client (local quantised models via llama-cpp-python)

For running `.gguf` model files locally on CPU or GPU.

```python
from agent.clients import get_client

client = get_client(
    "gguf",
    model_repo="TheBloke/CodeLlama-7B-Instruct-GGUF",   # HuggingFace repo ID
    model_file="codellama-7b-instruct.Q8_0.gguf",        # GGUF filename
    n_gpu_layers=-1,   # -1 = offload all layers to GPU, 0 = CPU only
    n_ctx=4096,        # context window size
    n_batch=512,       # batch size for prompt evaluation
)
```

The model file is auto-downloaded from HuggingFace Hub on first use and cached locally.

**Requires:** `llama-cpp-python`, `huggingface-hub`

#### HF Client (HuggingFace transformers)

For loading models natively via `transformers` (e.g. CodeGemma), with optional quantisation.

```python
import torch
from agent.clients import get_client

client = get_client(
    "hf",
    model_name="google/codegemma-7b-it",   # HuggingFace model ID
    device_map="auto",                       # auto GPU/CPU placement
    torch_dtype=torch.float16,               # weight precision
    # load_in_4bit=True,                     # optional: 4-bit quantisation
    # load_in_8bit=True,                     # optional: 8-bit quantisation
)
```

**Requires:** `transformers`, `torch`, `accelerate`, `bitsandbytes` (if quantising)

#### API Client (OpenAI-compatible endpoints)

For remote or local servers that speak the OpenAI API format (vLLM, Ollama, LM Studio, Together AI, OpenRouter, etc.).

```python
from agent.clients import get_client

# Completions endpoint (e.g. vLLM serving a local model)
client = get_client(
    "api",
    base_url="http://localhost:8000/v1",
    model="codellama-7b-instruct",
    api_key="EMPTY",       # local servers typically don't need a key
)

# Chat endpoint (e.g. OpenRouter, Together AI)
client = get_client(
    "api",
    base_url="https://openrouter.ai/api/v1",
    model="meta-llama/codellama-7b-instruct",
    api_key="sk-your-key-here",
    use_chat=True,         # use chat.completions instead of completions
)
```

**Requires:** `openai`

### Generating a Translation

Once you have a client, usage is the same regardless of backend:

```python
pytorch_code = open("datasets/torchleet/basic/pytorch_basic_input/basic_custom_activation.py").read()

# Use the default prompt template (the root `prompt` file)
jax_code = client.generate_translation(pytorch_code)

# Or use a custom template
jax_code = client.generate_translation(pytorch_code, prompt_template="Convert:\n{torch_code}\nJAX:")
```

### Executing Generated Code

Run generated JAX code inside a sandboxed Docker container:

```python
from agent.sandbox import DockerSandbox
from agent.agents.execution_agent import ExecutionAgent

sandbox = DockerSandbox(image="jax-sandbox:latest", timeout=120)
agent = ExecutionAgent(sandbox)
result = agent.execute(jax_code)

print(result.success)    # True if exit code == 0
print(result.stdout)     # captured stdout
print(result.stderr)     # captured stderr
```

### Translation, debug, and static validation agents

Higher-level wrappers around `ModelClient` and the root `prompt` file:

| Component | Role |
|-----------|------|
| **`TranslationAgent`** | Fills the Mistral-style `prompt` template with `{torch_code}`, calls the client, strips markdown fences. |
| **`DebugAgent`** | Takes failing JAX code plus sandbox `stderr`/`stdout` and asks the model for a fixed script. |
| **`StaticValidationAgent`** | No LLM: sanitizes common issues (BOM, CRLF, fences) and checks syntax with `ast.parse` + `compile`. Warns if `import torch` / `from torch` remains. |

Shared fence stripping lives in `agent/agents/code_utils.py`.

```python
from agent.clients import get_client
from agent.agents import TranslationAgent, DebugAgent, StaticValidationAgent
from agent.agents.execution_agent import ExecutionAgent
from agent.sandbox import DockerSandbox

client = get_client("api", base_url="http://localhost:8000/v1", model="your-model")

torch_src = open("path/to/pytorch.py", encoding="utf-8").read()
jax_code = TranslationAgent(client).translate(torch_src)

val = StaticValidationAgent().check(jax_code)
if not val.ok:
    print(val.syntax_error, val.fixes_applied)
jax_code = val.code  # use sanitized source

sandbox = DockerSandbox(image="jax-sandbox:latest", timeout=120)
run = ExecutionAgent(sandbox).execute(jax_code)
if not run.success:
    jax_code = DebugAgent(client).fix(jax_code, run.stderr, run.stdout)
```

Import note: importing `agent.agents` runs `agent/agents/__init__.py`, which loads `ExecutionAgent` and therefore requires the **`docker` Python package** (`pip install docker` from `requirements.txt`). That is separate from the Docker **daemon**: you need the library for imports to succeed; you only need Docker Desktop (or Linux Docker) running for `DockerSandbox` / `ExecutionAgent`.

```python
from agent.agents.static_validation_agent import StaticValidationAgent, ValidationResult
```

### Translation Pipeline

`TranslationPipeline` orchestrates the full translate-validate-execute loop with automated debug retries:

```
TranslationAgent → StaticValidationAgent ──ok──→ ExecutionAgent ──ok──→ output
                        │ fail                        │ fail
                        ▼                             ▼
                    DebugAgent ──→ re-validate     DebugAgent ──→ re-validate ──→ re-execute
                   (max 5 retries)                (max 5 retries)
```

```python
from agent.clients import get_client
from agent.sandbox import DockerSandbox
from agent.loop import TranslationPipeline

client = get_client("api", base_url="http://localhost:8000/v1", model="codellama")
sandbox = DockerSandbox(image="jax-sandbox:latest", timeout=120)

pipeline = TranslationPipeline(
    client=client,
    sandbox=sandbox,
    max_retries=5,          # max debug attempts per phase
    execution_timeout=120,  # Docker timeout in seconds
    verbose=True,           # print step-by-step progress to stderr
)

torch_code = open("path/to/pytorch_script.py").read()
result = pipeline.run(torch_code)

print(result.status)          # "success" | "validation_failed" | "execution_failed"
print(result.jax_code)        # final/best JAX code produced
print(result.syntax_retries)  # number of syntax-fix attempts used
print(result.runtime_retries) # number of runtime-fix attempts used
print(result.history)         # step-by-step log of every phase
```

`PipelineResult` fields:

| Field | Type | Description |
|-------|------|-------------|
| `jax_code` | `str` | Final or best JAX code produced |
| `status` | `str` | `"success"`, `"validation_failed"`, or `"execution_failed"` |
| `validated` | `bool` | Whether the code passed static validation |
| `executed` | `bool` | Whether the code ran successfully in Docker |
| `syntax_retries` | `int` | Total syntax-fix attempts used |
| `runtime_retries` | `int` | Total runtime-fix attempts used |
| `execution_result` | `ExecutionResult \| None` | Last execution result (if reached) |
| `validation_result` | `ValidationResult \| None` | Last validation result |
| `history` | `list[dict]` | Log of every step: phase, results, retries |

### CLI

The pipeline can be run from the command line via `python -m agent`.

#### Single file

```bash
python -m agent \
  --input datasets/torchleet/basic/pytorch_basic_input/basic_lin_regression.py \
  --output outputs/jax_basic_lin_regression.py \
  --backend gguf \
  --model-repo TheBloke/CodeLlama-7B-Instruct-GGUF \
  --model-file codellama-7b-instruct.Q8_0.gguf \
  --verbose
```

#### Batch directory

```bash
python -m agent \
  --input datasets/torchleet/basic/pytorch_basic_input/ \
  --output outputs/basic/ \
  --backend api \
  --base-url http://localhost:8000/v1 \
  --model codellama \
  --max-retries 5 \
  --timeout 120 \
  --summary
```

#### All options

```
usage: agent.cli [-h] --input INPUT --output OUTPUT --backend {gguf,hf,api}
                 [backend-specific options] [pipeline options]

Required:
  --input, -i       Path to a .py file or directory of .py files
  --output, -o      Output .py file (single) or directory (batch)
  --backend, -b     gguf | hf | api

GGUF backend:
  --model-repo      HuggingFace repo ID for the GGUF model
  --model-file      GGUF filename inside the repo
  --n-gpu-layers    GPU layers to offload (-1 = all, default: -1)
  --n-ctx           Context window size (default: 4096)
  --n-batch         Prompt eval batch size (default: 512)

HuggingFace backend:
  --model-name      HuggingFace model ID
  --device-map      Device placement strategy (default: auto)
  --load-in-4bit    Enable 4-bit quantisation
  --load-in-8bit    Enable 8-bit quantisation

API backend:
  --base-url        Server URL (e.g. http://localhost:8000/v1)
  --model           Model name the server expects
  --api-key         API key (default: EMPTY)
  --use-chat        Use chat completions endpoint

Pipeline:
  --max-retries     Max debug retries per phase (default: 5)
  --timeout         Docker execution timeout in seconds (default: 120)
  --docker-image    Docker image for sandbox (default: jax-sandbox:latest)

Output:
  --verbose, -v     Print step-by-step progress to stderr
  --summary         Print summary table after batch runs
```

#### Output files

- Each translation writes a `.py` file to the output path.
- A `_results.jsonl` file is written alongside with one JSON line per input:

```json
{"input": "basic_lin_regression.py", "status": "success", "validated": true, "executed": true, "syntax_retries": 1, "runtime_retries": 0, "duration": 45.2}
```
