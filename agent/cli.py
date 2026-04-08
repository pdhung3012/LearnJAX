#!/usr/bin/env python3
"""Entry point for the PyTorch-to-JAX translation pipeline.

Usage examples
--------------
Single file (GGUF backend)::

    python -m agent.cli \\
        --input datasets/torchleet/basic/pytorch_basic_input/basic_lin_regression.py \\
        --output outputs/jax_basic_lin_regression.py \\
        --backend gguf \\
        --model-repo TheBloke/CodeLlama-7B-Instruct-GGUF \\
        --model-file codellama-7b-instruct.Q8_0.gguf

Batch directory (API backend)::

    python -m agent.cli \\
        --input datasets/torchleet/basic/pytorch_basic_input/ \\
        --output outputs/basic/ \\
        --backend api \\
        --base-url http://localhost:8000/v1 \\
        --model codellama
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent.cli",
        description="Translate PyTorch scripts to JAX/Flax with automated debug retries.",
    )

    # ── I/O ───────────────────────────────────────────────────────────
    p.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Path to a .py file or directory of .py files.",
    )
    p.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Output .py file (single mode) or directory (batch mode).",
    )

    # ── Backend selection ─────────────────────────────────────────────
    p.add_argument(
        "--backend", "-b", required=True, choices=["gguf", "hf", "api"],
        help="Model inference backend.",
    )

    # ── GGUF-specific ─────────────────────────────────────────────────
    gguf = p.add_argument_group("GGUF backend options")
    gguf.add_argument("--model-repo", help="HuggingFace repo ID for the GGUF model.")
    gguf.add_argument("--model-file", help="GGUF filename inside the repo.")
    gguf.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers (-1 = all).")
    gguf.add_argument("--n-ctx", type=int, default=4096, help="Context window size.")
    gguf.add_argument("--n-batch", type=int, default=512, help="Prompt eval batch size.")

    # ── HF-specific ───────────────────────────────────────────────────
    hf = p.add_argument_group("HuggingFace transformers backend options")
    hf.add_argument("--model-name", help="HuggingFace model ID.")
    hf.add_argument("--device-map", default="auto", help="Device placement strategy.")
    hf.add_argument("--load-in-4bit", action="store_true", help="4-bit quantisation.")
    hf.add_argument("--load-in-8bit", action="store_true", help="8-bit quantisation.")

    # ── API-specific ──────────────────────────────────────────────────
    api = p.add_argument_group("OpenAI-compatible API backend options")
    api.add_argument("--base-url", help="Server URL (e.g. http://localhost:8000/v1).")
    api.add_argument("--api-key", default=None, help="API key (falls back to .env / EMPTY).")
    api.add_argument("--model", dest="api_model", help="Model name the server expects.")
    api.add_argument("--use-chat", action="store_true", help="Use chat completions endpoint.")

    # ── Pipeline ──────────────────────────────────────────────────────
    pipe = p.add_argument_group("Pipeline options")
    pipe.add_argument("--max-retries", type=int, default=5, help="Max debug retries per phase.")
    pipe.add_argument("--timeout", type=int, default=120, help="Docker execution timeout (seconds).")
    pipe.add_argument("--docker-image", default="jax-sandbox:latest", help="Docker image for sandbox.")

    # ── Output control ────────────────────────────────────────────────
    p.add_argument("--verbose", "-v", action="store_true", help="Print step-by-step progress to stderr.")
    p.add_argument("--summary", action="store_true", help="Print summary table after batch runs.")

    return p


def _build_client(args: argparse.Namespace):
    """Instantiate the correct ModelClient from parsed CLI args."""
    from .clients import get_client

    if args.backend == "gguf":
        if not args.model_repo or not args.model_file:
            sys.exit("Error: --model-repo and --model-file are required for the gguf backend.")
        return get_client(
            "gguf",
            model_repo=args.model_repo,
            model_file=args.model_file,
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx,
            n_batch=args.n_batch,
        )

    if args.backend == "hf":
        if not args.model_name:
            sys.exit("Error: --model-name is required for the hf backend.")
        return get_client(
            "hf",
            model_name=args.model_name,
            device_map=args.device_map,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )

    if args.backend == "api":
        if not args.base_url or not args.api_model:
            sys.exit("Error: --base-url and --model are required for the api backend.")
        return get_client(
            "api",
            base_url=args.base_url,
            model=args.api_model,
            api_key=args.api_key,
            use_chat=args.use_chat,
        )

    sys.exit(f"Error: unknown backend {args.backend!r}")


def _collect_inputs(input_path: Path) -> list[Path]:
    """Return a sorted list of .py files from *input_path* (file or directory)."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.py"))
        if not files:
            sys.exit(f"Error: no .py files found in {input_path}")
        return files
    sys.exit(f"Error: {input_path} is neither a file nor a directory.")


def _run_single(
    pipeline,
    input_path: Path,
    output_path: Path,
    verbose: bool,
) -> dict:
    """Translate one file. Returns a summary dict for the results JSONL."""
    torch_code = input_path.read_text(encoding="utf-8")

    if verbose:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Processing: {input_path.name}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

    t0 = time.monotonic()
    result = pipeline.run(torch_code)
    duration = round(time.monotonic() - t0, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.jax_code, encoding="utf-8")

    summary = {
        "input": input_path.name,
        "status": result.status,
        "validated": result.validated,
        "executed": result.executed,
        "syntax_retries": result.syntax_retries,
        "runtime_retries": result.runtime_retries,
        "duration": duration,
    }

    if verbose:
        status_tag = "OK" if result.status == "success" else result.status.upper()
        print(
            f"  -> [{status_tag}] syntax_retries={result.syntax_retries} "
            f"runtime_retries={result.runtime_retries} ({duration}s)",
            file=sys.stderr,
        )

    return summary


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the pipeline."""
    args = _build_parser().parse_args(argv)

    client = _build_client(args)

    from .sandbox import DockerSandbox
    from .loop import TranslationPipeline

    sandbox = DockerSandbox(
        image=args.docker_image,
        timeout=args.timeout,
    )
    pipeline = TranslationPipeline(
        client=client,
        sandbox=sandbox,
        max_retries=args.max_retries,
        execution_timeout=args.timeout,
        verbose=args.verbose,
    )

    inputs = _collect_inputs(args.input)
    is_batch = len(inputs) > 1 or args.input.is_dir()

    if is_batch:
        args.output.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []

    for src in inputs:
        if is_batch:
            dst = args.output / src.name
        else:
            dst = args.output

        summary = _run_single(pipeline, src, dst, args.verbose)
        summaries.append(summary)

    # ── Write results JSONL ───────────────────────────────────────────
    if is_batch:
        results_path = args.output / "_results.jsonl"
    else:
        results_path = args.output.with_name(args.output.stem + "_results.jsonl")

    with open(results_path, "w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ── Summary table ─────────────────────────────────────────────────
    if args.summary or (is_batch and args.verbose):
        total = len(summaries)
        ok = sum(1 for s in summaries if s["status"] == "success")
        val_fail = sum(1 for s in summaries if s["status"] == "validation_failed")
        exec_fail = sum(1 for s in summaries if s["status"] == "execution_failed")

        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  Total: {total}  |  Success: {ok}  |  "
              f"Val-fail: {val_fail}  |  Exec-fail: {exec_fail}", file=sys.stderr)
        print(f"  Results saved to: {results_path}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
