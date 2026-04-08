from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from .base import ModelClient


class GGUFClient(ModelClient):
    """Inference backend for GGUF models via *llama-cpp-python*.

    Parameters
    ----------
    model_repo:
        HuggingFace repo ID (e.g. ``TheBloke/CodeLlama-13B-Instruct-GGUF``).
    model_file:
        GGUF filename inside the repo
        (e.g. ``codellama-13b-instruct.Q8_0.gguf``).
    n_gpu_layers:
        Layers to offload to GPU (``-1`` = all).
    n_ctx:
        Context window size in tokens.
    n_batch:
        Batch size for prompt evaluation.
    local_dir:
        Optional local cache directory for the downloaded model file.
    """

    def __init__(
        self,
        model_repo: str,
        model_file: str,
        *,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        n_batch: int = 512,
        local_dir: str | Path | None = None,
    ) -> None:
        self.model_repo = model_repo
        self.model_file = model_file

        download_kwargs: dict = {
            "repo_id": model_repo,
            "filename": model_file,
        }
        if local_dir is not None:
            download_kwargs["local_dir"] = str(local_dir)

        model_path = hf_hub_download(**download_kwargs)

        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        **kwargs,
    ) -> str:
        response = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return response["choices"][0]["text"]
