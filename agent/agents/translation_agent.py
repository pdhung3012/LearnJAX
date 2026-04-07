from __future__ import annotations

from pathlib import Path

from ..clients.base import ModelClient
from .code_utils import strip_markdown_fences

_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompt"


class TranslationAgent:
    """Convert PyTorch source to JAX/Flax using a :class:`ModelClient`.

    Uses the project root ``prompt`` file (Mistral-style ``[INST]`` template with
    ``{torch_code}``) unless a custom *prompt_template* is supplied.
    """

    def __init__(self, client: ModelClient) -> None:
        self._client = client

    def translate(
        self,
        torch_code: str,
        *,
        prompt_template: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        strip_fences: bool = True,
        **kwargs,
    ) -> str:
        """Return JAX/Flax translation for *torch_code*.

        Parameters
        ----------
        torch_code:
            PyTorch script or snippet to convert.
        prompt_template:
            Full prompt with a ``{torch_code}`` placeholder. If ``None``, the
            default project ``prompt`` file is loaded.
        max_tokens, temperature:
            Passed through to :meth:`ModelClient.generate`.
        strip_fences:
            If ``True``, strip `` ```python` `` / `` ``` `` wrappers from the
            model output.
        """
        if prompt_template is None:
            prompt_template = _DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
        filled = prompt_template.replace("{torch_code}", torch_code)
        raw = self._client.generate(
            filled,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return strip_markdown_fences(raw) if strip_fences else raw
