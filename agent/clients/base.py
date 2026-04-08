from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompt"


class ModelClient(ABC):
    """Backend-agnostic interface that every inference client implements."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        **kwargs,
    ) -> str:
        """Send *prompt* to the model and return the generated text."""

    def generate_translation(
        self,
        torch_code: str,
        prompt_template: str | None = None,
    ) -> str:
        """Fill a prompt template with *torch_code* and call :meth:`generate`.

        If *prompt_template* is ``None`` the project-level ``prompt`` file is
        loaded automatically.
        """
        if prompt_template is None:
            prompt_template = _DEFAULT_PROMPT_PATH.read_text()
        filled = prompt_template.replace("{torch_code}", torch_code)
        return self.generate(filled)
