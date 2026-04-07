from __future__ import annotations

from openai import OpenAI

from .base import ModelClient


class APIClient(ModelClient):
    """Inference backend for OpenAI-compatible API endpoints.

    Works with vLLM, Ollama, LM Studio, Together AI, OpenRouter, and any
    server that exposes an ``/v1/completions`` or ``/v1/chat/completions``
    endpoint.

    Parameters
    ----------
    base_url:
        Server URL (e.g. ``http://localhost:8000/v1``).
    model:
        Model name the server expects.
    api_key:
        Optional API key (use ``"EMPTY"`` for local servers that don't
        require one).
    use_chat:
        If ``True`` use ``chat.completions.create`` instead of the plain
        ``completions.create`` endpoint.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        use_chat: bool = False,
    ) -> None:
        self.model = model
        self._use_chat = use_chat
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        **kwargs,
    ) -> str:
        if self._use_chat:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            return response.choices[0].message.content or ""

        response = self._client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].text
