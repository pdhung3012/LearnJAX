from __future__ import annotations

from ..clients.base import ModelClient
from .code_utils import strip_markdown_fences

_DEBUG_PROMPT = """[INST] You are an expert Machine Learning Engineer specializing in JAX, Flax, and Optax.
The following JAX code was translated from PyTorch but did not run successfully. Fix it so the script is complete, runnable, and preserves the intended behavior of the original PyTorch code.
Use jax, jax.numpy (as jnp), flax.linen (as nn), and optax as appropriate. Output RAW Python code only. Do NOT use markdown backticks (```).

ORIGINAL PYTORCH CODE:
{torch_code}

EXECUTION STDERR:
{stderr}

EXECUTION STDOUT:
{stdout}

CURRENT JAX CODE:
{jax_code}

FIXED JAX CODE:
[/INST]"""


class DebugAgent:
    """Ask the model to repair JAX code using sandbox stderr/stdout."""

    def __init__(self, client: ModelClient) -> None:
        self._client = client

    def fix(
        self,
        jax_code: str,
        stderr: str,
        stdout: str = "",
        torch_code: str = "",
        *,
        prompt_template: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        strip_fences: bool = True,
        **kwargs,
    ) -> str:
        """Return a revised JAX script that should address the execution failure.

        Parameters
        ----------
        jax_code:
            The JAX code that was run in the sandbox.
        stderr, stdout:
            Captured streams from :class:`ExecutionAgent` / :class:`DockerSandbox`.
        torch_code:
            The original PyTorch source so the model understands intended behavior.
        prompt_template:
            Template with ``{jax_code}``, ``{stderr}``, ``{stdout}``, and
            ``{torch_code}`` placeholders.  If ``None``, a built-in template is used.
        """
        if prompt_template is None:
            prompt_template = _DEBUG_PROMPT
        filled = (
            prompt_template.replace("{jax_code}", jax_code)
            .replace("{stderr}", stderr)
            .replace("{stdout}", stdout)
            .replace("{torch_code}", torch_code)
        )
        raw = self._client.generate(
            filled,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return strip_markdown_fences(raw) if strip_fences else raw
