from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from .code_utils import strip_markdown_fences

_TORCH_IMPORT_RE = re.compile(
    r"^\s*(import\s+torch\b|from\s+torch\b)",
    re.MULTILINE,
)


@dataclass
class ValidationResult:
    """Outcome of static checks on generated Python (e.g. translated JAX code)."""

    ok: bool
    """True if the code is non-empty and syntactically valid after sanitization."""

    code: str
    """Sanitized source (BOM removed, newlines normalized, fences stripped when enabled)."""

    fixes_applied: list[str] = field(default_factory=list)
    """Human-readable list of automatic normalizations applied."""

    syntax_error: str | None = None
    """``SyntaxError`` message + location, if parsing failed."""

    warnings: list[str] = field(default_factory=list)
    """Non-fatal issues (e.g. leftover ``torch`` imports)."""


class StaticValidationAgent:
    """Fundamental Python checks: sanitization fallbacks, ``ast.parse``, and ``compile``.

    Does not execute code. Use before :class:`ExecutionAgent` to catch bad generations
    early and reduce manual cleanup.
    """

    def __init__(
        self,
        *,
        strip_fences: bool = True,
        warn_on_torch: bool = True,
    ) -> None:
        self._strip_fences = strip_fences
        self._warn_on_torch = warn_on_torch

    def sanitize(self, code: str) -> tuple[str, list[str]]:
        """Apply safe string-level fixes that models often get wrong.

        Returns
        -------
        normalized_code, fixes_applied
        """
        fixes: list[str] = []
        s = code
        if s.startswith("\ufeff"):
            s = s[1:]
            fixes.append("removed UTF-8 BOM")
        if "\r" in s:
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            fixes.append("normalized newlines (CRLF/CR → LF)")
        if self._strip_fences:
            before = s
            s = strip_markdown_fences(s)
            if s != before:
                fixes.append("stripped markdown code fences")
        return s, fixes

    def check(self, code: str) -> ValidationResult:
        """Sanitize *code*, then verify it is valid Python with ``ast`` + ``compile``."""
        normalized, fixes = self.sanitize(code)
        warnings: list[str] = []

        if not normalized.strip():
            return ValidationResult(
                ok=False,
                code=normalized,
                fixes_applied=fixes,
                syntax_error="Source is empty after sanitization.",
                warnings=warnings,
            )

        if self._warn_on_torch and _TORCH_IMPORT_RE.search(normalized):
            warnings.append(
                "Code still references PyTorch (`import torch` / `from torch`); "
                "translation may be incomplete."
            )

        try:
            ast.parse(normalized, filename="<generated>", mode="exec")
            compile(normalized, "<generated>", "exec", dont_inherit=True)
        except SyntaxError as e:
            loc = f"line {e.lineno}"
            if e.offset is not None:
                loc += f", offset {e.offset}"
            msg = f"{e.msg} ({loc})"
            if e.text:
                msg += f": {e.text.strip()!r}"
            return ValidationResult(
                ok=False,
                code=normalized,
                fixes_applied=fixes,
                syntax_error=msg,
                warnings=warnings,
            )

        return ValidationResult(
            ok=True,
            code=normalized,
            fixes_applied=fixes,
            syntax_error=None,
            warnings=warnings,
        )
