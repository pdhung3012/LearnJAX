from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from .clients.base import ModelClient
from .sandbox import DockerSandbox
from .agents.translation_agent import TranslationAgent
from .agents.static_validation_agent import StaticValidationAgent, ValidationResult
from .agents.execution_agent import ExecutionAgent, ExecutionResult
from .agents.debug_agent import DebugAgent


@dataclass
class PipelineResult:
    """Full outcome of a single PyTorch-to-JAX translation attempt."""

    jax_code: str
    status: str  # "success" | "validation_failed" | "execution_failed"
    validated: bool
    executed: bool
    syntax_retries: int
    runtime_retries: int
    execution_result: ExecutionResult | None = None
    validation_result: ValidationResult | None = None
    history: list[dict] = field(default_factory=list)


class TranslationPipeline:
    """Two-phase retry pipeline: translate -> validate -> execute.

    Phase 1 (syntax): translate, then validate; on failure ask the
    DebugAgent to fix and re-validate, up to *max_retries* times.

    Phase 2 (runtime): execute in Docker; on failure debug and retry,
    re-validating each patch before running.  Up to *max_retries* times.
    """

    def __init__(
        self,
        client: ModelClient,
        sandbox: DockerSandbox,
        max_retries: int = 5,
        execution_timeout: int = 120,
        verbose: bool = False,
    ) -> None:
        self.translator = TranslationAgent(client)
        self.validator = StaticValidationAgent()
        self.executor = ExecutionAgent(sandbox, timeout=execution_timeout)
        self.debugger = DebugAgent(client)
        self.max_retries = max_retries
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[pipeline] {msg}", file=sys.stderr)

    def _validate_with_retries(
        self,
        jax_code: str,
        history: list[dict],
        syntax_retries: int,
        budget: int,
    ) -> tuple[str, ValidationResult, int]:
        """Run validation loop, returning (code, result, updated_syntax_retries).

        Attempts up to *budget* debug-fix cycles if validation fails.
        """
        for _ in range(budget + 1):  # first attempt + budget retries
            vr = self.validator.check(jax_code)
            history.append({
                "phase": "validate",
                "ok": vr.ok,
                "syntax_error": vr.syntax_error,
                "warnings": vr.warnings,
                "fixes_applied": vr.fixes_applied,
            })
            self._log(
                f"validate: ok={vr.ok}"
                + (f"  error={vr.syntax_error}" if vr.syntax_error else "")
            )

            if vr.ok:
                jax_code = vr.code
                return jax_code, vr, syntax_retries

            if syntax_retries >= self.max_retries:
                return jax_code, vr, syntax_retries

            syntax_retries += 1
            self._log(f"debug_syntax: retry {syntax_retries}/{self.max_retries}")
            jax_code = self.debugger.fix(
                jax_code, stderr=vr.syntax_error or "", stdout=""
            )
            history.append({"phase": "debug_syntax", "retry": syntax_retries})

        return jax_code, vr, syntax_retries

    def run(self, torch_code: str, **translate_kwargs) -> PipelineResult:
        """Execute the full translation pipeline and return a :class:`PipelineResult`."""
        history: list[dict] = []
        syntax_retries = 0
        runtime_retries = 0

        # ── Step 1: translate ─────────────────────────────────────────
        self._log("translate: starting")
        jax_code = self.translator.translate(torch_code, **translate_kwargs)
        history.append({"phase": "translate", "code_len": len(jax_code)})
        self._log(f"translate: done ({len(jax_code)} chars)")

        # ── Phase 1: validate (with syntax-debug retries) ────────────
        jax_code, vr, syntax_retries = self._validate_with_retries(
            jax_code, history, syntax_retries, budget=self.max_retries,
        )

        if not vr.ok:
            self._log("pipeline: validation failed after all retries")
            return PipelineResult(
                jax_code=jax_code,
                status="validation_failed",
                validated=False,
                executed=False,
                syntax_retries=syntax_retries,
                runtime_retries=runtime_retries,
                validation_result=vr,
                history=history,
            )

        # ── Phase 2: execute (with runtime-debug retries) ────────────
        best_code = jax_code

        for attempt in range(self.max_retries + 1):  # first run + retries
            self._log(f"execute: attempt {attempt + 1}")
            er = self.executor.execute(jax_code)
            history.append({
                "phase": "execute",
                "success": er.success,
                "exit_code": er.exit_code,
                "timed_out": er.timed_out,
                "duration": er.duration_seconds,
                "stderr_snippet": er.stderr[:200] if er.stderr else "",
            })
            self._log(
                f"execute: success={er.success}  exit_code={er.exit_code}"
                + (f"  stderr={er.stderr[:120]!r}" if er.stderr else "")
            )

            if er.success:
                self._log("pipeline: success")
                return PipelineResult(
                    jax_code=jax_code,
                    status="success",
                    validated=True,
                    executed=True,
                    syntax_retries=syntax_retries,
                    runtime_retries=runtime_retries,
                    execution_result=er,
                    validation_result=vr,
                    history=history,
                )

            if runtime_retries >= self.max_retries:
                break

            runtime_retries += 1
            self._log(f"debug_runtime: retry {runtime_retries}/{self.max_retries}")
            jax_code = self.debugger.fix(
                jax_code, stderr=er.stderr, stdout=er.stdout,
            )
            history.append({"phase": "debug_runtime", "retry": runtime_retries})

            remaining_syntax_budget = self.max_retries - syntax_retries
            jax_code, vr, syntax_retries = self._validate_with_retries(
                jax_code, history, syntax_retries, budget=remaining_syntax_budget,
            )
            if not vr.ok:
                self._log("pipeline: debug patch failed validation, skipping execution")
                continue

            best_code = jax_code

        self._log("pipeline: execution failed after all retries")
        return PipelineResult(
            jax_code=best_code,
            status="execution_failed",
            validated=True,
            executed=False,
            syntax_retries=syntax_retries,
            runtime_retries=runtime_retries,
            execution_result=er,
            validation_result=vr,
            history=history,
        )
