from __future__ import annotations

from dataclasses import dataclass

from ..sandbox import DockerSandbox, SandboxResult


@dataclass
class ExecutionResult:
    """Structured result from running JAX code in the sandbox."""

    success: bool
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_seconds: float


class ExecutionAgent:
    """Run JAX code inside a :class:`DockerSandbox` and return structured results.

    Parameters
    ----------
    sandbox:
        A configured :class:`DockerSandbox` instance.
    timeout:
        Per-execution timeout in seconds (overrides the sandbox default).
    """

    def __init__(
        self,
        sandbox: DockerSandbox,
        timeout: int = 120,
    ) -> None:
        self._sandbox = sandbox
        self._timeout = timeout

    def execute(self, code: str) -> ExecutionResult:
        """Run *code* in the sandbox and return an :class:`ExecutionResult`."""
        raw: SandboxResult = self._sandbox.run(code, timeout=self._timeout)
        return ExecutionResult(
            success=raw.exit_code == 0,
            stdout=raw.stdout,
            stderr=raw.stderr,
            exit_code=raw.exit_code,
            timed_out=raw.timed_out,
            duration_seconds=raw.duration_seconds,
        )
