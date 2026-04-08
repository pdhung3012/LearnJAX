from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import docker
from docker.errors import ContainerError, ImageNotFound


@dataclass
class SandboxResult:
    """Raw output returned by :meth:`DockerSandbox.run`."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_seconds: float


class DockerSandbox:
    """Manage Docker container lifecycle for running untrusted Python code.

    Parameters
    ----------
    image:
        Docker image to use (should have JAX pre-installed).
    timeout:
        Maximum wall-clock seconds before the container is killed.
    mem_limit:
        Memory cap (Docker format, e.g. ``"4g"``).
    network_disabled:
        If ``True`` the container has no network access.
    """

    def __init__(
        self,
        image: str = "jax-sandbox:latest",
        timeout: int = 120,
        mem_limit: str = "4g",
        network_disabled: bool = True,
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.mem_limit = mem_limit
        self.network_disabled = network_disabled
        self._docker = docker.from_env()

    def _ensure_image(self) -> None:
        try:
            self._docker.images.get(self.image)
        except ImageNotFound:
            raise ImageNotFound(
                f"Docker image {self.image!r} not found. "
                f"Build it first with: docker build -t {self.image} "
                f"-f agent/docker/Dockerfile.jax ."
            )

    def run(self, code: str, timeout: int | None = None) -> SandboxResult:
        """Write *code* to a temp file, mount it into a container, and execute.

        Returns a :class:`SandboxResult` with captured stdout/stderr, exit
        code, timeout flag, and wall-clock duration.
        """
        self._ensure_image()
        timeout = timeout or self.timeout

        tmp = tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        )
        try:
            tmp.write(code)
            tmp.flush()
            tmp.close()
            host_path = Path(tmp.name).resolve()

            container = self._docker.containers.run(
                self.image,
                command=["/app/script.py"],
                volumes={str(host_path): {"bind": "/app/script.py", "mode": "ro"}},
                mem_limit=self.mem_limit,
                network_disabled=self.network_disabled,
                detach=True,
                stderr=True,
            )

            start = time.monotonic()
            timed_out = False
            try:
                result = container.wait(timeout=timeout)
                exit_code = result.get("StatusCode", -1)
            except Exception:
                timed_out = True
                container.kill()
                exit_code = -1
            duration = time.monotonic() - start

            stdout = container.logs(stdout=True, stderr=False).decode(
                errors="replace"
            )
            stderr = container.logs(stdout=False, stderr=True).decode(
                errors="replace"
            )

            container.remove(force=True)

            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                timed_out=timed_out,
                duration_seconds=round(duration, 3),
            )
        finally:
            Path(tmp.name).unlink(missing_ok=True)
