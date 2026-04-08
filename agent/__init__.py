def __getattr__(name: str):
    """Lazy imports so subpackages can be used independently."""
    if name in ("ExecutionAgent", "ExecutionResult"):
        from .agents import ExecutionAgent, ExecutionResult

        return {"ExecutionAgent": ExecutionAgent, "ExecutionResult": ExecutionResult}[name]
    if name == "DockerSandbox":
        from .sandbox import DockerSandbox

        return DockerSandbox
    if name == "ModelClient":
        from .clients import ModelClient

        return ModelClient
    if name == "get_client":
        from .clients import get_client

        return get_client
    if name == "TranslationPipeline":
        from .loop import TranslationPipeline

        return TranslationPipeline
    if name == "PipelineResult":
        from .loop import PipelineResult

        return PipelineResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DockerSandbox",
    "ExecutionAgent",
    "ExecutionResult",
    "ModelClient",
    "PipelineResult",
    "TranslationPipeline",
    "get_client",
]
