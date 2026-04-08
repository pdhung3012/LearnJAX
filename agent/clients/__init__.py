from __future__ import annotations

from .base import ModelClient


def get_client(backend: str, **kwargs) -> ModelClient:
    """Instantiate a :class:`ModelClient` for the requested *backend*.

    Parameters
    ----------
    backend:
        One of ``"gguf"``, ``"hf"``, or ``"api"``.
    **kwargs:
        Forwarded to the chosen client's constructor.

    Returns
    -------
    ModelClient
        A ready-to-use client instance.
    """
    if backend == "gguf":
        from .gguf_client import GGUFClient

        return GGUFClient(**kwargs)

    if backend == "hf":
        from .hf_client import HFClient

        return HFClient(**kwargs)

    if backend == "api":
        from .api_client import APIClient

        return APIClient(**kwargs)

    raise ValueError(
        f"Unknown backend {backend!r}. Choose from 'gguf', 'hf', or 'api'."
    )


__all__ = ["ModelClient", "get_client"]
