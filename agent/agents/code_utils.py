from __future__ import annotations


def strip_markdown_fences(text: str) -> str:
    """Remove common markdown code fences from model output (see COMS5990_Task1)."""
    return text.replace("```python", "").replace("```", "").strip()
