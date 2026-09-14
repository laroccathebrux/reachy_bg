"""Text embeddings through the local Ollama server (bge-m3 by default)."""

from __future__ import annotations

from collections.abc import Iterable

import httpx

from src.config import EMBED_MODEL, OLLAMA_BASE_URL

_BATCH = 32


class EmbeddingError(RuntimeError):
    """Raised when Ollama cannot produce embeddings."""


def embed_texts(
    texts: Iterable[str],
    *,
    model: str = EMBED_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: float = 300.0,
) -> list[list[float]]:
    """Embed every text, preserving order. Empty input returns an empty list."""
    items = [t if t.strip() else " " for t in texts]
    if not items:
        return []
    out: list[list[float]] = []
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        for start in range(0, len(items), _BATCH):
            batch = items[start : start + _BATCH]
            try:
                response = client.post("/api/embed", json={"model": model, "input": batch})
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"Ollama embedding request failed: {exc}") from exc
            vectors = response.json().get("embeddings")
            if not vectors or len(vectors) != len(batch):
                raise EmbeddingError("Ollama returned an unexpected number of embeddings")
            out.extend(vectors)
    return out


def embed_text(text: str, **kwargs) -> list[float]:
    """Embed a single string."""
    return embed_texts([text], **kwargs)[0]
