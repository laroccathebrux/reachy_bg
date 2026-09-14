"""Retrieve reference passages for a question from the rules and knowledge collections."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from src.rag.collections import GAME_ID, KNOWLEDGE, RULES
from src.rag.embeddings import embed_text
from src.rag.store import build_filter, get_client, search


def retrieve(
    question: str,
    *,
    rules_limit: int = 5,
    knowledge_limit: int = 3,
    base_game_only: bool = True,
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]:
    """Top passages from ``bg_rules`` and ``bg_knowledge`` for ``question``, best first.

    One embedding call serves both collections. Each returned dict is the stored payload
    plus ``score`` and ``collection``.
    """
    client = client or get_client()
    vector = embed_text(question)
    hits: list[dict[str, Any]] = []
    for name, limit, filters in (
        (RULES, rules_limit, build_filter(game_id=GAME_ID)),
        (
            KNOWLEDGE,
            knowledge_limit,
            build_filter(game_id=GAME_ID, base_game=True if base_game_only else None),
        ),
    ):
        if limit <= 0 or not client.collection_exists(name):
            continue
        for hit in search(client, name, vector, limit=limit, filters=filters):
            hits.append({**hit["payload"], "score": hit["score"], "collection": name})
    hits.sort(key=lambda h: -h["score"])
    return hits


__all__ = ["retrieve"]
