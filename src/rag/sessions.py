"""Long-term memory of played games: the ``bg_sessions`` collection.

One point per closed round. The robot writes a short English summary at the end of every
round; later it can ask "what happened the last time we faced Azathoth?" and get the closest
summaries back.

    from src.rag import sessions
    sessions.record_round(session_id="20260914-2030", round_number=3,
                          ancient_one="Cthulhu", investigators=["Lily Chen", "Mark Harrigan"],
                          summary="Lily closed the gate in Shanghai; Mark lost 2 Health to a Deep One ...")
    sessions.recall("gate in Shanghai", ancient_one="Cthulhu")
"""

from __future__ import annotations

import time
from typing import Any

from qdrant_client import QdrantClient

from src.rag.collections import GAME_ID, SESSIONS
from src.rag.embeddings import embed_text
from src.rag.store import build_filter, ensure_collection, get_client, point_id, search, upsert


def session_text(session_id: str, ancient_one: str | None, round_number: int, summary: str) -> str:
    """The string that gets embedded; keeps the search-relevant context in front."""
    opponent = ancient_one or "an unknown Ancient One"
    return f"Session {session_id} vs {opponent}, round {round_number}: {summary.strip()}"


def record_round(
    *,
    session_id: str,
    round_number: int,
    summary: str,
    ancient_one: str | None = None,
    investigators: list[str] | None = None,
    client: QdrantClient | None = None,
) -> dict[str, Any]:
    """Embed and store one round summary. Returns the stored payload."""
    summary = " ".join(summary.split())
    if not summary:
        raise ValueError("summary must not be empty")
    client = client or get_client()
    ensure_collection(client, SESSIONS)
    payload = {
        "game_id": GAME_ID,
        "session_id": session_id,
        "round": int(round_number),
        "ancient_one": ancient_one,
        "investigators": list(investigators or []),
        "summary": summary,
        "text": session_text(session_id, ancient_one, round_number, summary),
        "recorded_at": time.time(),
    }
    vector = embed_text(payload["text"])
    upsert(client, SESSIONS, [point_id(GAME_ID, session_id, round_number)], [vector], [payload])
    return payload


def recall(
    query: str,
    *,
    limit: int = 5,
    session_id: str | None = None,
    ancient_one: str | None = None,
    client: QdrantClient | None = None,
) -> list[dict[str, Any]]:
    """Round summaries most similar to ``query``, optionally restricted to one session or Ancient One."""
    client = client or get_client()
    if not client.collection_exists(SESSIONS):
        return []
    filters = build_filter(game_id=GAME_ID, session_id=session_id, ancient_one=ancient_one)
    hits = search(client, SESSIONS, embed_text(query), limit=limit, filters=filters)
    return [{"score": h["score"], **h["payload"]} for h in hits]


__all__ = ["record_round", "recall", "session_text"]
