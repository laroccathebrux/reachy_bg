"""Retrieve reference passages for a question from the rules and knowledge collections."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from qdrant_client import QdrantClient

from src.logger import get_logger
from src.rag.collections import GAME_ID, KNOWLEDGE, RULES
from src.rag.embeddings import embed_text
from src.rag.store import build_filter, get_client, search

log = get_logger(__name__)


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


__all__ = ["advice", "card", "cards_in_reserve", "retrieve"]


def card(name: str, *, client: Any = None) -> dict[str, Any] | None:
    """One card by its exact name, without going near the embeddings.

    "What does Bull Whip do?" is the commonest question about a card, and a vector search is
    the wrong tool for it: asked that way it returned Vatican Missionary at 0.32, because a
    card's *name* says almost nothing about its meaning while its effect text dominates the
    embedding. The name is a keyword-indexed payload field, so it is looked up as one.
    """
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    from src.strategy.reference import _key  # the same folding used for investigator names

    client = client or get_client()
    wanted = _key(name)
    for kind in ("asset", "condition", "task"):
        points, _ = client.scroll(
            KNOWLEDGE,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="game_id", match=MatchValue(value=GAME_ID)),
                    FieldCondition(key="kind", match=MatchValue(value=kind)),
                ]
            ),
            limit=500,
            with_payload=True,
        )
        for point in points:
            payload = dict(point.payload or {})
            if _key(str(payload.get("name", ""))) == wanted:
                return payload
    return None


def advice(question: str, *, limit: int = 3, client: Any = None) -> list[dict[str, Any]]:
    """Community strategy entries for a situation - player advice, never rules.

    These carry a source and are phrased as opinion (``kind="strategy"`` in ``bg_knowledge``).
    They are kept in their own lookup, and labelled as advice wherever they are used, because a
    good habit read as a rule is exactly the mistake this project refuses to make: the rulebook
    is the authority and a player's tip is not.
    """
    client = client or get_client()
    if not client.collection_exists(KNOWLEDGE):
        return []
    hits = search(
        client,
        KNOWLEDGE,
        embed_text(question),
        limit=limit,
        filters=build_filter(game_id=GAME_ID, kind="strategy"),
    )
    return [{**h["payload"], "score": h["score"]} for h in hits]


def cards_in_reserve(names: Iterable[str], *, client: Any = None) -> list[dict[str, Any]]:
    """What each card in the reserve does, for deciding whether any is worth buying."""
    found = []
    for name in names:
        payload = card(name, client=client)
        if payload is not None:
            found.append(payload)
        else:
            log.info("card not in the knowledge base: %s", name)
    return found
