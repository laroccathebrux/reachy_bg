"""Thin wrapper around the Qdrant client: collection lifecycle, upserts and search."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from src.config import QDRANT_API_KEY, QDRANT_URL
from src.rag.collections import BOOL_INDEXES, INTEGER_INDEXES, KEYWORD_INDEXES, VECTOR_DIM

_UPSERT_BATCH = 64


def get_client(url: str = QDRANT_URL, api_key: str = QDRANT_API_KEY) -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key or None)


def point_id(*parts: Any) -> str:
    """Deterministic UUID for a point, so re-ingesting overwrites instead of duplicating."""
    key = "/".join(str(p) for p in parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def ensure_collection(client: QdrantClient, name: str, *, recreate: bool = False) -> None:
    """Create the collection and its payload indexes if missing (or from scratch when recreate)."""
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
    for field in KEYWORD_INDEXES.get(name, ()):
        _ensure_index(client, name, field, PayloadSchemaType.KEYWORD)
    for field in BOOL_INDEXES.get(name, ()):
        _ensure_index(client, name, field, PayloadSchemaType.BOOL)
    for field in INTEGER_INDEXES.get(name, ()):
        _ensure_index(client, name, field, PayloadSchemaType.INTEGER)


def _ensure_index(client: QdrantClient, name: str, field: str, schema: PayloadSchemaType) -> None:
    info = client.get_collection(name)
    if field in (info.payload_schema or {}):
        return
    client.create_payload_index(name, field_name=field, field_schema=schema)


def upsert(
    client: QdrantClient,
    name: str,
    ids: Sequence[str],
    vectors: Sequence[Sequence[float]],
    payloads: Sequence[dict[str, Any]],
) -> int:
    """Upsert points in batches. Returns the number of points written."""
    if not (len(ids) == len(vectors) == len(payloads)):
        raise ValueError("ids, vectors and payloads must have the same length")
    written = 0
    for start in range(0, len(ids), _UPSERT_BATCH):
        batch = [
            PointStruct(id=i, vector=list(v), payload=p)
            for i, v, p in zip(
                ids[start : start + _UPSERT_BATCH],
                vectors[start : start + _UPSERT_BATCH],
                payloads[start : start + _UPSERT_BATCH],
                strict=True,
            )
        ]
        client.upsert(name, points=batch, wait=True)
        written += len(batch)
    return written


def scroll_all(client: QdrantClient, name: str, *, with_vectors: bool = False) -> list[dict[str, Any]]:
    """Return every point of a collection as ``{"id", "payload", "vector"?}`` dicts."""
    out: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            name, limit=256, offset=offset, with_payload=True, with_vectors=with_vectors
        )
        for p in points:
            item: dict[str, Any] = {"id": str(p.id), "payload": dict(p.payload or {})}
            if with_vectors:
                item["vector"] = p.vector
            out.append(item)
        if offset is None:
            break
    return out


def build_filter(**equals: Any) -> Filter | None:
    """``build_filter(kind="monster", base_game=True)`` -> Qdrant must-filter; None when empty."""
    conditions = [
        FieldCondition(key=key, match=MatchValue(value=value))
        for key, value in equals.items()
        if value is not None
    ]
    return Filter(must=conditions) if conditions else None


def search(
    client: QdrantClient,
    name: str,
    vector: Sequence[float],
    *,
    limit: int = 6,
    filters: Filter | None = None,
) -> list[dict[str, Any]]:
    """Nearest neighbours as ``{"id", "score", "payload"}`` dicts, best first."""
    result = client.query_points(
        name, query=list(vector), query_filter=filters, limit=limit, with_payload=True
    )
    return [
        {"id": str(p.id), "score": float(p.score), "payload": dict(p.payload or {})} for p in result.points
    ]


def count(client: QdrantClient, name: str) -> int:
    return int(client.count(name, exact=True).count)


def collection_names(client: QdrantClient) -> Iterable[str]:
    return (c.name for c in client.get_collections().collections)
