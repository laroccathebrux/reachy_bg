"""Split section text into overlapping chunks sized for embedding and retrieval."""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?:;])\s+")


def chunk_text(text: str, *, target: int = 800, overlap: int = 120, minimum: int = 200) -> list[str]:
    """Return chunks of roughly ``target`` characters, split on sentence boundaries.

    Paragraph breaks (blank lines) are preferred split points; sentences are the fallback.
    The last ``overlap`` characters of a chunk are repeated at the start of the next one so a
    rule cut in half is still retrievable. Chunks shorter than ``minimum`` are merged into
    their predecessor.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= target:
        return [text]

    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= target:
            units.append(paragraph)
        else:
            units.extend(s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip())

    chunks: list[str] = []
    buffer = ""
    for unit in units:
        candidate = f"{buffer}\n\n{unit}".strip() if buffer else unit
        if len(candidate) <= target or not buffer:
            buffer = candidate
            continue
        chunks.append(buffer)
        tail = buffer[-overlap:].strip() if overlap else ""
        tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
        buffer = f"{tail} {unit}".strip() if tail else unit
    if buffer:
        if chunks and len(buffer) < minimum:
            chunks[-1] = f"{chunks[-1]}\n\n{buffer}"
        else:
            chunks.append(buffer)
    return chunks


__all__ = ["chunk_text"]
