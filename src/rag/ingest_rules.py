"""Ingest the official English rules PDFs into the ``bg_rules`` collection.

Usage (from the repository root):

    uv run python -m src.rag.ingest_rules                      # rulebook + reference guide
    uv run python -m src.rag.ingest_rules --dry-run            # print sections/chunks, no writes
    uv run python -m src.rag.ingest_rules --recreate           # drop and rebuild the collection
    uv run python -m src.rag.ingest_rules --only reference_guide

Each PDF is split into sections by typography (see ``pdf_sections``), each section into
overlapping chunks (see ``chunking``), and each chunk is embedded with bge-m3 and stored with
the payload described in ``collections.py``. Point ids are deterministic, so re-running
overwrites the same points.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from src.config import RULEBOOK_DIR
from src.logger import get_logger
from src.rag.chunking import chunk_text
from src.rag.collections import GAME_ID, RULES
from src.rag.embeddings import embed_texts
from src.rag.pdf_sections import HeadingRules, Section, extract_sections
from src.rag.store import count, ensure_collection, get_client, point_id, upsert

log = get_logger(__name__)


@dataclass(frozen=True)
class RuleDocument:
    source: str  # payload value: "rulebook" | "reference_guide"
    filename: str
    rules: HeadingRules


DOCUMENTS: tuple[RuleDocument, ...] = (
    RuleDocument("rulebook", "eh01_rulebook.pdf", HeadingRules(skip_pages=(1,))),
    RuleDocument("reference_guide", "eh01_reference_guide.pdf", HeadingRules(skip_pages=(1, 16))),
)


def build_points(source: str, sections: list[Section]) -> tuple[list[str], list[dict], list[str]]:
    """Turn sections into (ids, payloads, texts) ready for embedding."""
    ids: list[str] = []
    payloads: list[dict] = []
    texts: list[str] = []
    for section in sections:
        for index, chunk in enumerate(chunk_text(section.text)):
            ids.append(point_id(GAME_ID, source, section.path, index))
            payloads.append(
                {
                    "game_id": GAME_ID,
                    "source": source,
                    "section": section.section,
                    "subsection": section.subsection,
                    "path": section.path,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "chunk_index": index,
                    "text": chunk,
                }
            )
            # The breadcrumb is prepended for embedding only: it disambiguates short chunks
            # ("Roll 1 die" under Combat vs under Rest) without polluting the stored text.
            texts.append(f"{section.path}\n{chunk}")
    return ids, payloads, texts


def ingest(document: RuleDocument, *, dry_run: bool, rulebook_dir: Path = RULEBOOK_DIR) -> int:
    pdf = rulebook_dir / document.filename
    if not pdf.exists():
        raise FileNotFoundError(f"{pdf} not found; download the official PDF first (see GETTING_STARTED.md)")
    sections = extract_sections(pdf, document.rules)
    ids, payloads, texts = build_points(document.source, sections)
    log.info("%s: %d sections, %d chunks", document.source, len(sections), len(ids))
    if dry_run:
        for payload in payloads:
            print(
                json.dumps({k: v for k, v in payload.items() if k != "text"}),
                f"| {len(payload['text'])} chars",
            )
        return len(ids)
    vectors = embed_texts(texts)
    client = get_client()
    ensure_collection(client, RULES)
    written = upsert(client, RULES, ids, vectors, payloads)
    log.info(
        "%s: wrote %d points to %s (total now %d)", document.source, written, RULES, count(client, RULES)
    )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="extract and chunk only; print, do not write")
    parser.add_argument("--recreate", action="store_true", help="drop the collection before ingesting")
    parser.add_argument("--only", choices=[d.source for d in DOCUMENTS], help="ingest a single document")
    args = parser.parse_args(argv)

    if args.recreate and not args.dry_run:
        ensure_collection(get_client(), RULES, recreate=True)
        log.info("recreated collection %s", RULES)

    total = 0
    for document in DOCUMENTS:
        if args.only and document.source != args.only:
            continue
        total += ingest(document, dry_run=args.dry_run)
    log.info("done: %d chunks%s", total, " (dry run)" if args.dry_run else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
