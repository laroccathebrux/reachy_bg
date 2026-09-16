"""Asset cards of the base game into ``bg_knowledge``, so the robot knows what a card does.

    uv run python -m src.rag.ingest_assets --dry-run
    uv run python -m src.rag.ingest_assets

The camera can see *that* a Reserve slot holds a card; it will never read *which* card from a
250 px blur, and it can never see a card in somebody's hand at all. The name comes from the
person, the way investigator names do. What is missing is the third question - what does that
card actually do - and without it "should I buy the Bull Whip?" cannot be reasoned about.

Two things are tracked separately and must not be confused:

- ``confidence: "verified"`` - the card is named in ``docs/GAME_REFERENCE.md`` as a starting
  possession, so its existence in the 2013 base box is established by this project's own
  reference, and its effect is recorded here.
- ``confidence: "effect_unchecked"`` - the card exists (it was seen on the owner's table or is
  named in the reference) but the wording of its effect has not been read off the card by this
  project. The text says what is known and what is not.

Nothing here is invented. A card whose effect is not known is entered as a card whose effect is
not known, because a plausible wrong effect would be worse than a gap: the reasoning layer
would spend Influence on a card that does not do what it was told.

The remaining base-game assets are not in this file. The honest ways to add them are to read the
cards - photographed close up, where OCR actually works - or to import a checked list::

    uv run python -m src.rag.ingest_assets --from data/assets.json --dry-run

The file is a JSON list of objects with ``name``, ``category``, ``effect`` and ``expansion``.
``expansion`` is required and is the point of the exercise: nearly every asset list online
covers the game with all eight expansions mixed in, and this project is base game only, so a
card whose set is not the 2013 box is refused rather than imported. An entry with an effect
becomes "verified"; one without keeps "effect_unchecked".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.logger import get_logger
from src.rag.collections import GAME_ID, KNOWLEDGE
from src.rag.embeddings import embed_texts
from src.rag.store import count, ensure_collection, get_client, point_id, upsert

log = get_logger(__name__)

# The card types the base box actually uses. "magical", "tome", "relic" and "unique asset" are
# kinds of Asset in the game's own vocabulary, so they belong here; a Condition is not an Asset
# at all (it is something that happens *to* an investigator) and gets its own knowledge kind.
CATEGORIES = (
    "item",
    "weapon",
    "ally",
    "spell",
    "trinket",
    "service",
    "tome",
    "magical",
    "relic",
    "unique asset",
)
NOT_ASSETS = {"condition": "condition", "task": "task"}

BASE = {
    "game_id": GAME_ID,
    "kind": "asset",
    "expansion": "Eldritch Horror (base game)",
    "base_game": True,
}


def asset(
    name: str,
    category: str,
    text: str,
    *,
    confidence: str = "verified",
    starting_for: str = "",
    **extra: Any,
) -> dict[str, Any]:
    return {
        **BASE,
        "name": name,
        "category": category,  # item | weapon | ally | spell | trinket | service
        "starting_for": starting_for,  # the investigator who begins the game holding it
        "confidence": confidence,
        "text": text,
        **extra,
    }


# The base-game cards, versioned beside this file so the knowledge base can always be rebuilt.
# Sourced as a list covering every expansion and filtered to the 2013 box on the way in; the
# effects are the printed wording, which is what makes them worth anything to the reasoning
# layer ("Bull Whip: +1 Strength during Combat Encounters" is actionable, "helps in a fight"
# is not).
CARDS_FILE = Path(__file__).parent / "cards" / "eldritch_base_game.json"


def starting_possession_of(name: str) -> str:
    """The investigator who begins the game holding this card, or "" - checked, not guessed."""
    from src.strategy.reference import INVESTIGATORS

    for sheet in INVESTIGATORS:
        if name in sheet.starting_possessions:
            return sheet.name
    return ""


def base_game_cards() -> list[dict[str, Any]]:
    """Every card of the 2013 box as knowledge records."""
    records, refused = from_file(str(CARDS_FILE))
    if refused:  # the shipped file is filtered already, so anything refused is a bug in it
        raise ValueError(f"{CARDS_FILE} contains cards that are not base game: {refused[:3]}")
    for record in records:
        record["starting_for"] = starting_possession_of(record["name"])
    return records


BASE_SET_NAMES = ("eldritch horror", "base", "base game", "core", "core set", "2013")


def from_file(path: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Read a checked asset list; returns the records and the reasons entries were refused.

    Every online list this project has met covers the game with its expansions, so the set each
    card belongs to is required and anything outside the 2013 box is refused. A card already in
    ENTRIES is upgraded when the file supplies an effect, and left alone when it does not.
    """
    from pathlib import Path

    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(items, dict):
        items = items.get("assets") or items.get("cards") or []
    records: list[dict[str, Any]] = []
    refused: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            refused.append(f"not an object: {item!r:.60}")
            continue
        name = str(item.get("name") or "").strip()
        expansion = str(item.get("expansion") or item.get("set") or "").strip()
        if not name:
            refused.append(f"no name: {item!r:.60}")
            continue
        if not expansion:
            refused.append(f"{name}: no expansion given, so it cannot be checked")
            continue
        if expansion.lower().strip() not in BASE_SET_NAMES:
            refused.append(f"{name}: from {expansion}, not the base game")
            continue
        effect = str(item.get("effect") or item.get("text") or "").strip()
        category = str(item.get("category") or item.get("type") or "item").strip().lower()
        kind = NOT_ASSETS.get(category, "asset")
        if kind == "asset" and category not in CATEGORIES:
            refused.append(f"{name}: unknown card type {category!r}")
            continue
        record = asset(
            name,
            category,
            effect or f"{name} is a base-game card whose effect has not been read yet.",
            confidence="verified" if effect else "effect_unchecked",
            starting_for=str(item.get("starting_for") or "").strip(),
            source=str(item.get("source") or path),
        )
        record["kind"] = kind  # a Condition is not an Asset and must not answer as one
        records.append(record)
    return records, refused


def merge(base: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Incoming entries win only where they actually add an effect."""
    by_name = {r["name"]: dict(r) for r in base}
    for record in incoming:
        existing = by_name.get(record["name"])
        if existing is None or record["confidence"] == "verified":
            by_name[record["name"]] = record
    return list(by_name.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the records, write nothing")
    parser.add_argument(
        "--from",
        dest="source",
        default="",
        metavar="FILE",
        help="a JSON asset list with name, category, effect and expansion",
    )
    args = parser.parse_args(argv)

    entries = base_game_cards()
    if args.source:
        incoming, refused = from_file(args.source)
        for reason in refused:
            log.debug("card refused: %s", reason)
        entries = merge(entries, incoming)
        log.info("%s: %d accepted, %d refused", args.source, len(incoming), len(refused))

    for record in entries:
        if record["confidence"] not in ("verified", "effect_unchecked"):
            raise ValueError(f"{record['name']}: unknown confidence {record['confidence']}")

    unchecked = sum(1 for r in entries if r["confidence"] == "effect_unchecked")
    log.info("%d assets, %d of them with the effect still unread", len(entries), unchecked)

    if args.dry_run:
        for r in entries:
            print(json.dumps(r, ensure_ascii=False))
        return 0

    kinds: dict[str, int] = {}
    for r in entries:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    log.info("by kind: %s", kinds)

    client = get_client()
    ids = [point_id(GAME_ID, r["kind"], r["name"]) for r in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("two assets share a name")
    vectors = embed_texts(r["text"] for r in entries)
    ensure_collection(client, KNOWLEDGE)
    written = upsert(client, KNOWLEDGE, ids, vectors, entries)
    log.info("wrote %d asset points to %s (total now %d)", written, KNOWLEDGE, count(client, KNOWLEDGE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
