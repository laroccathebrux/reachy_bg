"""The 16 Mysteries of the 2013 box: the thing the table is actually trying to do.

    uv run python -m src.rag.ingest_mysteries            # -> bg_knowledge, kind "mystery"
    uv run python -m src.rag.ingest_mysteries --dry-run  # print the records and check them

A game of Eldritch Horror is won by solving three Mysteries, and until now the robot knew the
word and nothing else: the Mystery in ``GameState`` was free text the owner had read out, so
"The Deep Ones Attack!" arrived as "Q-Mystery" and meant nothing to the reasoning.

``cards/eldritch_base_mysteries.json`` is the four Mysteries of each base-game Ancient One as
**checkable data**, not as their printed text: the name, the Ancient One, which of the five kinds
of Mystery it is, one line on what solving it takes, the spaces the card names, and the Epic
Monster it spawns. The spaces are the point - "Rituals in the Wild" puts Eldritch tokens on 4, 10,
21 and Tunguska, and those are names ``src/vision/spaces.py`` knows, so the robot can walk there.

What is deliberately not here is the card's own wording. The effects the robot has to quote - the
76 asset cards - are in ``eldritch_base_game.json`` because a player reads them off a card in
front of them and the robot has to say exactly what they say; a Mystery is read once by the
person holding it, and what the robot needs from it is what it asks of the group. For the exact
wording of any card, ``src/rag/dictate.py`` takes it from the owner's own copy.

Every entry carries the page it was read from. The counts ("one per investigator", "half the
investigators") are written in words rather than numbers because they scale with the group, and
the rulebook is the authority on the rounding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.logger import get_logger
from src.rag.collections import GAME_ID, KNOWLEDGE
from src.strategy.reference import ANCIENT_ONES
from src.vision.spaces import BY_NAME

log = get_logger(__name__)

MYSTERIES_FILE = Path(__file__).parent / "cards" / "eldritch_base_mysteries.json"
BASE_GAME = "Eldritch Horror (base game)"
TYPES = ("research_encounter", "epic_monster", "special_encounter", "eldritch_tokens", "misc")
PER_ANCIENT_ONE = 4  # the base box holds four Mysteries for each of its four Ancient Ones


def load(path: Path = MYSTERIES_FILE) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(entries: list[dict[str, Any]]) -> list[str]:
    """Everything that would make this list wrong, in words. Empty means it can be trusted.

    The checks are the ones the data can answer on its own: the Ancient One must be one of the
    four in this box, the type must be one of the five, every space named must exist on the map,
    and each Ancient One must have exactly four Mysteries - which is what catches an expansion
    card slipping in, the way the asset import catches them by their set.
    """
    problems: list[str] = []
    names = {one.name for one in ANCIENT_ONES}
    counts: dict[str, int] = dict.fromkeys(names, 0)
    seen: set[str] = set()
    for entry in entries:
        name = str(entry.get("name", "")).strip()
        if not name:
            problems.append("an entry has no name")
            continue
        if name in seen:
            problems.append(f"{name}: listed twice")
        seen.add(name)
        one = str(entry.get("ancient_one", "")).strip()
        if one not in names:
            problems.append(f"{name}: {one!r} is not an Ancient One of the base game")
        else:
            counts[one] += 1
        if entry.get("type") not in TYPES:
            problems.append(f"{name}: {entry.get('type')!r} is not a kind of Mystery")
        if not str(entry.get("requirement", "")).strip():
            problems.append(f"{name}: nothing says what solving it takes")
        for space in entry.get("spaces") or []:
            if space not in BY_NAME:
                problems.append(f"{name}: {space!r} is not a space on the board")
        if not str(entry.get("source_url", "")).strip():
            problems.append(f"{name}: no source")
    for one, count in counts.items():
        if count != PER_ANCIENT_ONE:
            problems.append(f"{one} has {count} Mysteries, not {PER_ANCIENT_ONE}")
    return problems


def record(entry: dict[str, Any]) -> dict[str, Any]:
    """One Mystery as a ``bg_knowledge`` point."""
    spaces = list(entry.get("spaces") or [])
    where = f" It names {', '.join(spaces)} on the board." if spaces else ""
    monster = f" It spawns the {entry['epic_monster']} Epic Monster." if entry.get("epic_monster") else ""
    return {
        "game_id": GAME_ID,
        "kind": "mystery",
        "name": entry["name"],
        "expansion": BASE_GAME,
        "base_game": True,
        "ancient_one": entry["ancient_one"],
        "mystery_type": entry["type"],
        "spaces": spaces,
        "epic_monster": entry.get("epic_monster", ""),
        "needs": entry.get("needs", {}),
        "confidence": "stated",
        "source_kind": "community_wiki",
        "source_url": entry.get("source_url", ""),
        "text": (
            f"Mystery of {entry['ancient_one']}: {entry['name']}. {entry['requirement']}{where}{monster}"
        ),
    }


def records(path: Path = MYSTERIES_FILE) -> list[dict[str, Any]]:
    entries = load(path)
    problems = check(entries)
    if problems:
        raise ValueError("the Mystery list does not check out: " + "; ".join(problems[:5]))
    return [record(e) for e in entries]


def point_id(name: str) -> str:
    """The shared convention (game, kind, name), so re-importing corrects instead of doubling."""
    from src.rag.store import point_id as shared_id

    return shared_id(GAME_ID, "mystery", name)


def ingest(entries: list[dict[str, Any]] | None = None, *, client: Any = None) -> int:
    from src.rag.embeddings import embed_texts
    from src.rag.store import ensure_collection, get_client, upsert

    points = entries if entries is not None else records()
    client = client or get_client()
    ensure_collection(client, KNOWLEDGE)
    vectors = embed_texts([p["text"] for p in points])
    upsert(client, KNOWLEDGE, ids=[point_id(p["name"]) for p in points], vectors=vectors, payloads=points)
    return len(points)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the records, write nothing")
    args = parser.parse_args(argv)

    entries = load()
    problems = check(entries)
    for problem in problems:
        print(f"! {problem}")
    if problems:
        return 1
    points = [record(e) for e in entries]
    for point in points:
        print(f"{point['ancient_one']:16s} {point['mystery_type']:20s} {point['text']}")
    if args.dry_run:
        print(f"\n{len(points)} Mysteries, nothing written")
        return 0
    written = ingest(points)
    print(f"\n{written} Mysteries written to {KNOWLEDGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
