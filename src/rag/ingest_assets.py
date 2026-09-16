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
from typing import Any

from src.logger import get_logger
from src.rag.collections import GAME_ID, KNOWLEDGE
from src.rag.embeddings import embed_texts
from src.rag.store import count, ensure_collection, get_client, point_id, upsert

log = get_logger(__name__)

CATEGORIES = ("item", "weapon", "ally", "spell", "trinket", "service")

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


# Starting possessions: every one of these is named in docs/GAME_REFERENCE.md against the
# investigator who begins holding it, so the card is certainly in the base box.
ENTRIES: list[dict[str, Any]] = [
    asset(
        "Mists of Releh",
        "spell",
        "Mists of Releh is a Spell and the starting possession of Akachi Onyele, the Shaman. It is "
        "an evade-type Spell used to avoid monsters rather than fight them, which suits an "
        "investigator whose job is reaching and sealing gates rather than winning combats. The "
        "exact wording on the card has not been read by this project.",
        confidence="effect_unchecked",
        starting_for="Akachi Onyele",
    ),
    asset(
        "Personal Assistant",
        "ally",
        "Personal Assistant is an Ally and the starting possession of Charlie Kane, the Politician. "
        "Allies stay with an investigator and add to skills or grant effects. The exact wording has "
        "not been read by this project.",
        confidence="effect_unchecked",
        starting_for="Charlie Kane",
    ),
    asset(
        "Arcane Manuscripts",
        "item",
        "Arcane Manuscripts is a starting possession of Diana Stanley, the Redeemed Cultist, whose "
        "strongest skill is Lore 4. The exact wording has not been read by this project.",
        confidence="effect_unchecked",
        starting_for="Diana Stanley",
    ),
    asset(
        "Wither",
        "spell",
        "Wither is a Spell and a starting possession of Diana Stanley. It is an attack Spell, used "
        "with Lore instead of Strength in combat, which matters for an investigator with Lore 4 and "
        "Strength 3. The exact wording has not been read by this project.",
        confidence="effect_unchecked",
        starting_for="Diana Stanley",
    ),
    asset(
        "Flesh Ward",
        "spell",
        "Flesh Ward is a Spell and the starting possession of Jacqueline Fine, the Psychic. It is a "
        "protective Spell that prevents damage, which matters for an investigator with only 4 "
        "Health. The exact wording has not been read by this project.",
        confidence="effect_unchecked",
        starting_for="Jacqueline Fine",
    ),
    asset(
        "Shriveling",
        "spell",
        "Shriveling is a Spell and the starting possession of Jim Culver, the Musician. It is an "
        "attack Spell cast with Lore. The exact wording has not been read by this project.",
        confidence="effect_unchecked",
        starting_for="Jim Culver",
    ),
    asset(
        "Hired Muscle",
        "ally",
        "Hired Muscle is an Ally and the starting possession of Leo Anderson, the Expedition Leader. "
        "Allies of this sort add to combat. The exact wording has not been read by this project.",
        confidence="effect_unchecked",
        starting_for="Leo Anderson",
    ),
    asset(
        "Protective Amulet",
        "trinket",
        "Protective Amulet is a starting possession of Lily Chen, the Martial Artist. The exact "
        "wording has not been read by this project.",
        confidence="effect_unchecked",
        starting_for="Lily Chen",
    ),
    asset(
        "Lucky Rabbit's Foot",
        "trinket",
        "Lucky Rabbit's Foot is a starting possession of Lily Chen, the Martial Artist. Trinkets of "
        "this kind help with rerolls or luck. The exact wording has not been read by this project.",
        confidence="effect_unchecked",
        starting_for="Lily Chen",
    ),
    asset(
        ".18 Derringer",
        "weapon",
        ".18 Derringer is a Weapon and the starting possession of Lola Hayes, the Actress. It is the "
        "smallest of the base-game firearms, giving a small bonus to Strength in a Combat Encounter. "
        "The exact bonus has not been read off the card by this project.",
        confidence="effect_unchecked",
        starting_for="Lola Hayes",
    ),
    asset(
        ".38 Revolver",
        "weapon",
        ".38 Revolver is a Weapon and a starting possession of Mark Harrigan, the Soldier. It adds to "
        "Strength in a Combat Encounter. The exact bonus has not been read off the card by this "
        "project.",
        confidence="effect_unchecked",
        starting_for="Mark Harrigan",
    ),
    asset(
        "Kerosene",
        "item",
        "Kerosene is an Item and a starting possession of Mark Harrigan, the Soldier. It was in the "
        "reserve during the owner's setup on 2026-09-16. The exact wording has not been read off the "
        "card by this project.",
        confidence="effect_unchecked",
        starting_for="Mark Harrigan",
    ),
    asset(
        "Feed the Mind",
        "spell",
        "Feed the Mind is a Spell and the starting possession of Norman Withers, the Astronomer, "
        "whose Lore is 3 and Will 4. The exact wording has not been read by this project.",
        confidence="effect_unchecked",
        starting_for="Norman Withers",
    ),
    asset(
        "Fishing Net",
        "item",
        "Fishing Net is an Item and the starting possession of Silas Marsh, the Sailor. The exact "
        "wording has not been read by this project.",
        confidence="effect_unchecked",
        starting_for="Silas Marsh",
    ),
    asset(
        ".45 Automatic",
        "weapon",
        ".45 Automatic is a Weapon and the starting possession of Trish Scarborough, the Spy. The "
        "community guide in bg_knowledge names it as what makes Trish strong in combat alongside her "
        "extra die. The exact bonus has not been read off the card by this project.",
        confidence="effect_unchecked",
        starting_for="Trish Scarborough",
    ),
    # Seen in the reserve on the owner's table, 2026-09-16. Existence observed, effects unread.
    asset(
        "Lucky Cigarette Case",
        "trinket",
        "Lucky Cigarette Case is a base-game asset seen in the reserve during the owner's setup on "
        "2026-09-16. Its effect has not been read off the card by this project.",
        confidence="effect_unchecked",
    ),
    asset(
        "Private Investigator",
        "ally",
        "Private Investigator is a base-game Ally seen in the reserve during the owner's setup on "
        "2026-09-16. Its effect has not been read off the card by this project.",
        confidence="effect_unchecked",
    ),
    asset(
        "Bull Whip",
        "weapon",
        "Bull Whip is a base-game Weapon seen in the reserve during the owner's setup on 2026-09-16. "
        "Its effect has not been read off the card by this project.",
        confidence="effect_unchecked",
    ),
]


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
        if category not in CATEGORIES:
            category = "item"
        records.append(
            asset(
                name,
                category,
                effect or f"{name} is a base-game asset whose effect has not been read yet.",
                confidence="verified" if effect else "effect_unchecked",
                starting_for=str(item.get("starting_for") or "").strip(),
                source=str(item.get("source") or path),
            )
        )
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

    entries = ENTRIES
    if args.source:
        incoming, refused = from_file(args.source)
        for reason in refused:
            log.warning("asset refused: %s", reason)
        entries = merge(ENTRIES, incoming)
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
