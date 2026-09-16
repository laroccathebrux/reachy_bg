"""Cards read from the owner's own box into the knowledge base, one card at a time.

    reading = read_card("Mystery, Azathoth: The Key and the Gate. Close 3 gates to solve it.")
    reading.record["kind"], reading.record["name"]      # "mystery", "The Key and the Gate"
    save([reading.record], CARDS_FILE)                  # data/cards/dictated.json, git-ignored
    ingest([reading.record])                            # -> bg_knowledge

The 76 asset cards of the 2013 box are in the repository with their printed effects. Everything
else a game is made of - the Mysteries that decide whether the table wins, the encounter decks,
the Mythos cards - is not, and the two places that hold it (the Fandom wiki, BoardGameGeek) both
refuse a fetch, while the community card lists carry names and types without any card text.

So the source is the box on the table. A person reads a card out loud or types it, the local
model puts it into fields, and it is stored with ``confidence="dictated"`` and who read it -
never a guess, never a paraphrase of something nobody checked. That is the same rule the rest of
this project follows for anything it cannot see: the person at the table is the authority on
what is printed on their own components.

The dictated file lives in ``data/`` and is **git-ignored**, like the picture of the board: it is
the game's copyrighted text, kept on the machine that owns a copy of the game, not published.

What this is not for: typing in four hundred encounter cards. An encounter is drawn and read
aloud when it happens, and what the robot needs then is to hear it, not to have memorised the
deck. The cards worth dictating are the ones that decide a plan before they are drawn - the
Mysteries above all, and the printed value of a Reserve card, which the robot has to ask for
today.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import DATA_DIR
from src.logger import get_logger
from src.rag.collections import GAME_ID, KNOWLEDGE
from src.strategy.reference import ancient_one, closest_ancient_ones, region_of
from src.vision.spaces import BY_NAME

log = get_logger(__name__)

CARDS_FILE = DATA_DIR / "cards" / "dictated.json"
BASE_GAME = "Eldritch Horror (base game)"

# The decks a card can come from. "asset" and "condition" are already in the repository for the
# base game; a dictated one corrects or completes what is there.
CARD_KINDS = ("mystery", "encounter", "mythos", "asset", "condition", "artifact", "spell")

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "kind": {"type": "string", "enum": list(CARD_KINDS)},
        "ancient_one": {"type": "string"},
        "deck": {"type": "string"},
        "number": {"type": ["integer", "null"]},
        "space": {"type": "string"},
        "text": {"type": "string"},
        "value": {"type": ["integer", "null"]},
    },
    "required": ["name", "kind", "text"],
}

INSTRUCTIONS = """You put one card of the board game Eldritch Horror (Fantasy Flight, 2013) into fields, and answer with JSON only.

The person is reading a card they are holding. Copy what they read; never add, complete or correct it from your own knowledge of the game.

- name: the card's title, exactly as read.
- kind: mystery, encounter, mythos, asset, condition, artifact or spell.
- ancient_one: for a Mystery or a Research encounter, the Ancient One whose deck it belongs to; "" when they did not say.
- deck: which deck it comes from when they say it ("America", "Europe", "Asia/Australia", "General", "Other World", "Expedition", "Research", "Special"); "" otherwise.
- number: the number printed on an Encounter card, or null. "Saiu a carta 8" means number 8.
- space: for a Location Encounter, the city whose part they are reading ("Rome"); "" otherwise.
- text: what the card does. The cards are printed in English, so write this in English even when the person reads it translated, and keep every game term exactly as the card prints it (Strength, Combat Encounter, Eldritch token, Clue). Do not add anything they did not read.
- value: the number printed on an Asset card, or null.

If they did not read a card at all, answer with an empty name.

Write "name" and "text" in English, whatever language the person read the card in. The card in their hands is printed in English; a Portuguese reading of it is their translation, and what goes in the fields is the card."""


@dataclass
class CardReading:
    """One dictated card: the fields, what could not be checked, and how long it took."""

    record: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.record) and not self.problems

    def describe(self) -> str:
        if not self.record:
            return "I did not catch a card in that." + (f" ({self.problems[0]})" if self.problems else "")
        record = self.record
        where = f", {record['ancient_one']}" if record.get("ancient_one") else ""
        value = f", value {record['value']}" if record.get("value") is not None else ""
        head = f"{record['kind']}{where}{value}: {record['name']}"
        return head if not self.problems else f"{head} - {self.problems[0]}"


def encounter_key(deck: str, number: int | None, space: str = "") -> str:
    """How an Encounter card is called at the table: "Europe 8, Rome" or "Other World 3"."""
    if number is None:
        return ""
    head = f"{deck} {number}".strip()
    return f"{head}, {space}" if space else head


def find_encounter(
    number: int, *, space: str = "", deck: str = "", cards: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """The Encounter card somebody has read, by the number and the city it was drawn for.

    "Saiu a carta 8, lê a parte de Rome" is the whole interface: the number is printed on the
    card, the city says which deck it is, and the part of the card that matters is the city's.
    """
    wanted_deck = deck or region_of(space)
    for card in cards if cards is not None else load():
        if card.get("kind") != "encounter" or card.get("number") != number:
            continue
        if space and str(card.get("space", "")).lower() != space.lower():
            continue
        if wanted_deck and str(card.get("deck", "")).lower() != wanted_deck.lower():
            continue
        return card
    return None


def build_record(data: dict[str, Any], *, read_by: str = "") -> tuple[dict[str, Any], list[str]]:
    """Turn what the model extracted into a knowledge point, or say why it cannot be one.

    The Ancient One is the only field that can be checked against the reference, and it is
    checked: a Mystery filed under an Ancient One this box does not have is a Mystery from an
    expansion, and it does not belong in ``bg_knowledge`` with ``base_game=True``.
    """
    problems: list[str] = []
    name = str(data.get("name") or "").strip()
    kind = str(data.get("kind") or "").strip().lower()
    text = " ".join(str(data.get("text") or "").split())
    if kind not in CARD_KINDS:
        return {}, [f"{name or 'that'}: {kind!r} is not a kind of card in this box"]
    if not name and kind != "encounter":
        # An Encounter card has no title; it is called by its number and its city, and that name
        # is built below. Everything else in the box has one printed on it.
        return {}, ["no card name was read"]
    if not text:
        problems.append("nothing was read about what it does")

    said_one = str(data.get("ancient_one") or "").strip()
    resolved = ""
    if said_one:
        found = ancient_one(said_one)
        if found is None:
            hint = closest_ancient_ones(said_one)
            problems.append(
                f'"{said_one}" is not an Ancient One of the base game'
                + (f" (did you mean {' or '.join(hint)}?)" if hint else "")
            )
        else:
            resolved = found.name
    if kind == "mystery" and not resolved:
        problems.append("a Mystery belongs to an Ancient One, and I did not get which")

    value = data.get("value")
    if not isinstance(value, int):
        value = None

    number = data.get("number")
    number = number if isinstance(number, int) else None
    space = str(data.get("space") or "").strip()
    deck = str(data.get("deck") or "").strip()
    if kind == "encounter":
        # A Location Encounter is one card with a part per city, and at a table it is called by
        # its number and its city: "saiu a carta 8, lê a parte de Rome". The deck is not said
        # out loud - it follows from the city - so it is filled in here.
        if space and not deck:
            deck = region_of(space) or deck
        if space and space not in BY_NAME:
            problems.append(f'"{space}" is not a space on the board')
        if number is None:
            problems.append("an Encounter card has a number on it, and I did not get which")
        name = encounter_key(deck, number, space) or name
    if not name:
        return {}, problems + ["no card name was read"]

    label = f"{kind.capitalize()} card"
    if resolved:
        label += f" of {resolved}"
    record = {
        "game_id": GAME_ID,
        "kind": kind,
        "name": name,
        "expansion": BASE_GAME,
        "base_game": True,
        "ancient_one": resolved,
        "deck": deck,
        "number": number,
        "space": space,
        "value": value,
        "confidence": "dictated",
        "source_kind": "read_from_the_box",
        "source_author": read_by,
        "recorded_at": round(time.time(), 3),
        "text": f"{label}: {name}. {text}".strip(),
    }
    return record, problems


def read_card(
    said: str,
    *,
    language: str = "pt-BR",
    read_by: str = "",
    kind_hint: str = "",
    ancient_one_hint: str = "",
    chat_fn: Any = None,
) -> CardReading:
    """One card read out loud (or typed) into one knowledge point."""
    from src.llm.ollama_client import chat

    chat_fn = chat_fn or chat
    started = time.perf_counter()
    hints = []
    if kind_hint:
        hints.append(f"They are reading a {kind_hint} card.")
    if ancient_one_hint:
        hints.append(f"It belongs to {ancient_one_hint}'s deck.")
    prompt = f"The person is speaking {language}. {' '.join(hints)}\n\nThey read:\n\n{said}"
    reply = chat_fn(
        [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": prompt}],
        format=SCHEMA,
        temperature=0.0,
    )
    try:
        data = json.loads(reply.text or "{}")
    except (ValueError, TypeError) as exc:
        log.warning("dictate: the model did not answer with JSON (%s): %.200s", exc, reply.text)
        data = {}
    if kind_hint and not data.get("kind"):
        data["kind"] = kind_hint
    if ancient_one_hint and not data.get("ancient_one"):
        data["ancient_one"] = ancient_one_hint
    record, problems = build_record(data if isinstance(data, dict) else {}, read_by=read_by)
    reading = CardReading(record=record, raw=data, problems=problems, seconds=time.perf_counter() - started)
    log.info("dictate: %s (%.1fs)", reading.describe(), reading.seconds)
    return reading


def load(path: Path = CARDS_FILE) -> list[dict[str, Any]]:
    """Every card dictated so far, or [] when nothing has been."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("dictate: %s could not be read (%s)", path, exc)
        return []
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def save(records: list[dict[str, Any]], path: Path = CARDS_FILE) -> Path:
    """Add these cards to the dictated file, replacing any card of the same kind and name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {(r.get("kind"), r.get("name")): r for r in load(path)}
    for record in records:
        existing[(record.get("kind"), record.get("name"))] = record
    path.write_text(json.dumps(list(existing.values()), indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def point_id(record: dict[str, Any]) -> str:
    """The id the rest of the knowledge base uses for a card, so a dictated one *corrects* it.

    ``store.point_id(game, kind, name)`` is what ``ingest_assets`` writes, which means reading
    Bull Whip off the card lands on the same point as the imported Bull Whip rather than beside
    it. The box wins over the list: that is the whole reason for reading it out loud.
    """
    from src.rag.store import point_id as shared_id

    return shared_id(GAME_ID, record.get("kind"), record.get("name"))


def ingest(records: list[dict[str, Any]], *, client: Any = None) -> int:
    """Upsert dictated cards into ``bg_knowledge``; returns how many points were written."""
    from src.rag.embeddings import embed_texts
    from src.rag.store import ensure_collection, get_client, upsert

    if not records:
        return 0
    client = client or get_client()
    ensure_collection(client, KNOWLEDGE)
    ids = [point_id(r) for r in records]
    merged = [
        _merged_with_existing(client, point, record) for point, record in zip(ids, records, strict=True)
    ]
    vectors = embed_texts([r["text"] for r in merged])
    upsert(client, KNOWLEDGE, ids=ids, vectors=vectors, payloads=merged)
    return len(merged)


def _merged_with_existing(client: Any, point: str, record: dict[str, Any]) -> dict[str, Any]:
    """Keep what the imported card already knew (its category, who starts with it) under the new
    reading, so correcting a card's text does not throw away the rest of what is known about it."""
    try:
        found = client.retrieve(KNOWLEDGE, ids=[point], with_payload=True)
    except Exception as exc:  # a fresh collection, or Qdrant being unhappy: the new record stands
        log.info("dictate: no earlier point for %s (%s)", record.get("name"), exc)
        return record
    if not found:
        return record
    return {**dict(found[0].payload or {}), **record}


__all__ = [
    "BASE_GAME",
    "CARDS_FILE",
    "CARD_KINDS",
    "INSTRUCTIONS",
    "SCHEMA",
    "CardReading",
    "build_record",
    "encounter_key",
    "find_encounter",
    "ingest",
    "load",
    "point_id",
    "read_card",
    "save",
]
