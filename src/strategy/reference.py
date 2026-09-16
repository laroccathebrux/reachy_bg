"""The 12 investigators and 4 Ancient Ones of the base game, as data the code can check against.

    sheet = investigator("lily chen")        # tolerant lookup: case, accents, first name alone
    sheet.starting_space                     # "Shanghai" - the name spaces.py uses
    ancient_one("azathoth").starting_doom     # 15

These numbers also live in ``docs/GAME_REFERENCE.md`` (prose, for people) and in
``bg_knowledge`` (embedded, for retrieval). They are repeated here because setup has to
*validate* what a person dictates - "Lily Chen starts in Shanghai" is either right or worth
questioning - and a vector search is the wrong tool for an exact lookup that must not be
approximate. The three must agree; the reference document is the source of truth.

Base game only (CLAUDE.md hard rule 2). A name from an expansion is not found here, and the
caller is expected to say so rather than guess.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

SKILLS = ("lore", "influence", "observation", "strength", "will")


@dataclass(frozen=True)
class Investigator:
    """One investigator sheet as it starts the game."""

    name: str
    occupation: str
    health: int
    sanity: int
    lore: int
    influence: int
    observation: int
    strength: int
    will: int
    starting_space: str  # the name used by src.vision.spaces
    starting_possessions: tuple[str, ...] = ()
    starting_clues: int = 0

    @property
    def skills(self) -> dict[str, int]:
        return {s: getattr(self, s) for s in SKILLS}

    def describe(self) -> str:
        best = max(SKILLS, key=lambda s: getattr(self, s))
        return (
            f"{self.name}, {self.occupation}: health {self.health}, sanity {self.sanity}, "
            f"best skill {best} {getattr(self, best)}, starts at {self.starting_space}"
        )


@dataclass(frozen=True)
class AncientOne:
    name: str
    title: str
    starting_doom: int
    mysteries_to_win: int = 3

    def describe(self) -> str:
        return (
            f"{self.name}, {self.title}: doom starts at {self.starting_doom}, "
            f"{self.mysteries_to_win} mysteries to win"
        )


INVESTIGATORS: tuple[Investigator, ...] = (
    Investigator("Akachi Onyele", "The Shaman", 5, 7, 3, 2, 2, 2, 4, "15", ("Mists of Releh",), 1),
    Investigator(
        "Charlie Kane", "The Politician", 4, 8, 2, 4, 3, 2, 2, "San Francisco", ("Personal Assistant",)
    ),
    Investigator(
        "Diana Stanley", "The Redeemed Cultist", 7, 5, 4, 2, 3, 3, 1, "7", ("Arcane Manuscripts", "Wither")
    ),
    Investigator("Jacqueline Fine", "The Psychic", 4, 8, 4, 2, 3, 1, 3, "5", ("Flesh Ward",), 1),
    Investigator("Jim Culver", "The Musician", 7, 5, 3, 3, 2, 2, 3, "6", ("Shriveling",), 1),
    Investigator(
        "Leo Anderson", "The Expedition Leader", 6, 6, 2, 2, 3, 3, 3, "Buenos Aires", ("Hired Muscle",)
    ),
    Investigator(
        "Lily Chen",
        "The Martial Artist",
        6,
        6,
        2,
        2,
        2,
        4,
        3,
        "Shanghai",
        ("Protective Amulet", "Lucky Rabbit's Foot"),
    ),
    Investigator("Lola Hayes", "The Actress", 5, 7, 2, 4, 2, 2, 3, "Tokyo", (".18 Derringer",)),
    Investigator("Mark Harrigan", "The Soldier", 8, 4, 1, 2, 2, 4, 4, "14", (".38 Revolver", "Kerosene")),
    Investigator("Norman Withers", "The Astronomer", 5, 7, 3, 1, 3, 2, 4, "Arkham", ("Feed the Mind",)),
    Investigator("Silas Marsh", "The Sailor", 8, 4, 1, 3, 3, 3, 3, "Sydney", ("Fishing Net",)),
    Investigator("Trish Scarborough", "The Spy", 7, 5, 1, 3, 4, 3, 2, "16", (".45 Automatic",)),
)

ANCIENT_ONES: tuple[AncientOne, ...] = (
    AncientOne("Azathoth", "The Daemon Sultan", 15),
    AncientOne("Cthulhu", "The Madness from the Sea", 12),
    AncientOne("Shub-Niggurath", "The Black Goat of the Woods", 13),
    AncientOne("Yog-Sothoth", "The Lurker at the Threshold", 14),
)


def _key(text: str) -> str:
    """Fold a spoken name to something comparable: no accents, no punctuation, lower case."""
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return "".join(c for c in folded.lower() if c.isalnum() or c == " ").strip()


_INV_BY_KEY: dict[str, Investigator] = {}
for _sheet in INVESTIGATORS:
    _INV_BY_KEY[_key(_sheet.name)] = _sheet
    _INV_BY_KEY[_key(_sheet.name.split()[0])] = _sheet  # "Lily", "Akachi" - how people speak
    _INV_BY_KEY[_key(_sheet.name.split()[-1])] = _sheet  # "Chen", "Onyele"

_AO_BY_KEY: dict[str, AncientOne] = {}
for _ao in ANCIENT_ONES:
    _AO_BY_KEY[_key(_ao.name)] = _ao
    _AO_BY_KEY[_key(_ao.name.replace("-", " "))] = _ao
    _AO_BY_KEY[_key(_ao.name.split("-")[0])] = _ao  # "Shub", "Yog"


def investigator(name: str) -> Investigator | None:
    """Find a sheet by full name, first name or surname; None when it is not base-game content."""
    return _INV_BY_KEY.get(_key(name))


def ancient_one(name: str) -> AncientOne | None:
    return _AO_BY_KEY.get(_key(name))


def closest_investigators(name: str, limit: int = 3) -> list[str]:
    """Names to offer back when a spoken one is not recognised (ASR mangles names often)."""
    import difflib

    by_key = {_key(sheet.name): sheet.name for sheet in INVESTIGATORS}
    hits = difflib.get_close_matches(_key(name), list(by_key), n=limit, cutoff=0.4)
    return [by_key[h] for h in hits]


@dataclass(frozen=True)
class Mystery:
    """One of the 16 Mysteries of the box, as data: what it takes and where it happens."""

    name: str
    ancient_one: str
    type: str
    requirement: str
    spaces: tuple[str, ...] = ()
    epic_monster: str = ""

    def describe(self) -> str:
        where = f" It is at {', '.join(self.spaces)}." if self.spaces else ""
        return f"{self.name} ({self.ancient_one}): {self.requirement}{where}"


@lru_cache(maxsize=1)
def _mysteries() -> tuple[Mystery, ...]:
    """Read from the versioned file rather than typed here: it carries its own source per card."""
    import json

    path = Path(__file__).resolve().parents[1] / "rag" / "cards" / "eldritch_base_mysteries.json"
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - the file ships with the repository
        return ()
    return tuple(
        Mystery(
            name=e["name"],
            ancient_one=e["ancient_one"],
            type=e["type"],
            requirement=e["requirement"],
            spaces=tuple(e.get("spaces") or ()),
            epic_monster=e.get("epic_monster", ""),
        )
        for e in entries
    )


def mysteries(ancient_one_name: str = "") -> tuple[Mystery, ...]:
    """Every Mystery of the box, or the four of one Ancient One."""
    found = _mysteries()
    if not ancient_one_name:
        return found
    wanted = _key(ancient_one_name)
    return tuple(m for m in found if _key(m.ancient_one) == wanted)


def mystery(name: str) -> Mystery | None:
    """One Mystery by name, tolerant of how it is written ("the deep ones attack")."""
    wanted = _key(name)
    if not wanted:
        return None
    for found in _mysteries():
        if _key(found.name) == wanted:
            return found
    return None


def closest_mysteries(name: str, limit: int = 3) -> list[str]:
    """Names to offer back when a spoken Mystery is not recognised ("Q-Mystery", "Key Mystery")."""
    import difflib

    by_key = {_key(m.name): m.name for m in _mysteries()}
    hits = difflib.get_close_matches(_key(name), list(by_key), n=limit, cutoff=0.4)
    return [by_key[h] for h in hits]


def closest_ancient_ones(name: str, limit: int = 3) -> list[str]:
    """The same for the four Ancient Ones: Whisper writes "Azatov", "AsaTot", "Asa Tot"."""
    import difflib

    by_key = {_key(one.name): one.name for one in ANCIENT_ONES}
    hits = difflib.get_close_matches(_key(name), list(by_key), n=limit, cutoff=0.4)
    return [by_key[h] for h in hits]


# The three Location Encounter decks and the cities they cover (docs/GAME_REFERENCE.md, Board).
REGIONS: dict[str, tuple[str, ...]] = {
    "America": ("San Francisco", "Arkham", "Buenos Aires"),
    "Europe": ("London", "Rome", "Istanbul"),
    "Asia/Australia": ("Shanghai", "Sydney", "Tokyo"),
}


def region_of(space: str) -> str:
    """Which Location deck a city belongs to, or "" for a space that has no city encounter.

    It is what lets "saiu a carta 8, lê a parte de Rome" find the card: the deck is not said
    out loud at a table, it is implied by the city.
    """
    wanted = _key(space)
    for region, cities in REGIONS.items():
        if any(_key(city) == wanted for city in cities):
            return region
    return ""


def starting_spaces() -> dict[str, str]:
    """Investigator name -> the space it starts on, which is how setup names pieces on the board."""
    return {s.name: s.starting_space for s in INVESTIGATORS}


@dataclass
class Unknown:
    """A name the reference does not have, kept so the robot can ask instead of inventing."""

    said: str
    suggestions: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {"said": self.said, "suggestions": self.suggestions}


__all__ = [
    "ANCIENT_ONES",
    "INVESTIGATORS",
    "SKILLS",
    "AncientOne",
    "Investigator",
    "Mystery",
    "Unknown",
    "ancient_one",
    "closest_ancient_ones",
    "closest_investigators",
    "closest_mysteries",
    "investigator",
    "mysteries",
    "mystery",
    "REGIONS",
    "region_of",
    "starting_spaces",
]
