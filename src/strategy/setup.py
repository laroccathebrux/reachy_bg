"""Turning a spoken briefing into a game state: "the Ancient One is Azathoth, you play Lily Chen".

    reading = read_setup("o ancião é Azathoth, eu jogo com o Leo e você com a Lily", "pt-BR", game)
    reading.applied      # ["facing Azathoth", "Leo Anderson with Alessandro", "Lily Chen with you"]
    reading.questions    # what is still missing, one question at a time
    game.briefing()      # read this back out loud for confirmation

The person sets the board up and narrates it; the robot listens once and keeps it. Nothing here
is read from the table, because a setup is mostly facts a camera cannot see anyway - who
controls whom, which Mystery is face up, what the Ancient One is.

The local model extracts the fields; ``reference.py`` then decides whether they are real. That
division matters: the model is good at "who did they mean by 'you'" and unreliable at "is
Nyarlathotep in the base game", so a name it returns is always checked against the twelve
sheets and four Ancient Ones before it is believed. An unknown name becomes a question, never
an assumption.

The narration is free-form and in either language. It does not have to be complete: whatever is
missing comes back in ``questions``, and a second call adds to the same state rather than
replacing it, so the briefing can be given in pieces the way people actually talk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.llm.ollama_client import chat
from src.logger import get_logger
from src.strategy.game import ROBOT, GameState
from src.strategy.reference import (
    ancient_one,
    closest_ancient_ones,
    closest_investigators,
    investigator,
)

log = get_logger(__name__)

# The shape the model must answer in. Everything is optional: a briefing given in pieces is
# normal, and an empty field means "not said", never "none".
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ancient_one": {"type": "string"},
        "doom": {"type": ["integer", "null"]},
        "mystery": {"type": "string"},
        "investigators": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "controller": {"type": "string"},
                },
                "required": ["name", "controller"],
            },
        },
        "reserve": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ancient_one", "investigators", "reserve", "mystery"],
}

INSTRUCTIONS = """You extract the setup of a game of Eldritch Horror (Fantasy Flight, 2013, base game) from what a person just said, and answer with JSON only.

The robot listening is called Reachy. It plays one investigator like any other player.

Fields:
- ancient_one: the Ancient One named, exactly as said. "" when not mentioned.
- doom: the doom value if a number was given for it, otherwise null.
- mystery: the active Mystery if one was described, otherwise "".
- investigators: one entry per investigator mentioned, with:
  - name: the investigator's name as said.
  - controller: "reachy" when the robot plays it, otherwise the person's name who plays it.
- reserve: names of the cards said to be in the reserve.

Reading the controller:
- "you play Lily", "you are Lily Chen", "Lily is yours", "voce joga com a Lily" -> controller "reachy".
- "I play Leo", "eu jogo com o Leo", "Leo is mine" -> controller is the speaker; use their name if they gave it, otherwise "me".
- "Bruno plays Trish" -> controller "Bruno".

Report only what was actually said. Never add an investigator that was not mentioned, never guess a controller, and never translate or correct a name - copy it as heard, even if it sounds wrong."""


@dataclass
class SetupReading:
    """What one narration changed, and what still has to be asked."""

    applied: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)  # names the reference does not have, in English
    # The same names as data (kind, what was said, what it might have been), so the robot can ask
    # about them in the language of the table instead of reading an English sentence out loud.
    unknown_items: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def understood_anything(self) -> bool:
        return bool(self.applied)

    def describe(self) -> str:
        if not self.applied and not self.unknown:
            return "I did not catch any setup in that."
        parts = []
        if self.applied:
            parts.append("Got it: " + "; ".join(self.applied) + ".")
        if self.unknown:
            parts.append("I do not know " + ", ".join(self.unknown) + " in the base game.")
        if self.questions:
            parts.append(self.questions[0])  # ask one thing at a time, as the protocol says
        return " ".join(parts)

    def record(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "questions": self.questions,
            "unknown": self.unknown,
            "unknown_items": self.unknown_items,
            "seconds": round(self.seconds, 2),
            "text": self.describe(),
        }


def describe_unknown(item: dict[str, Any]) -> str:
    """One unrecognised name, in English, for the log and for an English table.

    The suggestions matter more than they look: Whisper writes "Azatov" for Azathoth and "Lily
    Shane" for Lily Chen, and offering the four real names back is the difference between a
    briefing that recovers in one sentence and one that repeats itself.
    """
    kind = "Ancient One" if item.get("kind") == "ancient_one" else "investigator"
    suggestions = item.get("suggestions") or []
    hint = f" (did you mean {' or '.join(suggestions)}?)" if suggestions else ""
    return f'the {kind} "{item.get("said", "")}"{hint}'


def extract(text: str, language: str, *, chat_fn: Any = chat) -> dict[str, Any]:
    """Ask the local model for the setup fields in ``text``; returns {} when it answers nothing."""
    reply = chat_fn(
        [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": f"The person is speaking {language}. They said:\n\n{text}"},
        ],
        format=SCHEMA,
        temperature=0.0,
    )
    try:
        data = json.loads(reply.text or "{}")
    except (ValueError, TypeError) as exc:
        log.warning("setup: the model did not answer with JSON (%s): %.200s", exc, reply.text)
        return {}
    return data if isinstance(data, dict) else {}


def apply_reading(data: dict[str, Any], game: GameState, *, speaker: str = "") -> SetupReading:
    """Check what the model extracted against the reference and put it into the game."""
    reading = SetupReading(raw=data)

    said_ancient = (data.get("ancient_one") or "").strip()
    if said_ancient:
        found = ancient_one(said_ancient)
        if found is None:
            item = {
                "kind": "ancient_one",
                "said": said_ancient,
                "suggestions": closest_ancient_ones(said_ancient),
            }
            reading.unknown_items.append(item)
            reading.unknown.append(describe_unknown(item))
        elif game.ancient_one is None or game.ancient_one.name != found.name:
            game.set_ancient_one(found.name)
            reading.applied.append(f"facing {found.name}, doom starts at {found.starting_doom}")

    doom = data.get("doom")
    if isinstance(doom, int) and doom > 0:
        game.doom = doom
        reading.applied.append(f"doom at {doom}")

    for item in data.get("investigators") or []:
        if not isinstance(item, dict):
            continue
        said = (item.get("name") or "").strip()
        controller = (item.get("controller") or "").strip()
        if not said:
            continue
        sheet = investigator(said)
        if sheet is None:
            item = {"kind": "investigator", "said": said, "suggestions": closest_investigators(said)}
            reading.unknown_items.append(item)
            reading.unknown.append(describe_unknown(item))
            continue
        if controller.lower() in ("reachy", "robot", "you", "robo", "robô"):
            controller = ROBOT
        elif controller.lower() in ("me", "i", "eu", "") and speaker:
            controller = speaker
        state = game.add_investigator(sheet.name, controller=controller or "unknown")
        if state is None:
            continue
        who = "you" if state.is_robot else state.controller
        reading.applied.append(f"{sheet.name} with {who}, starting at {state.space}")

    reserve = [str(c).strip() for c in (data.get("reserve") or []) if str(c).strip()]
    if reserve:
        game.reserve = reserve
        reading.applied.append(f"reserve: {', '.join(reserve)}")

    mystery = (data.get("mystery") or "").strip()
    if mystery:
        game.mystery = mystery
        reading.applied.append(f"mystery: {mystery}")

    reading.questions = questions_for(game, reading)
    return reading


def pending(game: GameState, reading: SetupReading | None = None) -> list[tuple[str, Any]]:
    """What is still missing, most important first, as ``(what, detail)`` pairs.

    The pairs are the questions without their words, so that the same list can be asked out
    loud in Portuguese or in English (``src/integration/game_session.py`` says them; this
    module renders the English for the log).
    """
    asks: list[tuple[str, Any]] = []
    if reading is not None and reading.unknown_items:
        asks.append(("unknown_name", reading.unknown_items[0]))
    elif reading is not None and reading.unknown:
        asks.append(("unknown_name", {"kind": "", "said": reading.unknown[0], "suggestions": []}))
    if game.ancient_one is None:
        asks.append(("ancient_one", ""))
    if not game.investigators:
        asks.append(("investigators", ""))
    elif game.robot_investigator is None:
        asks.append(("mine", ""))
    if game.investigators and not game.mystery:
        asks.append(("mystery", ""))
    unclaimed = [i.name for i in game.investigators if i.piece_id is None]
    if unclaimed and game.board.pieces:
        asks.append(("which_piece", unclaimed[0]))
    return asks


ENGLISH_ASKS = {
    "unknown_name": "I do not know {detail}. Which one is it?",
    "ancient_one": "Which Ancient One are we facing?",
    "investigators": "Which investigators are in play, and who plays each one?",
    "mine": "Which investigator am I playing?",
    "mystery": "What does the current Mystery ask for?",
    "which_piece": "I cannot tell which piece on the board is {detail}. Where is it?",
}


def questions_for(game: GameState, reading: SetupReading | None = None) -> list[str]:
    """What to ask next, most important first, phrased to be said out loud (English)."""
    out = []
    for what, detail in pending(game, reading):
        text = describe_unknown(detail) if what == "unknown_name" else str(detail)
        out.append(ENGLISH_ASKS[what].format(detail=text))
    return out


def read_setup(
    text: str,
    language: str,
    game: GameState,
    *,
    speaker: str = "",
    chat_fn: Any = chat,
) -> SetupReading:
    """One narration in, one updated game state out. Call again to add more."""
    import time

    started = time.perf_counter()
    data = extract(text, language, chat_fn=chat_fn)
    reading = apply_reading(data, game, speaker=speaker)
    reading.seconds = time.perf_counter() - started
    log.info("setup: %s", reading.describe())
    return reading


__all__ = [
    "INSTRUCTIONS",
    "SCHEMA",
    "SetupReading",
    "apply_reading",
    "describe_unknown",
    "extract",
    "pending",
    "ENGLISH_ASKS",
    "questions_for",
    "read_setup",
]
