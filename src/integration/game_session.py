"""The game the robot is in, kept across restarts: heard, written down, read back.

    session = GameSession.open(Path("data/game_logs/game_state.json"), say=speak)
    session.greeting("pt-BR")          # "Continuando: vamos enfrentar Azathoth..."
    session.handle("o ancião é Azathoth e eu jogo com a Jacqueline", "pt-BR", "holding_the_floor")

Until now the robot heard the setup and forgot it: ``src/strategy/setup.py`` could turn a spoken
briefing into a ``GameState`` and nothing in the running conversation ever called it, so what the
owner narrated lived only in the ElevenLabs session and died with it. This is the part that
remembers.

Two things happen here, both instead of the cloud rather than beside it:

- **The setup.** While the robot still does not know what it needs to play - the Ancient One, or
  which investigator is its own - a sentence that is not a question goes to the local extractor,
  the reference checks it, and the robot says back what it wrote down and asks the next missing
  thing. A question always goes to the agent: "Reachy, o que é um Gate?" is not a briefing.
- **The turn**, through ``TurnTaker``, exactly as before.

Everything that changes the state writes the file immediately, so closing the session loses
nothing: a crash, a Ctrl+C or a laptop lid all leave the game where it was. Reopening reads it
back and the robot says what it remembers, which is also how the owner catches a stale game -
the robot tells him it is still playing yesterday's, and he says otherwise.

The lines the robot says here are fixed text in both languages, built from the state rather than
translated from it: a briefing is a template, and paying a model to phrase "we are facing
Azathoth" would be slower and less reliable than writing it once.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.integration.turn_taking import TurnTaker
from src.logger import get_logger
from src.speech.addressee import is_question
from src.strategy.game import GameState
from src.strategy.setup import SetupReading, pending, read_setup

log = get_logger(__name__)

# What the robot says, per language. Templates, not translations of the English log lines.
LINES: dict[str, dict[str, str]] = {
    "pt-BR": {
        "facing": "Vamos enfrentar {ancient_one}, e o Doom começa em {doom}.",
        "mine": "Eu jogo com {name}, começando em {space}.",
        "others": "Com você: {others}.",
        "mystery": "O Mystery é {mystery}.",
        "reserve": "A Reserve tem {reserve}.",
        "noted": "Anotado.",
        "nothing": "Ainda não sei nada desta partida.",
        "continuing": "Continuando a partida.",
    },
    "en-US": {
        "facing": "We are facing {ancient_one}, and Doom starts at {doom}.",
        "mine": "I am playing {name}, starting at {space}.",
        "others": "With you: {others}.",
        "mystery": "The Mystery is {mystery}.",
        "reserve": "The Reserve holds {reserve}.",
        "noted": "Noted.",
        "nothing": "I know nothing about this game yet.",
        "continuing": "Carrying on with the game.",
    },
}

ASKS: dict[str, dict[str, str]] = {
    "pt-BR": {
        "unknown_name": 'Não conheço {kind} "{said}".{hint} Qual é?',
        "ancient_one": "Qual Ancient One a gente vai enfrentar?",
        "investigators": "Quais investigadores estão em jogo, e quem joga com cada um?",
        "mine": "Com qual investigador eu jogo?",
        "mystery": "O que o Mystery atual pede?",
        "which_piece": "Não sei qual peça no tabuleiro é {detail}. Onde ela está?",
    },
    "en-US": {
        "unknown_name": 'I do not know the {kind} "{said}".{hint} Which one is it?',
        "ancient_one": "Which Ancient One are we facing?",
        "investigators": "Which investigators are in play, and who plays each one?",
        "mine": "Which investigator am I playing?",
        "mystery": "What does the current Mystery ask for?",
        "which_piece": "I cannot tell which piece on the board is {detail}. Where is it?",
    },
}


def _lines(language: str) -> dict[str, str]:
    return LINES.get(language, LINES["en-US"])


def briefing(game: GameState, language: str) -> str:
    """What the robot knows about this game, said out loud in the table's language."""
    words = _lines(language)
    parts: list[str] = []
    if game.ancient_one is not None:
        parts.append(words["facing"].format(ancient_one=game.ancient_one.name, doom=game.doom))
    mine = game.robot_investigator
    if mine is not None:
        parts.append(words["mine"].format(name=mine.name, space=mine.space))
    others = [i for i in game.investigators if not i.is_robot]
    if others:
        parts.append(words["others"].format(others=", ".join(f"{i.name} ({i.controller})" for i in others)))
    if game.mystery:
        parts.append(words["mystery"].format(mystery=game.mystery))
    if game.reserve:
        parts.append(words["reserve"].format(reserve=", ".join(game.reserve)))
    return " ".join(parts) or words["nothing"]


# How an unrecognised name is spoken about, per language.
UNKNOWN_WORDS = {
    "pt-BR": {
        "ancient_one": "o Ancient One",
        "investigator": "o investigador",
        "": "",
        "hint": " Era {names}?",
    },
    "en-US": {
        "ancient_one": "Ancient One",
        "investigator": "investigator",
        "": "name",
        "hint": " Did you mean {names}?",
    },
}


def next_question(game: GameState, reading: SetupReading | None, language: str) -> str:
    """The one thing to ask next, in the table's language, or "" when nothing is missing.

    A name the reference refused is asked about with the names it might have been: Whisper
    writes "Azatov" and "Lily Shane", and "Era Azathoth?" ends that in one sentence.
    """
    asks = pending(game, reading)
    if not asks:
        return ""
    what, detail = asks[0]
    table = ASKS.get(language, ASKS["en-US"])
    template = table.get(what, "")
    if what != "unknown_name":
        return template.format(detail=detail)
    words = UNKNOWN_WORDS.get(language, UNKNOWN_WORDS["en-US"])
    suggestions = (detail or {}).get("suggestions") or []
    hint = words["hint"].format(
        names=" ou ".join(suggestions) if language == "pt-BR" else " or ".join(suggestions)
    )
    return template.format(
        kind=words.get((detail or {}).get("kind", ""), words[""]),
        said=(detail or {}).get("said", ""),
        hint=hint if suggestions else "",
    )


class GameSession:
    """The robot's memory of the game in progress, and what it does when spoken to about it."""

    def __init__(
        self,
        game: GameState,
        *,
        say: Callable[[str, str], None],
        path: Path | None = None,
        language: str = "pt-BR",
        plan: Any = None,
        setup_fn: Callable[..., SetupReading] = read_setup,
        taker: TurnTaker | None = None,
        background: bool = True,
        on_setup: Callable[[SetupReading, str], None] | None = None,
    ) -> None:
        self.game = game
        self.say = say
        self.path = Path(path) if path else None
        self.language = language
        self.setup_fn = setup_fn
        self.background = background
        self.on_setup = on_setup
        self.taker = taker or TurnTaker(game, say=say, language=language, plan=plan, background=background)
        self.readings = 0
        self._busy = threading.Lock()

    # ------------------------------------------------------------------ storage
    @classmethod
    def open(cls, path: Path | None, **kwargs: Any) -> GameSession:
        """Read the game back from ``path`` when it is there, or start a new one."""
        game = GameState()
        if path is not None and Path(path).exists():
            try:
                game = GameState.load(Path(path))
                log.info("game: read back from %s", path)
            except Exception as exc:  # a corrupt file must not stop the robot from playing
                log.warning("game: %s could not be read (%s); starting a new one", path, exc)
        return cls(game, path=path, **kwargs)

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.game.save(self.path)
        except OSError as exc:
            log.warning("game: could not write %s (%s)", self.path, exc)

    # ------------------------------------------------------------------ speaking
    def greeting(self, language: str = "") -> str:
        """What the robot says when it opens a game it already knew about."""
        language = language or self.language
        if self.game.ancient_one is None and not self.game.investigators:
            return ""
        words = _lines(language)
        question = next_question(self.game, None, language)
        return " ".join(p for p in (words["continuing"], briefing(self.game, language), question) if p)

    # ------------------------------------------------------------------ listening
    @property
    def wants_setup(self) -> bool:
        """True while something essential is missing, so a narration is worth extracting."""
        return not self.game.ready

    def handle(self, text: str, language: str = "", reason: str = "", speaker: str = "") -> str | None:
        """Deal with an utterance the gate judged to be for the robot.

        Returns the route it was handled as ("my_turn", "setup"), or None to let the agent hear
        it. A question is never taken as a briefing: it belongs to the conversation. ``speaker``
        is the voice the gate recognised, so "eu jogo com a Jacqueline" names a person.
        """
        language = language or self.language
        if self.taker.handle(text, language):
            return "my_turn"
        if not self.wants_setup or is_question(text, language) or self._busy.locked():
            return None
        if self.background:
            threading.Thread(
                target=self._read_setup, args=(text, language, speaker), name="setup-ear", daemon=True
            ).start()
        else:
            self._read_setup(text, language, speaker)
        return "setup"

    def _read_setup(self, text: str, language: str, speaker: str = "") -> None:
        with self._busy:
            started = time.monotonic()
            try:
                # The speaker's own name matters: "eu vou controlar a Jacqueline" has to end up
                # as "Jacqueline Fine with Alessandro", not with a controller called "me".
                reading = self.setup_fn(text, language, self.game, speaker=speaker)
            except Exception as exc:  # the extractor failing must not take the conversation down
                log.exception("setup: reading failed (%s)", exc)
                return
            self.readings += 1
            if self.on_setup is not None:
                self.on_setup(reading, text)
            if reading.applied or reading.unknown:
                self.save()
            log.info(
                "setup %d: %s (%.1fs)",
                self.readings,
                "; ".join(reading.applied) or "nothing new",
                time.monotonic() - started,
            )
            self.say(self.sentence(reading, language), language)

    def sentence(self, reading: SetupReading, language: str) -> str:
        """What the robot says back: what it wrote down, then the next thing it needs.

        Nothing extracted is not silence: the robot asks again, because a briefing the model
        could not read is usually one the person will rephrase. It repeats the whole state only
        when it did write something down, so a "no" does not get the full briefing read back.
        """
        words = _lines(language)
        question = next_question(self.game, reading, language)
        if reading.applied:
            return " ".join(p for p in (words["noted"], briefing(self.game, language), question) if p)
        if self.game.ancient_one is None and not self.game.investigators:
            return " ".join(p for p in (words["nothing"], question) if p)
        return question or briefing(self.game, language)


__all__ = ["ASKS", "LINES", "GameSession", "briefing", "next_question"]
