"""The whole game state: what the camera sees, plus everything it can never see.

    game = GameState()
    game.set_ancient_one("Azathoth")
    game.add_investigator("Lily Chen", controller="reachy")     # the robot's own
    game.add_investigator("Leo Anderson", controller="Alessandro")
    game.claim_pieces()                    # name the pieces a scan found, by starting space
    print(game.briefing())                 # what it understood, to read back out loud
    print(game.missing())                  # what it still has to ask before round 1

``BoardState`` tracks pieces the camera can follow. Most of a game of Eldritch Horror is not
on the board in that sense: doom, the active Mystery, the Omen, cards in hand, health and
sanity, whose turn it is. None of that is a piece that moves from one space to another, so
none of it can be seen - it has to be told. This holds both halves and keeps them consistent.

The two halves meet at ``claim_pieces``. A person says "Lily Chen starts in Shanghai"; the
first scan finds an unnamed piece at Shanghai; the piece becomes Lily Chen and the tracking in
``state.py`` carries that name through every later move. That is why naming a piece from its
picture, which measured 2/13 with a gallery and 0/13 with a VLM, is not on the critical path:
the setup conversation gives the names away for free, and continuity keeps them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.logger import get_logger
from src.strategy.reference import (
    AncientOne,
    Investigator,
    ancient_one,
    closest_investigators,
    investigator,
)
from src.strategy.state import BoardState

log = get_logger(__name__)

ROBOT = "reachy"  # the controller value that means "the robot plays this investigator"
PHASES = ("action", "encounter", "mythos")


@dataclass
class InvestigatorState:
    """One investigator in play: its sheet, who controls it, and what has happened to it."""

    sheet: Investigator
    controller: str  # ROBOT, or the human player's name
    space: str = ""  # current space; starts at the sheet's starting space
    health: int = 0
    sanity: int = 0
    clues: int = 0
    possessions: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    piece_id: int | None = None  # the tracked piece on the board, once claimed

    def __post_init__(self) -> None:
        self.space = self.space or self.sheet.starting_space
        self.health = self.health or self.sheet.health
        self.sanity = self.sanity or self.sheet.sanity
        if not self.possessions:
            self.possessions = list(self.sheet.starting_possessions)
        self.clues = self.clues or self.sheet.starting_clues

    @property
    def name(self) -> str:
        return self.sheet.name

    @property
    def is_robot(self) -> bool:
        return self.controller == ROBOT

    @property
    def defeated(self) -> bool:
        return self.health <= 0 or self.sanity <= 0

    def describe(self) -> str:
        who = "you" if self.is_robot else self.controller
        return (
            f"{self.name} ({who}) at {self.space}, {self.health} health, {self.sanity} sanity, "
            f"{self.clues} clue(s)"
        )

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "controller": self.controller,
            "space": self.space,
            "health": self.health,
            "sanity": self.sanity,
            "clues": self.clues,
            "possessions": list(self.possessions),
            "conditions": list(self.conditions),
            "piece_id": self.piece_id,
        }


class GameState:
    """Everything the robot knows about the game in progress."""

    def __init__(self, board: BoardState | None = None) -> None:
        self.board = board or BoardState()
        self.ancient_one: AncientOne | None = None
        self.doom: int | None = None
        self.omen: str = ""
        self.mystery: str = ""
        self.mysteries_solved: int = 0
        self.investigators: list[InvestigatorState] = []
        self.reserve: list[str] = []
        self.round: int = 0
        self.phase: str = ""
        self.lead: str = ""  # name of the Lead Investigator
        self.unresolved: list[str] = []  # names said that the reference did not recognise
        self.started_at = time.time()
        self.notes: list[str] = []

    # ------------------------------------------------------------------ setup
    def set_ancient_one(self, name: str) -> AncientOne | None:
        found = ancient_one(name)
        if found is None:
            self.unresolved.append(f"ancient one: {name}")
            log.warning("game: ancient one not recognised: %s", name)
            return None
        self.ancient_one = found
        if self.doom is None:
            self.doom = found.starting_doom
        log.info("game: %s", found.describe())
        return found

    def add_investigator(self, name: str, *, controller: str, space: str = "") -> InvestigatorState | None:
        sheet = investigator(name)
        if sheet is None:
            self.unresolved.append(f"investigator: {name}")
            log.warning(
                "game: investigator not recognised: %s (did you mean %s?)",
                name,
                ", ".join(closest_investigators(name)) or "-",
            )
            return None
        if self.by_name(sheet.name) is not None:
            return self.by_name(sheet.name)
        state = InvestigatorState(sheet=sheet, controller=controller, space=space)
        self.investigators.append(state)
        log.info("game: %s", state.describe())
        return state

    def by_name(self, name: str) -> InvestigatorState | None:
        sheet = investigator(name)
        if sheet is None:
            return None
        return next((i for i in self.investigators if i.name == sheet.name), None)

    @property
    def robot_investigator(self) -> InvestigatorState | None:
        return next((i for i in self.investigators if i.is_robot), None)

    @property
    def players(self) -> int:
        return len(self.investigators)

    # ------------------------------------------------------------------ board <-> sheets
    def claim_pieces(self) -> list[str]:
        """Name the pieces a scan found, using where each investigator is meant to be.

        This is the whole point of the spoken setup: the board shows an unnamed piece at
        Shanghai and the briefing says Lily Chen starts there, so the piece is Lily Chen.
        Returns what was claimed, in words, for reading back.
        """
        claimed: list[str] = []
        for state in self.investigators:
            if state.piece_id is not None:
                continue
            here = [p for p in self.board.at(state.space) if not p.name]
            if len(here) != 1:
                continue  # nothing there, or two pieces: ask rather than guess
            piece = here[0]
            self.board.name(piece.id, f"investigator:{state.name}")
            state.piece_id = piece.id
            claimed.append(f"{state.name} at {state.space}")
        if claimed:
            log.info("game: claimed %s", "; ".join(claimed))
        return claimed

    def sync_from_board(self) -> list[str]:
        """Follow each investigator's piece to where the board now says it is."""
        moved: list[str] = []
        for state in self.investigators:
            if state.piece_id is None:
                continue
            piece = self.board.pieces.get(state.piece_id)
            if piece is None or not piece.space or piece.space == state.space:
                continue
            moved.append(f"{state.name} is now at {piece.space} (was {state.space})")
            state.space = piece.space
        return moved

    # ------------------------------------------------------------------ what is still missing
    def missing(self) -> list[str]:
        """What the robot still has to be told before it can play round 1."""
        gaps: list[str] = []
        if self.ancient_one is None:
            gaps.append("which Ancient One we are facing")
        if not self.investigators:
            gaps.append("which investigators are in play and who controls each")
        elif self.robot_investigator is None:
            gaps.append("which investigator I am playing")
        if self.investigators and not self.mystery:
            gaps.append("the active Mystery")
        if self.unresolved:
            gaps.append("names I did not recognise: " + ", ".join(self.unresolved))
        unplaced = [i.name for i in self.investigators if i.piece_id is None]
        if unplaced:
            gaps.append("which piece on the board is " + ", ".join(unplaced))
        return gaps

    @property
    def ready(self) -> bool:
        """True when nothing essential is missing. The Mystery and the pieces can wait."""
        return self.ancient_one is not None and self.robot_investigator is not None

    # ------------------------------------------------------------------ speaking
    def briefing(self) -> str:
        """One spoken paragraph of what the robot understood, to read back for confirmation."""
        if self.ancient_one is None and not self.investigators:
            return "I have nothing set up yet."
        parts: list[str] = []
        if self.ancient_one is not None:
            parts.append(f"We are facing {self.ancient_one.name} and doom starts at {self.doom}.")
        mine = self.robot_investigator
        if mine is not None:
            parts.append(f"I am playing {mine.name}, starting at {mine.space}.")
        others = [i for i in self.investigators if not i.is_robot]
        if others:
            who = ", ".join(f"{i.name} with {i.controller}" for i in others)
            parts.append(f"With me: {who}.")
        if self.reserve:
            parts.append(f"The reserve holds {', '.join(self.reserve)}.")
        if self.mystery:
            parts.append(f"The mystery is {self.mystery}.")
        return " ".join(parts)

    def describe(self) -> str:
        """A fuller, written summary; for logs and for the reasoning prompt."""
        lines = [self.briefing()]
        for state in self.investigators:
            lines.append("  " + state.describe())
        if self.board.pieces:
            lines.append("  board: " + self.board.describe())
        gaps = self.missing()
        if gaps:
            lines.append("  still to ask: " + "; ".join(gaps))
        return "\n".join(lines)

    # ------------------------------------------------------------------ storage
    def record(self) -> dict[str, Any]:
        return {
            "started_at": round(self.started_at, 3),
            "ancient_one": None if self.ancient_one is None else self.ancient_one.name,
            "doom": self.doom,
            "omen": self.omen,
            "mystery": self.mystery,
            "mysteries_solved": self.mysteries_solved,
            "round": self.round,
            "phase": self.phase,
            "lead": self.lead,
            "reserve": list(self.reserve),
            "investigators": [i.record() for i in self.investigators],
            "unresolved": list(self.unresolved),
            "notes": list(self.notes),
            "board": self.board.record(),
            "briefing": self.briefing(),
            "missing": self.missing(),
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.record(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> GameState:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        game = cls()
        if data.get("ancient_one"):
            game.set_ancient_one(data["ancient_one"])
        game.doom = data.get("doom", game.doom)
        for key in ("omen", "mystery", "phase", "lead"):
            setattr(game, key, data.get(key, "") or "")
        game.mysteries_solved = int(data.get("mysteries_solved", 0))
        game.round = int(data.get("round", 0))
        game.reserve = list(data.get("reserve", []))
        game.unresolved = list(data.get("unresolved", []))
        game.notes = list(data.get("notes", []))
        game.started_at = float(data.get("started_at", time.time()))
        for item in data.get("investigators", []):
            state = game.add_investigator(
                item["name"], controller=item.get("controller", ""), space=item.get("space", "")
            )
            if state is None:
                continue
            for key in ("health", "sanity", "clues", "piece_id"):
                if item.get(key) is not None:
                    setattr(state, key, item[key])
            state.possessions = list(item.get("possessions", state.possessions))
            state.conditions = list(item.get("conditions", []))
        return game


__all__ = ["PHASES", "ROBOT", "GameState", "InvestigatorState"]
