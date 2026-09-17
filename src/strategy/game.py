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
MAX_ACTIONS = 2  # GAME_REFERENCE.md: "up to 2 actions, each distinct action at most once per round"
PHASE_NAMES = {"action": "Action Phase", "encounter": "Encounter Phase", "mythos": "Mythos Phase"}
UNDO_DEPTH = 20  # how many changes back the table can go; a round is a handful of them


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
    train_tickets: int = 0  # Travel tickets held; two of any kind is the printed maximum
    ship_tickets: int = 0
    # What this investigator has already done in the round in progress. Cleared by begin_round.
    actions: list[str] = field(default_factory=list)  # distinct action keys, at most MAX_ACTIONS
    components: list[str] = field(default_factory=list)  # components used; each once per round
    encountered: bool = False  # resolved its encounter in this round's Encounter Phase

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
        tickets = []
        if self.train_tickets:
            tickets.append(f"{self.train_tickets} Train ticket(s)")
        if self.ship_tickets:
            tickets.append(f"{self.ship_tickets} Ship ticket(s)")
        carried = (", " + " and ".join(tickets)) if tickets else ""
        return (
            f"{self.name} ({who}) at {self.space}, {self.health} health, {self.sanity} sanity, "
            f"{self.clues} clue(s){carried}"
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
            "train_tickets": self.train_tickets,
            "ship_tickets": self.ship_tickets,
            "actions": list(self.actions),
            "components": list(self.components),
            "encountered": self.encountered,
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
        # What the state looked like before each of the last few things the table asked for, so
        # that "you resolved that wrong, do it again" has somewhere to go. In memory only: a
        # correction belongs to the conversation it happens in, and a file that carried every
        # snapshot would grow with the game for a button nobody presses the next day.
        self.history: list[tuple[str, dict[str, Any]]] = []

    # ------------------------------------------------------------------ taking it back
    # Everything written here is written because somebody said so out loud, and people misspeak,
    # misread a card and change their minds. Until this existed a turn resolved wrongly could not
    # be taken back: told "you resolved that wrong, do it again", the machine refused and the
    # robot restated what it had done instead of saying it could not undo it.
    #
    # One checkpoint per thing the table asked for, not per field: a turn spends two actions and
    # "refaz" means the turn, not half of it. The callers in src/integration/game_session.py are
    # the doors, so the granularity is the tool call.

    def checkpoint(self, label: str) -> None:
        """Remember the state as it is now, before something changes it."""
        self.history.append((label, self.record()))
        del self.history[:-UNDO_DEPTH]

    @staticmethod
    def _undone_by(snapshot: dict[str, Any]) -> dict[str, Any]:
        """The parts of a record undo puts back, for comparing two of them.

        The board is left out because it is the camera's, not the table's: a scan that happened
        between the checkpoint and now must not make a change look like one.
        """
        return {k: v for k, v in snapshot.items() if k not in ("board", "briefing", "missing")}

    def undo(self) -> str | None:
        """Put the state back as it was before the last recorded change. Returns what was undone.

        A door that was asked for and refused - a phase that could not advance, a note that was
        already written - leaves a checkpoint that changes nothing, and undoing that would look
        to the table like the robot ignoring them. Those are skipped, so one "refaz" always
        reaches one real change.
        """
        while self.history:
            label, snapshot = self.history.pop()
            if self._undone_by(snapshot) == self._undone_by(self.record()):
                continue  # nothing happened after this checkpoint; keep looking
            self.apply_record(snapshot, quiet=True)
            log.info("game: undid %s; back to %s", label, self.where_we_are())
            return label
        return None

    @property
    def undoable(self) -> str:
        """What "do it again" would take back, or "" when there is nothing to take back."""
        now = self._undone_by(self.record())
        for label, snapshot in reversed(self.history):
            if self._undone_by(snapshot) != now:
                return label
        return ""

    # ------------------------------------------------------------------ notes
    def note(self, text: str) -> str | None:
        """Write down something the table said that no other field holds.

        Gates, monsters, what was agreed - the state has a field for the setup and one for each
        investigator, and nothing for "there is a Gate open in Rome and a Serpent People on it".
        That is what this is: the table's own words, kept as they were said, read back by
        game_state. Returns the note, or None when it says nothing new.
        """
        text = " ".join(text.split())
        if not text or text in self.notes:
            return None
        self.notes.append(text)
        return text

    # ------------------------------------------------------------------ skill tests
    def test_result(self, name: str, dice: list[int]) -> dict[str, Any]:
        """Count the successes in a roll somebody read out loud, and say whether it passed.

        GAME_REFERENCE.md, Skill Tests: a 5 or 6 is a success and any success passes; Blessed
        counts 4s as well and Cursed counts only 6s. This is arithmetic, not judgement, and it
        belongs here for the same reason the legal moves do: on 2026-09-17 the robot quoted the
        rule correctly and then called two 4s "dois sucessos", which silently changes the game.
        """
        state = self.by_name(name) if name else self.robot_investigator
        conditions = {c.strip().lower() for c in (state.conditions if state else [])}
        blessed, cursed = "blessed" in conditions, "cursed" in conditions
        floor = 4 if blessed else 6 if cursed else 5
        rolled = [int(d) for d in dice if isinstance(d, (int, float))]
        successes = sum(1 for d in rolled if d >= floor)
        return {
            "investigator": state.name if state else name,
            "dice": rolled,
            "successes": successes,
            "passed": successes > 0,
            "counts_as_success": f"{floor} or more",
            "blessed": blessed,
            "cursed": cursed,
        }

    def apply_effect(
        self,
        name: str = "",
        *,
        health: int = 0,
        sanity: int = 0,
        clues: int = 0,
        gain: list[str] | None = None,
        lose: list[str] | None = None,
        conditions_gained: list[str] | None = None,
        conditions_lost: list[str] | None = None,
    ) -> dict[str, Any]:
        """What a resolved encounter or effect did to an investigator, written into the state.

        Deltas, not totals, because the table says "perde 1 de Sanity" and never "fica com 5".
        Health and Sanity are held between 0 and the sheet's maximum. Until this existed the
        robot said "vou atualizar isso aqui" and nothing moved: the state has the fields and the
        conversation had no door to them.
        """
        state = self.by_name(name) if name else self.robot_investigator
        if state is None:
            return {"applied": False, "note": f"I do not know an investigator called {name!r}."}
        before = (state.health, state.sanity, state.clues)
        state.health = max(0, min(state.sheet.health, state.health + int(health)))
        state.sanity = max(0, min(state.sheet.sanity, state.sanity + int(sanity)))
        state.clues = max(0, state.clues + int(clues))
        for card in gain or []:
            if card and card not in state.possessions:
                state.possessions.append(card)
        for card in lose or []:
            if card in state.possessions:
                state.possessions.remove(card)
        for item in conditions_gained or []:
            if item and item not in state.conditions:
                state.conditions.append(item)
        for item in conditions_lost or []:
            if item in state.conditions:
                state.conditions.remove(item)
        log.info(
            "game: %s now %d/%d health, %d/%d sanity, %d clue(s)",
            state.name,
            state.health,
            state.sheet.health,
            state.sanity,
            state.sheet.sanity,
            state.clues,
        )
        return {
            "applied": True,
            "investigator": state.name,
            "health": state.health,
            "sanity": state.sanity,
            "clues": state.clues,
            "possessions": list(state.possessions),
            "conditions": list(state.conditions),
            "defeated": state.defeated,
            "changed": (state.health, state.sanity, state.clues) != before,
        }

    # ------------------------------------------------------------------ the round
    # GAME_REFERENCE.md, "Round structure": every round is Action Phase -> Encounter Phase ->
    # Mythos Phase, the Lead Investigator acts first and then round the table. Until this
    # existed the robot re-derived the same plan from an unchanged state every time it was
    # asked to play, so it repeated its strategy and never got anywhere.

    @property
    def started(self) -> bool:
        """Has round 1 begun? Round 0 with no phase is a game that is still being set up."""
        return self.round > 0 and self.phase in PHASES

    @property
    def phase_name(self) -> str:
        return PHASE_NAMES.get(self.phase, "")

    def begin_round(self) -> int:
        """Open the next round at the Action Phase and forget what everyone did in the last."""
        self.round += 1
        self.phase = PHASES[0]
        for state in self.investigators:
            state.actions = []
            state.components = []
            state.encountered = False
        return self.round

    def advance_phase(self) -> str:
        """Move to the next phase, opening a new round after the Mythos Phase. Returns the phase."""
        if not self.started:
            self.begin_round()
        elif self.phase == PHASES[-1]:
            self.begin_round()
        else:
            self.phase = PHASES[PHASES.index(self.phase) + 1]
        return self.phase

    def order(self) -> list[InvestigatorState]:
        """Turn order: the Lead Investigator first, then the others as they sit round the table."""
        names = [i.name for i in self.investigators]
        start = names.index(self.lead) if self.lead in names else 0
        return self.investigators[start:] + self.investigators[:start]

    def actions_left(self, name: str) -> int:
        state = self.by_name(name)
        return 0 if state is None else max(0, MAX_ACTIONS - len(state.actions))

    def may_act(self, name: str, action: str, component: str = "") -> tuple[bool, str]:
        """May this investigator take this action now? ``(allowed, why not)``."""
        state = self.by_name(name)
        if state is None:
            return False, f"{name} is not in this game"
        if self.phase != PHASES[0]:
            return False, f"it is the {self.phase_name}, not the Action Phase"
        if not self.actions_left(name):
            return False, f"{name} has already taken {MAX_ACTIONS} actions this round"
        # A Component Action is limited per component rather than per action: the reference
        # gives it its own rule, "each component once per round", which would say nothing if
        # the action itself could only be taken once.
        if component:
            if component in state.components:
                return False, f"{component} has already been used this round"
            return True, ""
        if action in state.actions:
            return False, f"{name} has already used {action} this round"
        return True, ""

    def record_action(self, name: str, action: str, component: str = "") -> tuple[bool, str]:
        """Write down that this investigator took this action. ``(recorded, why not)``."""
        allowed, why = self.may_act(name, action, component)
        if not allowed:
            return False, why
        state = self.by_name(name)
        assert state is not None
        if action not in state.actions:
            state.actions.append(action)
        if component:
            state.components.append(component)
        return True, ""

    def record_encounter(self, name: str) -> tuple[bool, str]:
        state = self.by_name(name)
        if state is None:
            return False, f"{name} is not in this game"
        if self.phase != PHASES[1]:
            return False, f"it is the {self.phase_name}, not the Encounter Phase"
        if state.encountered:
            return False, f"{name} has already had an encounter this round"
        state.encountered = True
        return True, ""

    def to_act(self) -> InvestigatorState | None:
        """Whose turn it is in the Action Phase, in turn order; None when everyone has acted."""
        if self.phase != PHASES[0]:
            return None
        for state in self.order():
            if not state.defeated and self.actions_left(state.name):
                return state
        return None

    def to_encounter(self) -> InvestigatorState | None:
        if self.phase != PHASES[1]:
            return None
        for state in self.order():
            if not state.defeated and not state.encountered:
                return state
        return None

    def where_we_are(self) -> str:
        """One line for the agent: the round, the phase and who the table is waiting on."""
        if not self.started:
            return "the game has not started; round 1 has not been opened yet"
        head = f"round {self.round}, {self.phase_name}"
        if self.phase == PHASES[0]:
            waiting = self.to_act()
            if waiting is None:
                return f"{head}; everyone has acted, the Encounter Phase is next"
            return f"{head}; {waiting.name} to act ({self.actions_left(waiting.name)} action(s) left)"
        if self.phase == PHASES[1]:
            waiting = self.to_encounter()
            if waiting is None:
                return f"{head}; everyone has had an encounter, the Mythos Phase is next"
            return f"{head}; {waiting.name} has an encounter to resolve"
        # Naming the Lead matters: the line used to say "the Lead Investigator draws the Mythos
        # card" and the robot, which was the Lead, asked the table whose turn it was.
        lead = self.lead or (self.order()[0].name if self.investigators else "")
        who = (
            "I draw"
            if lead and lead == (self.robot_investigator.name if self.robot_investigator else "")
            else f"{lead} draws"
            if lead
            else "the Lead Investigator draws"
        )
        return f"{head}; {who} the Mythos card, then the round ends"

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

    def add_investigator(
        self, name: str, *, controller: str, space: str = "", quiet: bool = False
    ) -> InvestigatorState | None:
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
        if not quiet:
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

    def apply_record(self, data: dict[str, Any], *, quiet: bool = False) -> GameState:
        """Make this state be what ``record()`` wrote: reading a saved game, and undoing.

        ``self.board`` is deliberately left alone. It is the camera's live view of the table, and
        a scan is not something anybody can take back by saying "resolveu errado". A saved file
        never carried it either.

        Investigators already in play are written over in place rather than replaced. Undo runs
        in the middle of a live session, where the turn taker and the board link are holding
        those objects; handing them a fresh list would leave every one of them pointing at a
        sheet nobody updates again.
        """
        self.ancient_one, self.doom = None, None
        if data.get("ancient_one"):
            self.set_ancient_one(data["ancient_one"])
        self.doom = data.get("doom", self.doom)
        for key in ("omen", "mystery", "phase", "lead"):
            setattr(self, key, data.get(key, "") or "")
        self.mysteries_solved = int(data.get("mysteries_solved", 0))
        self.round = int(data.get("round", 0))
        self.reserve = list(data.get("reserve", []))
        self.unresolved = list(data.get("unresolved", []))
        self.notes = list(data.get("notes", []))
        self.started_at = float(data.get("started_at", time.time()))
        saved = data.get("investigators", [])
        keep = {investigator(item["name"]).name for item in saved if investigator(item.get("name", ""))}
        self.investigators = [i for i in self.investigators if i.name in keep]
        for item in saved:
            # Quietly, because add_investigator logs the sheet and the saved Health, Sanity and
            # Clues are restored below it. Logging at creation printed a full sheet for an
            # investigator who had taken damage, which sent the owner and this session chasing an
            # edit that had in fact been written.
            state = self.add_investigator(
                item["name"],
                controller=item.get("controller", ""),
                space=item.get("space", ""),
                quiet=True,
            )
            if state is None:
                continue
            state.controller = item.get("controller", state.controller)
            state.space = item.get("space", "") or state.space
            for key in ("health", "sanity", "clues", "piece_id", "train_tickets", "ship_tickets"):
                if item.get(key) is not None:
                    setattr(state, key, item[key])
            state.possessions = list(item.get("possessions", state.possessions))
            state.conditions = list(item.get("conditions", []))
            state.actions = list(item.get("actions", []))
            state.components = list(item.get("components", []))
            state.encountered = bool(item.get("encountered", False))
            if not quiet:
                log.info("game: %s", state.describe())
        return self

    @classmethod
    def load(cls, path: Path) -> GameState:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls().apply_record(data)


__all__ = [
    "MAX_ACTIONS",
    "PHASES",
    "PHASE_NAMES",
    "ROBOT",
    "UNDO_DEPTH",
    "GameState",
    "InvestigatorState",
]
