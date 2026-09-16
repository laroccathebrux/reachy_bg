"""Everything the investigator is allowed to do this turn, and nothing it is not.

    situation = situation_of(game, game.robot_investigator)
    for candidate in candidates(game, situation):
        print(candidate.describe())          # "Travel to Tokyo: Shanghai to Tokyo by ship"
    for plan in plans(game, situation):
        print(plan.describe())               # "Travel to Tokyo, then Acquire Assets at Tokyo"
    refusals(game, situation)                # why Rest is not on the list: a Monster is here

The Action Phase is **up to two actions, each distinct action at most once, one fully resolved
before the next** (Rulebook, Action Phase). So the unit here is the whole turn, not one action:
"Travel to Tokyo and buy there" is a different idea from "buy here", and a plan built one action
at a time can never see it. ``plans`` walks the second action from the state the first one
leaves behind - after a Travel the city is another city, and the reserve on offer is judged from
there.

**Where each piece of the decision comes from** is deliberate, and it is the reason this module
has almost no game text in it:

- The six actions and their conditions are printed on the game's own reference card and cannot
  change while the box is the 2013 box. They live in ``ACTIONS`` as a table to read, not as
  conditions scattered through the code.
- What a card *does* is never written here. It comes from ``src/rag/cards/eldritch_base_game.json``
  with the printed wording, which is also where Component Actions come from: a card whose text
  begins "Action:" gives its holder an action, and the card says what it is.
- What the robot cannot know stays unknown. The camera sees *a piece* at Rome, not *which
  Monster*; the reserve holds four cards whose effects are known and whose printed values are
  not. Those become ``unknowns`` on the candidate, so the layer above asks instead of assuming.

Nothing here scores anything. A candidate is a legal option, in the game's own words; which one
is worth taking is decided in ``src/strategy/decide.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

from src.logger import get_logger
from src.strategy import map_graph
from src.strategy.game import GameState, InvestigatorState
from src.vision.spaces import BY_NAME

log = get_logger(__name__)

MAX_ACTIONS = 2  # per investigator, per Action Phase
MAX_TICKETS = 2  # Travel tickets of both kinds together

TRAVEL = "travel"
PREPARE = "prepare_for_travel"
ACQUIRE = "acquire_assets"
REST = "rest"
TRADE = "trade"
COMPONENT = "component"

RULEBOOK = "Rulebook, Action Phase"


@dataclass(frozen=True)
class ActionRule:
    """One of the six actions of the Action Phase, with the conditions printed beside it."""

    key: str
    name: str  # exactly as printed on the components
    summary: str
    source: str = RULEBOOK
    space_kinds: tuple[str, ...] = ()  # empty means any space
    needs_empty_of_monsters: bool = False
    needs_another_investigator: bool = False

    def describe(self) -> str:
        return f"{self.name}: {self.summary} ({self.source})"


# The Action Phase as the reference card prints it. Read it, do not rewrite it in code.
ACTIONS: tuple[ActionRule, ...] = (
    ActionRule(
        TRAVEL,
        "Travel",
        "move to an adjacent space, then spend any number of tickets for one extra move each",
    ),
    ActionRule(
        PREPARE,
        "Prepare for Travel",
        "gain 1 Train or Ship ticket matching a path that touches this space; two tickets is the maximum",
        space_kinds=("city",),
    ),
    ActionRule(
        ACQUIRE,
        "Acquire Assets",
        "test Influence and gain Reserve cards whose total value is at most the successes",
        space_kinds=("city",),
        needs_empty_of_monsters=True,
    ),
    ActionRule(
        REST,
        "Rest",
        "recover 1 Health and 1 Sanity",
        needs_empty_of_monsters=True,
    ),
    ActionRule(
        TRADE,
        "Trade",
        "exchange possessions with an investigator on this space; never Conditions, Health, Sanity or Improvements",
        needs_another_investigator=True,
    ),
    ActionRule(
        COMPONENT,
        "Component Action",
        'use an "Action:" printed on a sheet, a possession or a Condition; each component once per round',
    ),
)
BY_KEY: dict[str, ActionRule] = {a.key: a for a in ACTIONS}


@lru_cache(maxsize=1)
def _cards() -> dict[str, dict[str, Any]]:
    """The 2013 cards by lowercase name, straight from the versioned file (no Qdrant, no network)."""
    from src.rag.ingest_assets import CARDS_FILE

    try:
        items = json.loads(CARDS_FILE.read_text(encoding="utf-8"))
    except OSError as exc:  # pragma: no cover - the file ships with the repository
        log.warning("moves: cannot read %s: %s", CARDS_FILE, exc)
        return {}
    return {str(c.get("name", "")).lower(): c for c in items if isinstance(c, dict)}


def card_effect(name: str) -> str:
    """The printed effect of a card, or "" when this box has no such card."""
    return str(_cards().get(name.strip().lower(), {}).get("effect") or "")


def component_action_text(effect: str) -> str:
    """The "Action:" clause of a printed effect, or "" when the card grants no action.

    A card's text holds several clauses separated by "|" (Reckoning, flip sides, and so on);
    only the one that starts with "Action:" is something an investigator may do on its turn.
    """
    for clause in effect.split("|"):
        clause = clause.strip()
        if clause.startswith("Action:"):
            return clause
    return ""


@dataclass(frozen=True)
class Situation:
    """The investigator as it stands right now: what the legality of an action depends on.

    Kept apart from ``InvestigatorState`` because ``plans`` has to ask "and what if I had already
    travelled?" without moving anything on the real board.
    """

    name: str
    space: str
    health: int
    sanity: int
    clues: int
    train_tickets: int = 0
    ship_tickets: int = 0
    possessions: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    max_health: int = 0
    max_sanity: int = 0

    @property
    def tickets(self) -> int:
        return self.train_tickets + self.ship_tickets

    @property
    def delayed(self) -> bool:
        return any(c.strip().lower() == "delayed" for c in self.conditions)

    @property
    def detained(self) -> bool:
        return any(c.strip().lower() == "detained" for c in self.conditions)

    def moved_to(self, space: str, *, train_spent: int = 0, ship_spent: int = 0) -> Situation:
        return replace(
            self,
            space=space,
            train_tickets=self.train_tickets - train_spent,
            ship_tickets=self.ship_tickets - ship_spent,
        )

    def with_ticket(self, kind: str) -> Situation:
        if kind == map_graph.TRAIN:
            return replace(self, train_tickets=self.train_tickets + 1)
        return replace(self, ship_tickets=self.ship_tickets + 1)

    def rested(self) -> Situation:
        return replace(
            self,
            health=min(self.health + 1, self.max_health or self.health + 1),
            sanity=min(self.sanity + 1, self.max_sanity or self.sanity + 1),
        )


def situation_of(game: GameState, state: InvestigatorState) -> Situation:
    """Snapshot one investigator of the game state."""
    return Situation(
        name=state.name,
        space=state.space,
        health=state.health,
        sanity=state.sanity,
        clues=state.clues,
        train_tickets=state.train_tickets,
        ship_tickets=state.ship_tickets,
        possessions=tuple(state.possessions),
        conditions=tuple(state.conditions),
        max_health=state.sheet.health,
        max_sanity=state.sheet.sanity,
    )


@dataclass(frozen=True)
class Candidate:
    """One legal action, in the game's own words, with what the robot does not know about it."""

    key: str
    detail: str = ""  # "to Tokyo", "Train ticket", the card's name
    route: map_graph.Route | None = None
    effect: str = ""  # printed text, for a Component Action
    unknowns: tuple[str, ...] = ()
    partner: str = ""

    @property
    def rule(self) -> ActionRule:
        return BY_KEY[self.key]

    @property
    def name(self) -> str:
        return self.rule.name

    @property
    def destination(self) -> str:
        return self.route.destination if self.route is not None else ""

    def describe(self, start: str = "") -> str:
        head = f"{self.name}{(' ' + self.detail) if self.detail else ''}"
        if self.route is not None and start:
            return f"{head}: {self.route.describe(start)}"
        if self.effect:
            return f"{head}: {self.effect}"
        return head

    def record(self) -> dict[str, Any]:
        return {
            "action": self.name,
            "key": self.key,
            "detail": self.detail,
            "destination": self.destination,
            "effect": self.effect,
            "unknowns": list(self.unknowns),
        }


@dataclass
class TurnPlan:
    """Up to two distinct actions, in the order they would be resolved."""

    actions: tuple[Candidate, ...]
    start: str
    id: int = 0
    unknowns: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ends_at(self) -> str:
        here = self.start
        for action in self.actions:
            if action.destination:
                here = action.destination
        return here

    def describe(self) -> str:
        here, parts = self.start, []
        for action in self.actions:
            parts.append(action.describe(here))
            if action.destination:
                here = action.destination
        return ", then ".join(parts) if parts else "do nothing"

    def record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "ends_at": self.ends_at,
            "actions": [a.record() for a in self.actions],
            "unknowns": list(self.unknowns),
            "text": self.describe(),
        }


def others_at(game: GameState, space: str, *, except_name: str = "") -> list[str]:
    """The other investigators standing on ``space``."""
    return [
        i.name for i in game.investigators if i.space == space and i.name != except_name and not i.defeated
    ]


def unnamed_pieces_at(game: GameState, space: str) -> int:
    """Pieces the camera sees on ``space`` that nothing has named yet.

    A Monster, a Gate and a Clue all look like "a piece" from a metre away, and the difference
    decides whether Rest and Acquire Assets are legal at all. So they are counted, never guessed.
    """
    return sum(1 for piece in game.board.at(space) if not piece.name)


def monsters_at(game: GameState, space: str) -> list[str]:
    """Pieces on ``space`` that somebody has named as a Monster."""
    return [
        (p.name or "").split(":", 1)[-1]
        for p in game.board.at(space)
        if (p.name or "").lower().startswith("monster")
    ]


def _monster_note(game: GameState, space: str) -> tuple[bool, str]:
    """(is a Monster known to be here, what is uncertain about that)."""
    named = monsters_at(game, space)
    if named:
        return True, ""
    unknown = unnamed_pieces_at(game, space)
    if unknown:
        plural = "pieces" if unknown > 1 else "piece"
        return False, f"{unknown} {plural} at {space} that nobody has named; is any of them a Monster?"
    return False, ""


def _space_kind(space: str) -> str:
    found = BY_NAME.get(space)
    return found.kind if found else ""


def candidates(game: GameState, situation: Situation) -> list[Candidate]:
    """Every action this investigator may take right now, one entry per distinct choice.

    Travel is one candidate per destination (a different destination is a different decision);
    the others are one candidate each, except Component Actions, one per card that prints one.
    """
    if situation.delayed or situation.detained:
        return []
    out: list[Candidate] = []
    monster_here, monster_doubt = _monster_note(game, situation.space)
    city = _space_kind(situation.space) == "city"

    # --- Travel. The path table is map_graph's; whether it has been checked against the board
    # is map_graph.VERIFIED, and the layer that decides to act is the one that must refuse.
    for route in map_graph.travel_options(
        situation.space, train=situation.train_tickets, ship=situation.ship_tickets
    ):
        out.append(Candidate(TRAVEL, f"to {route.destination}", route=route))

    # --- Prepare for Travel -------------------------------------------------------------
    if city and situation.tickets < MAX_TICKETS:
        for kind in map_graph.ticket_kinds(situation.space):
            out.append(Candidate(PREPARE, f"{kind.capitalize()} ticket"))

    # --- Acquire Assets -----------------------------------------------------------------
    if city and not monster_here:
        unknowns = []
        if monster_doubt:
            unknowns.append(monster_doubt)
        if game.reserve:
            unknowns.append(
                "the printed value of each Reserve card: I know what "
                + ", ".join(game.reserve)
                + " do, not what they cost"
            )
        else:
            unknowns.append("which cards are in the Reserve")
        out.append(Candidate(ACQUIRE, f"at {situation.space}", unknowns=tuple(unknowns)))

    # --- Rest ---------------------------------------------------------------------------
    if not monster_here and (
        situation.health < (situation.max_health or situation.health)
        or situation.sanity < (situation.max_sanity or situation.sanity)
    ):
        out.append(
            Candidate(REST, f"at {situation.space}", unknowns=(monster_doubt,) if monster_doubt else ())
        )

    # --- Trade --------------------------------------------------------------------------
    for other in others_at(game, situation.space, except_name=situation.name):
        out.append(Candidate(TRADE, f"with {other}", partner=other))

    # --- Component Actions --------------------------------------------------------------
    for card in situation.possessions:
        effect = card_effect(card)
        if not effect:
            out.append(
                Candidate(
                    COMPONENT,
                    card,
                    unknowns=(f"what {card} does: it is not in the base-game card list I have",),
                )
            )
            continue
        action = component_action_text(effect)
        if action:
            out.append(Candidate(COMPONENT, card, effect=action))
    for condition in situation.conditions:
        action = component_action_text(card_effect(condition))
        if action:
            out.append(Candidate(COMPONENT, condition, effect=action))
    return out


def refusals(game: GameState, situation: Situation) -> list[str]:
    """Why an action a person might expect is not on the list - in the rule's own terms.

    The robot has to be able to say "I cannot Rest, there is a Monster here" rather than simply
    not offering it; a silent omission looks like a mistake to the table.
    """
    out: list[str] = []
    if situation.delayed:
        out.append(f"{situation.name} is Delayed, so the token stands up instead of acting ({RULEBOOK})")
        return out
    if situation.detained:
        out.append(f"{situation.name} is Detained, so only the action printed on that card ({RULEBOOK})")
        return out
    monster_here, _ = _monster_note(game, situation.space)
    city = _space_kind(situation.space) == "city"
    if monster_here:
        out.append(f"a Monster is at {situation.space}: no Rest and no Acquire Assets ({RULEBOOK})")
    if not city:
        out.append(
            f"{situation.space} is not a City space: no Acquire Assets, no Prepare for Travel ({RULEBOOK})"
        )
    if city and situation.tickets >= MAX_TICKETS:
        out.append(
            f"{situation.name} already holds {MAX_TICKETS} tickets, the printed maximum: no Prepare for Travel"
        )
    if not others_at(game, situation.space, except_name=situation.name):
        out.append(f"nobody else is at {situation.space}: no Trade ({RULEBOOK})")
    if situation.health >= (situation.max_health or 0) and situation.sanity >= (situation.max_sanity or 0):
        out.append(f"{situation.name} is at full Health and Sanity: Rest would recover nothing")
    return out


def _after(situation: Situation, candidate: Candidate) -> Situation:
    """The investigator after that action resolves, as far as legality is concerned."""
    if candidate.key == TRAVEL and candidate.route is not None:
        return situation.moved_to(
            candidate.route.destination,
            train_spent=candidate.route.train_spent,
            ship_spent=candidate.route.ship_spent,
        )
    if candidate.key == PREPARE:
        kind = map_graph.TRAIN if candidate.detail.lower().startswith("train") else map_graph.SHIP
        return situation.with_ticket(kind)
    if candidate.key == REST:
        return situation.rested()
    return situation


def plans(game: GameState, situation: Situation) -> list[TurnPlan]:
    """Every legal turn: one action, or two distinct ones in order, the second seen from the first.

    A second Travel is not offered because an action may be taken only once per round, and two
    Component Actions are allowed only when they are different components.
    """
    out: list[TurnPlan] = []
    first_round = candidates(game, situation)
    for first in first_round:
        out.append(TurnPlan((first,), situation.space, unknowns=first.unknowns))
        moved = _after(situation, first)
        for second in candidates(game, moved):
            if second.key == first.key and not (second.key == COMPONENT and second.detail != first.detail):
                continue
            out.append(
                TurnPlan(
                    (first, second),
                    situation.space,
                    unknowns=tuple(dict.fromkeys(first.unknowns + second.unknowns)),
                )
            )
    for index, plan in enumerate(out, 1):
        plan.id = index
    return out


__all__ = [
    "ACTIONS",
    "ACQUIRE",
    "BY_KEY",
    "COMPONENT",
    "Candidate",
    "MAX_ACTIONS",
    "MAX_TICKETS",
    "PREPARE",
    "REST",
    "TRADE",
    "TRAVEL",
    "ActionRule",
    "Situation",
    "TurnPlan",
    "candidates",
    "card_effect",
    "component_action_text",
    "monsters_at",
    "others_at",
    "plans",
    "refusals",
    "situation_of",
    "unnamed_pieces_at",
]
