"""The paths printed between the spaces: what "an adjacent space" means, and how far a ticket goes.

    neighbours("Shanghai")                  # -> the paths that touch Shanghai
    kinds_touching("Shanghai")              # -> {"train", "ship"}: what Prepare for Travel may give
    travel_options("Shanghai", train=1)     # -> every space reachable this turn, and the route
    steps_between("Shanghai", "Rome")       # -> moves on foot, ignoring tickets

``src/vision/spaces.py`` knows *where* each space is, which is all the camera needs. It does not
know which space touches which, and without that the robot cannot take three of the six actions
of the Action Phase: Travel is "move to an adjacent space", a ticket is "one extra move along a
Train path" and Prepare for Travel gives "a ticket matching a path that touches this space".

So this is the board's other half, read off the printed map: 56 paths of three kinds. The board
legend names them Train (red dashes), Ship (a white line) and Uncharted (yellow dots), and the
difference is what a ticket may be spent on - a Train ticket buys an extra move along a Train
path and nothing else. An Uncharted path can only ever be walked as the free first move.

**Three paths leave the map and come back on the other side** (``wrap=True``): the board's left
and right edges are the same Pacific. 1 and 19 are the Bering Strait, 2 and Tokyo the northern
crossing, 3 and Sydney the southern one.

The travel rule itself (Rulebook, Action Phase, and docs/GAME_REFERENCE.md): move to one
adjacent space along a path of any kind, then spend any number of tickets, each buying one more
move along a path matching that ticket. ``travel_options`` is that rule and nothing else - what
is *worth* travelling to is decided elsewhere.

**Where the data comes from.** Nothing here was recalled from memory: every path was traced on
``BOARD_REFERENCE_IMAGE`` (the top-down picture of the real 2013 board) at 4-7x magnification,
one region at a time. ``scripts/draw_map_graph.py`` draws this table back over that picture so a
person can check it in one look. Until the owner has done that, ``VERIFIED`` stays False and
callers that move a piece must refuse to act: a path that is not on the board is a rule the
robot would be inventing, which is exactly what CLAUDE.md forbids.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from src.logger import get_logger
from src.vision.spaces import BY_NAME

log = get_logger(__name__)

# Set to True only after a person has compared scripts/draw_map_graph.py's picture with the
# board. Until then the table is a careful reading, not a checked fact.
VERIFIED = False

TRAIN = "train"
SHIP = "ship"
UNCHARTED = "uncharted"
KINDS = (TRAIN, SHIP, UNCHARTED)


@dataclass(frozen=True)
class Path:
    """One printed path between two spaces. ``wrap`` marks the ones that cross the map's edge."""

    a: str
    b: str
    kind: str
    wrap: bool = False

    def other(self, space: str) -> str:
        if space == self.a:
            return self.b
        if space == self.b:
            return self.a
        raise ValueError(f"{space} is not an end of {self.a}-{self.b}")

    def describe(self) -> str:
        edge = ", across the edge of the map" if self.wrap else ""
        return f"{self.a} to {self.b} by {self.kind}{edge}"


# Read off the printed board, west to east. Names are the ones in src/vision/spaces.py.
PATHS: tuple[Path, ...] = (
    # --- Train: the red dashes, over land ---
    Path("San Francisco", "5", TRAIN),
    Path("5", "Arkham", TRAIN),
    Path("San Francisco", "6", TRAIN),
    Path("6", "Arkham", TRAIN),
    Path("6", "7", TRAIN),
    Path("Rome", "14", TRAIN),
    Path("14", "16", TRAIN),
    Path("14", "Istanbul", TRAIN),
    Path("Rome", "Istanbul", TRAIN),
    Path("16", "Istanbul", TRAIN),
    Path("16", "Tunguska", TRAIN),
    Path("Tunguska", "19", TRAIN),
    Path("19", "Shanghai", TRAIN),
    Path("Istanbul", "17", TRAIN),
    Path("17", "Shanghai", TRAIN),
    # --- Ship: the white lines, over water ---
    Path("1", "San Francisco", SHIP),
    Path("1", "19", SHIP, wrap=True),
    Path("2", "San Francisco", SHIP),
    Path("2", "Tokyo", SHIP, wrap=True),
    Path("San Francisco", "7", SHIP),
    Path("7", "8", SHIP),
    Path("7", "Buenos Aires", SHIP),
    Path("8", "Arkham", SHIP),
    Path("8", "10", SHIP),
    Path("9", "Arkham", SHIP),
    Path("Arkham", "London", SHIP),
    Path("London", "13", SHIP),
    Path("London", "Rome", SHIP),
    Path("Rome", "The Pyramids", SHIP),
    Path("Rome", "10", SHIP),
    Path("10", "15", SHIP),
    Path("3", "Buenos Aires", SHIP),
    Path("3", "Sydney", SHIP, wrap=True),
    Path("Buenos Aires", "11", SHIP),
    Path("Buenos Aires", "12", SHIP),
    Path("11", "15", SHIP),
    Path("12", "Antarctica", SHIP),
    Path("Antarctica", "Sydney", SHIP),
    Path("15", "17", SHIP),
    Path("15", "18", SHIP),
    Path("18", "Sydney", SHIP),
    Path("17", "20", SHIP),
    Path("20", "Shanghai", SHIP),
    Path("Shanghai", "Tokyo", SHIP),
    Path("19", "Tokyo", SHIP),
    Path("Tokyo", "Sydney", SHIP),
    # --- Uncharted: the yellow dots, where no line and no rail goes ---
    Path("1", "4", UNCHARTED),
    Path("4", "5", UNCHARTED),
    Path("7", "The Amazon", UNCHARTED),
    Path("The Amazon", "Buenos Aires", UNCHARTED),
    Path("10", "The Pyramids", UNCHARTED),
    Path("The Pyramids", "The Heart of Africa", UNCHARTED),
    Path("The Heart of Africa", "15", UNCHARTED),
    Path("The Himalayas", "17", UNCHARTED),
    Path("The Himalayas", "Shanghai", UNCHARTED),
    Path("21", "Sydney", UNCHARTED),
)


def _index() -> dict[str, tuple[Path, ...]]:
    table: dict[str, list[Path]] = {name: [] for name in BY_NAME}
    for path in PATHS:
        for end in (path.a, path.b):
            if end not in table:
                raise ValueError(f"map_graph: {end} is not a space in src/vision/spaces.py")
        if path.kind not in KINDS:
            raise ValueError(f"map_graph: {path.describe()} has an unknown kind")
        table[path.a].append(path)
        table[path.b].append(path)
    return {name: tuple(paths) for name, paths in table.items()}


BY_SPACE: dict[str, tuple[Path, ...]] = _index()


def neighbours(space: str) -> tuple[Path, ...]:
    """Every path that touches ``space``. Empty for a name the board does not have."""
    return BY_SPACE.get(space, ())


def adjacent(space: str) -> tuple[str, ...]:
    """The spaces one move away on foot, whatever the kind of path."""
    return tuple(sorted({p.other(space) for p in neighbours(space)}))


def kinds_touching(space: str) -> set[str]:
    """Which kinds of path touch ``space`` - the tickets Prepare for Travel may give there.

    Uncharted paths are included because they touch the space; the caller keeps only Train and
    Ship, since those are the only tickets that exist.
    """
    return {p.kind for p in neighbours(space)}


def ticket_kinds(space: str) -> tuple[str, ...]:
    """The tickets Prepare for Travel can actually give at ``space``: Train and Ship only."""
    return tuple(k for k in (TRAIN, SHIP) if k in kinds_touching(space))


@dataclass(frozen=True)
class Route:
    """One way to end a Travel action somewhere: the spaces walked and the tickets it costs."""

    destination: str
    paths: tuple[Path, ...]
    train_spent: int = 0
    ship_spent: int = 0

    @property
    def moves(self) -> int:
        return len(self.paths)

    def spaces(self, start: str) -> tuple[str, ...]:
        here, out = start, []
        for path in self.paths:
            here = path.other(here)
            out.append(here)
        return tuple(out)

    def describe(self, start: str) -> str:
        if not self.paths:
            return f"stay at {start}"
        legs = []
        here = start
        for path in self.paths:
            nxt = path.other(here)
            legs.append(f"{here} to {nxt} by {path.kind}")
            here = nxt
        cost = []
        if self.train_spent:
            cost.append(f"{self.train_spent} Train ticket" + ("s" if self.train_spent > 1 else ""))
        if self.ship_spent:
            cost.append(f"{self.ship_spent} Ship ticket" + ("s" if self.ship_spent > 1 else ""))
        tail = f" (spending {' and '.join(cost)})" if cost else ""
        return "; ".join(legs) + tail


def travel_options(space: str, *, train: int = 0, ship: int = 0) -> list[Route]:
    """Where a Travel action from ``space`` can end, with the tickets each route costs.

    The rule, and only the rule: one move along a path of any kind, then one extra move per
    ticket spent, a Train ticket only along a Train path and a Ship ticket only along a Ship
    path. Routes never revisit a space, and a destination reachable in several ways is returned
    once, by its cheapest route (fewest tickets, then fewest moves).
    """
    if space not in BY_SPACE:
        return []
    best: dict[str, Route] = {}

    def cheaper(candidate: Route, current: Route) -> bool:
        return (candidate.train_spent + candidate.ship_spent, candidate.moves) < (
            current.train_spent + current.ship_spent,
            current.moves,
        )

    def walk(here: str, taken: tuple[Path, ...], seen: frozenset[str], t_left: int, s_left: int) -> None:
        if taken:
            route = Route(here, taken, train - t_left, ship - s_left)
            if here not in best or cheaper(route, best[here]):
                best[here] = route
        for path in neighbours(here):
            nxt = path.other(here)
            if nxt in seen:
                continue
            if not taken:  # the free first move: any kind of path
                walk(nxt, (*taken, path), seen | {nxt}, t_left, s_left)
            elif path.kind == TRAIN and t_left:
                walk(nxt, (*taken, path), seen | {nxt}, t_left - 1, s_left)
            elif path.kind == SHIP and s_left:
                walk(nxt, (*taken, path), seen | {nxt}, t_left, s_left - 1)

    walk(space, (), frozenset({space}), max(train, 0), max(ship, 0))
    return sorted(best.values(), key=lambda r: (r.moves, r.destination))


def steps_between(start: str, goal: str) -> int | None:
    """Fewest Travel actions from ``start`` to ``goal`` with no tickets, or None if unknown.

    One Travel action is one move, so this is also "how many rounds away is it" for an
    investigator who never buys a ticket - the pessimistic distance a plan can count on.
    """
    if start not in BY_SPACE or goal not in BY_SPACE:
        return None
    if start == goal:
        return 0
    seen, edge, steps = {start}, [start], 0
    while edge:
        steps += 1
        nxt: list[str] = []
        for here in edge:
            for space in adjacent(here):
                if space in seen:
                    continue
                if space == goal:
                    return steps
                seen.add(space)
                nxt.append(space)
        edge = nxt
    return None


def nearest(start: str, goals: Iterable[str]) -> tuple[str | None, int | None]:
    """The closest of ``goals`` to ``start`` on foot, and how many moves away it is."""
    best, best_steps = None, None
    for goal in goals:
        steps = steps_between(start, goal)
        if steps is None:
            continue
        if best_steps is None or steps < best_steps:
            best, best_steps = goal, steps
    return best, best_steps


def all_spaces() -> tuple[str, ...]:
    """Every space printed on the board, by the name src/vision/spaces.py gives it.

    The authority on whether a space exists. Anything that takes a space name from outside this
    code - what a person said, what a model wrote - checks it here first.
    """
    return tuple(BY_NAME)


def unreachable() -> tuple[str, ...]:
    """Spaces no path touches. A space missing from the table would silently disappear."""
    return tuple(name for name in BY_NAME if not BY_SPACE.get(name))


__all__ = [
    "VERIFIED",
    "all_spaces",
    "TRAIN",
    "SHIP",
    "UNCHARTED",
    "KINDS",
    "Path",
    "PATHS",
    "BY_SPACE",
    "Route",
    "neighbours",
    "adjacent",
    "kinds_touching",
    "ticket_kinds",
    "travel_options",
    "steps_between",
    "nearest",
    "unreachable",
]
