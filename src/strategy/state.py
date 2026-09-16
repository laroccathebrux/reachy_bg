"""Where every piece is: the board state the vision keeps, seeded by a scan and moved by motion.

    state = BoardState()
    state.seed(pieces)                    # from a scan: what is on the board right now
    changes = state.apply(move)           # from motion: a piece left X and arrived at Y
    state.at("London")                    # -> the pieces standing there
    print(state.describe())               # "3 pieces: #1 at London, #2 at Rome, #3 (die, 5) at Tokyo"
    state.save(path)

The scan answers "what is on the board" and costs 14 seconds of the robot turning; motion
answers "what just changed" at 9 frames a second. Neither keeps a memory, so this does: a scan
seeds it, every move updates it, and it is the only thing that can answer "where is the piece
that was in Rome" a minute later.

A gallery match is kept as ``guess``, never as ``name``: it was measured at 2 of 13, so a
confident wrong label would both mislead and, worse, stop the spoken setup from claiming the
piece later. ``name`` is set only by something that knows - the setup, or a person saying it.

**Identity comes from continuity, not from recognition.** Naming a piece from its picture was
measured and does not work at this resolution: the ResNet gallery got 2 of 13 labelled crops
right and a Qwen3-VL 8B got 0 of 13, because a piece is about 250 px of a 1920-wide frame and
out of focus. But a piece that leaves Rome and arrives at Istanbul is *the same piece*, so it
carries whatever it was called. A name only has to be learnt once - by voice, by a good crop,
or by the owner typing it - and tracking keeps it from then on.

The pairing of departures to arrivals is by distance on the map: with one of each it is the
obvious pair, and with several the closest pairs are taken first. What is left over is a piece
placed on the board (an arrival with no departure) or taken off it (the reverse).

This covers only what the camera can see. The rest of the game state - doom, the Mystery in
play, cards in hand - never appears on the board as a moved piece and comes from the
conversation instead.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.logger import get_logger

log = get_logger(__name__)

MAX_PAIR_DISTANCE = 600.0  # rectified-map pixels: beyond this a departure and an arrival are
# not the same piece crossing the board but two unrelated events in one disturbance.


@dataclass
class TrackedPiece:
    """One piece, followed across moves. ``name`` survives every move once something sets it."""

    id: int
    space: str | None = None  # the board space it stands on, None when it is between spaces
    near: str | None = None  # the nearest space, when it is on none
    kind: str = "piece"  # piece | die
    name: str | None = None  # "investigator:Akachi Onyele", set only by something reliable
    guess: str | None = None  # what the image gallery thought it was, which is usually wrong
    guess_score: float = 0.0
    value: int | None = None  # what a die shows
    x: float = 0.0  # last known position on the rectified map
    y: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    history: list[dict[str, Any]] = field(default_factory=list)  # {"at", "space"} per sighting

    @property
    def where(self) -> str:
        if self.space:
            return self.space
        return f"near {self.near}" if self.near else "off the spaces"

    @property
    def label(self) -> str:
        """What to call it out loud: its name if it has one, otherwise what it is.

        A gallery guess is never spoken as a name. It was measured at 2 correct out of 13, so
        saying "Akachi Onyele" because a 250 px blur resembled one would be worse than saying
        nothing: the state would look certain while being wrong, and a real name from the setup
        could no longer claim the piece.
        """
        if self.name:
            kind, _, rest = self.name.partition(":")
            return rest or kind
        if self.kind == "die":
            return f"die showing {self.value}" if self.value is not None else "a die"
        if self.guess:
            kind, _, rest = self.guess.partition(":")
            return f"piece #{self.id} (maybe {rest or kind})"
        return f"piece #{self.id}"

    def place(self, space: str | None, near: str | None, x: float, y: float) -> None:
        self.space, self.near, self.x, self.y = space, near, x, y
        self.last_seen = time.time()
        self.history.append({"at": round(self.last_seen, 3), "space": self.where})

    def record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "space": self.space,
            "near": self.near,
            "kind": self.kind,
            "name": self.name,
            "guess": self.guess,
            "guess_score": round(self.guess_score, 3),
            "value": self.value,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "history": self.history[-20:],
        }

    @classmethod
    def restore(cls, data: dict[str, Any]) -> TrackedPiece:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Change:
    """One thing that happened to one piece, ready to be spoken or logged."""

    kind: str  # placed | moved | removed | named
    piece: TrackedPiece
    from_space: str | None = None
    to_space: str | None = None
    at: float = field(default_factory=time.time)

    def describe(self) -> str:
        who = self.piece.label
        if self.kind == "moved":
            return f"{who} moved from {self.from_space} to {self.to_space}"
        if self.kind == "placed":
            return f"{who} was placed on {self.to_space}"
        if self.kind == "removed":
            return f"{who} was taken off {self.from_space}"
        if self.kind == "named":
            return f"the piece at {self.to_space} is {who}"
        return f"{who}: {self.kind}"

    def record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "piece_id": self.piece.id,
            "name": self.piece.name,
            "from": self.from_space,
            "to": self.to_space,
            "at": round(self.at, 3),
            "text": self.describe(),
        }


def _distance(a: Any, b: Any) -> float:
    return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)


class BoardState:
    """Every piece the camera has seen, and where it is now."""

    def __init__(self) -> None:
        self.pieces: dict[int, TrackedPiece] = {}
        self.changes: list[Change] = []
        self._next_id = 1
        self.seeded_at = 0.0

    # ------------------------------------------------------------------ contents
    def __len__(self) -> int:
        return len(self.pieces)

    @property
    def on_board(self) -> list[TrackedPiece]:
        return sorted(self.pieces.values(), key=lambda p: p.id)

    def at(self, space: str) -> list[TrackedPiece]:
        return [p for p in self.on_board if p.space == space]

    def by_name(self, name: str) -> TrackedPiece | None:
        wanted = name.strip().lower()
        for piece in self.on_board:
            if piece.name and (piece.name.lower() == wanted or piece.name.lower().endswith(f":{wanted}")):
                return piece
        return None

    def describe(self) -> str:
        if not self.pieces:
            return "the board is empty"
        parts = [f"{p.label} at {p.where}" for p in self.on_board]
        return f"{len(parts)} piece(s): " + ", ".join(parts)

    # ------------------------------------------------------------------ updates
    def _add(self, sighting: Any) -> TrackedPiece:
        # What a sighting carries as "name" came from the image gallery, which is a guess.
        piece = TrackedPiece(
            id=self._next_id,
            kind=getattr(sighting, "kind", "piece") or "piece",
            guess=getattr(sighting, "name", None),
            guess_score=float(getattr(sighting, "name_score", 0.0) or 0.0),
            value=getattr(sighting, "value", None),
        )
        self._next_id += 1
        piece.place(
            getattr(sighting, "space", None),
            getattr(sighting, "near", None),
            float(getattr(sighting, "x", 0.0)),
            float(getattr(sighting, "y", 0.0)),
        )
        self.pieces[piece.id] = piece
        return piece

    def seed(self, sightings: list[Any]) -> list[Change]:
        """Replace the state with what a scan just saw; names already known are carried over."""
        remembered = {p.space: p for p in self.on_board if p.space and p.name}
        self.pieces.clear()
        self._next_id = 1
        changes = []
        for sighting in sightings:
            piece = self._add(sighting)
            known = remembered.get(piece.space)
            if known is not None and piece.name is None:
                piece.name = known.name  # a scan does not name pieces; do not lose what did
            changes.append(Change("placed", piece, to_space=piece.where))
        self.seeded_at = time.time()
        self.changes.extend(changes)
        log.info("state: seeded with %d piece(s) from a scan", len(self.pieces))
        return changes

    def apply(self, move: Any) -> list[Change]:
        """Update the state from one ``motion.Move``; returns what changed, in words."""
        departed = list(getattr(move, "departed", []))
        arrived = list(getattr(move, "arrived", []))
        changes: list[Change] = []

        # Pair departures with arrivals, closest first: with one of each this is the obvious
        # pair, and with several it keeps a piece's identity across the shortest journey.
        pairs: list[tuple[float, Any, Any]] = sorted(
            ((_distance(d, a), d, a) for d in departed for a in arrived), key=lambda t: t[0]
        )
        used_d: list[int] = []
        used_a: list[int] = []
        for distance, d, a in pairs:
            if id(d) in used_d or id(a) in used_a or distance > MAX_PAIR_DISTANCE:
                continue
            used_d.append(id(d))
            used_a.append(id(a))
            piece = self._piece_for(d)
            was = piece.where if piece else self._where_of(d)
            if piece is None:
                piece = self._add(d)  # it left a place we had not recorded; start following it
            piece.place(
                getattr(a, "space", None),
                getattr(a, "near", None),
                float(getattr(a, "x", 0.0)),
                float(getattr(a, "y", 0.0)),
            )
            if getattr(a, "value", None) is not None:
                piece.value = a.value
            changes.append(Change("moved", piece, from_space=was, to_space=piece.where))

        for a in arrived:
            if id(a) in used_a:
                continue
            piece = self._add(a)
            changes.append(Change("placed", piece, to_space=piece.where))

        for d in departed:
            if id(d) in used_d:
                continue
            piece = self._piece_for(d)
            if piece is None:
                continue
            self.pieces.pop(piece.id, None)
            changes.append(Change("removed", piece, from_space=piece.where))

        self.changes.extend(changes)
        for change in changes:
            log.info("state: %s", change.describe())
        return changes

    def name(self, piece_id: int, name: str) -> Change | None:
        """Give a tracked piece a name; it keeps it across every later move."""
        piece = self.pieces.get(piece_id)
        if piece is None:
            return None
        piece.name = name.strip() or None
        change = Change("named", piece, to_space=piece.where)
        self.changes.append(change)
        log.info("state: %s", change.describe())
        return change

    def name_at(self, space: str, name: str) -> Change | None:
        """Name the piece standing on a space, which is how a person refers to it out loud."""
        here = self.at(space)
        if len(here) != 1:
            return None
        return self.name(here[0].id, name)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _where_of(sighting: Any) -> str:
        space = getattr(sighting, "space", None)
        near = getattr(sighting, "near", None)
        return space or (f"near {near}" if near else "off the spaces")

    def _piece_for(self, sighting: Any) -> TrackedPiece | None:
        """The tracked piece a departure refers to: same space when possible, else the nearest."""
        space = getattr(sighting, "space", None)
        if space:
            here = self.at(space)
            if here:
                return min(here, key=lambda p: _distance(p, sighting))
        if not self.pieces:
            return None
        nearest = min(self.on_board, key=lambda p: _distance(p, sighting))
        return nearest if _distance(nearest, sighting) <= MAX_PAIR_DISTANCE else None

    # ------------------------------------------------------------------ storage
    def record(self) -> dict[str, Any]:
        return {
            "seeded_at": round(self.seeded_at, 3),
            "next_id": self._next_id,
            "pieces": [p.record() for p in self.on_board],
            "changes": [c.record() for c in self.changes[-50:]],
            "text": self.describe(),
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.record(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> BoardState:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        state = cls()
        for item in data.get("pieces", []):
            piece = TrackedPiece.restore(item)
            state.pieces[piece.id] = piece
        state._next_id = int(data.get("next_id", max(state.pieces, default=0) + 1))
        state.seeded_at = float(data.get("seeded_at", 0.0))
        return state


__all__ = ["BoardState", "Change", "TrackedPiece"]
