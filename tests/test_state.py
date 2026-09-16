"""The board state: seeded by a scan, moved by motion, and keeping a name across every move."""

import json
from dataclasses import dataclass

from src.strategy.state import BoardState, TrackedPiece


@dataclass
class Sighting:
    """What a scan or a motion verdict hands over (duck-typed: vision.detect.Piece fits)."""

    space: str | None = None
    near: str | None = None
    x: float = 0.0
    y: float = 0.0
    kind: str = "piece"
    name: str | None = None
    value: int | None = None


@dataclass
class FakeMove:
    departed: list = None
    arrived: list = None

    def __post_init__(self):
        self.departed = self.departed or []
        self.arrived = self.arrived or []


def test_an_empty_state_says_so():
    state = BoardState()
    assert len(state) == 0 and state.describe() == "the board is empty"


def test_a_scan_seeds_the_state():
    state = BoardState()
    changes = state.seed([Sighting("London", x=10, y=10), Sighting("Rome", x=50, y=50)])
    assert len(state) == 2
    assert [c.kind for c in changes] == ["placed", "placed"]
    assert [p.space for p in state.on_board] == ["London", "Rome"]
    assert "London" in state.describe() and "Rome" in state.describe()
    assert state.at("London")[0].id == 1 and state.at("Tokyo") == []


def test_a_move_keeps_the_same_piece():
    state = BoardState()
    state.seed([Sighting("Rome", x=100, y=100)])
    piece_id = state.on_board[0].id

    changes = state.apply(
        FakeMove(departed=[Sighting("Rome", x=100, y=100)], arrived=[Sighting("Istanbul", x=160, y=110)])
    )
    assert len(changes) == 1 and changes[0].kind == "moved"
    assert changes[0].from_space == "Rome" and changes[0].to_space == "Istanbul"
    assert len(state) == 1, "a move must not invent a second piece"
    assert state.on_board[0].id == piece_id, "the piece kept its identity"
    assert state.at("Istanbul") and state.at("Rome") == []


def test_a_name_survives_every_later_move():
    """The whole point of tracking: naming is impossible per-frame, so it must be learnt once."""
    state = BoardState()
    state.seed([Sighting("Rome", x=100, y=100)])
    state.name_at("Rome", "investigator:Akachi Onyele")
    assert state.on_board[0].label == "Akachi Onyele"

    for a, b, ax, bx in (("Rome", "Istanbul", 100, 160), ("Istanbul", "Tunguska", 160, 230)):
        state.apply(FakeMove(departed=[Sighting(a, x=ax, y=100)], arrived=[Sighting(b, x=bx, y=100)]))
    piece = state.on_board[0]
    assert piece.space == "Tunguska"
    assert piece.name == "investigator:Akachi Onyele", "the name was lost across moves"
    assert state.by_name("Akachi Onyele") is piece
    assert "Akachi Onyele moved from Istanbul to Tunguska" == state.changes[-1].describe()


def test_an_arrival_with_no_departure_is_a_new_piece():
    state = BoardState()
    changes = state.apply(FakeMove(arrived=[Sighting("London", x=10, y=10)]))
    assert [c.kind for c in changes] == ["placed"]
    assert len(state) == 1 and state.at("London")
    assert "was placed on London" in changes[0].describe()


def test_a_departure_with_no_arrival_takes_the_piece_off_the_board():
    state = BoardState()
    state.seed([Sighting("London", x=10, y=10)])
    changes = state.apply(FakeMove(departed=[Sighting("London", x=10, y=10)]))
    assert [c.kind for c in changes] == ["removed"]
    assert len(state) == 0 and "taken off London" in changes[0].describe()


def test_two_pieces_moving_at_once_are_paired_by_distance():
    state = BoardState()
    state.seed([Sighting("London", x=0, y=0), Sighting("Tokyo", x=900, y=0)])
    london, tokyo = state.on_board[0].id, state.on_board[1].id

    # Both move a short way; the far pairing would swap their identities.
    changes = state.apply(
        FakeMove(
            departed=[Sighting("London", x=0, y=0), Sighting("Tokyo", x=900, y=0)],
            arrived=[Sighting("Shanghai", x=880, y=20), Sighting("Arkham", x=40, y=20)],
        )
    )
    assert len([c for c in changes if c.kind == "moved"]) == 2
    assert len(state) == 2
    assert next(p for p in state.on_board if p.id == london).space == "Arkham"
    assert next(p for p in state.on_board if p.id == tokyo).space == "Shanghai"


def test_a_departure_and_an_arrival_far_apart_are_not_the_same_piece():
    state = BoardState()
    state.seed([Sighting("London", x=0, y=0)])
    changes = state.apply(
        FakeMove(departed=[Sighting("London", x=0, y=0)], arrived=[Sighting("Sydney", x=5000, y=3000)])
    )
    kinds = sorted(c.kind for c in changes)
    assert kinds == ["placed", "removed"], f"got {kinds}"
    assert len(state) == 1 and state.at("Sydney")


def test_a_piece_between_spaces_is_described_by_the_nearest():
    state = BoardState()
    state.seed([Sighting(None, near="Rome", x=10, y=10)])
    piece = state.on_board[0]
    assert piece.where == "near Rome" and "near Rome" in state.describe()
    state.seed([Sighting(None, None, x=10, y=10)])
    assert state.on_board[0].where == "off the spaces"


def test_a_die_is_described_by_its_face():
    state = BoardState()
    state.seed([Sighting("Rome", x=10, y=10, kind="die", value=5)])
    assert state.on_board[0].label == "die showing 5"
    state.seed([Sighting("Rome", x=10, y=10, kind="die")])
    assert state.on_board[0].label == "a die"


def test_reseeding_carries_names_over_for_pieces_that_did_not_move():
    """A second scan must not throw away what the owner already told the robot."""
    state = BoardState()
    state.seed([Sighting("Rome", x=100, y=100)])
    state.name_at("Rome", "investigator:Akachi Onyele")
    state.seed([Sighting("Rome", x=102, y=101), Sighting("London", x=10, y=10)])
    assert state.at("Rome")[0].name == "investigator:Akachi Onyele"
    assert state.at("London")[0].name is None


def test_naming_a_space_with_two_pieces_is_refused():
    state = BoardState()
    state.seed([Sighting("Rome", x=100, y=100), Sighting("Rome", x=110, y=105)])
    assert state.name_at("Rome", "investigator:Akachi Onyele") is None
    assert state.name_at("Tokyo", "monster:Cultist") is None
    assert state.name(999, "x") is None


def test_the_state_round_trips_through_a_file(tmp_path):
    state = BoardState()
    state.seed([Sighting("Rome", x=100, y=100), Sighting("London", x=10, y=10)])
    state.name_at("Rome", "investigator:Akachi Onyele")
    state.apply(
        FakeMove(departed=[Sighting("Rome", x=100, y=100)], arrived=[Sighting("Istanbul", x=160, y=110)])
    )

    path = state.save(tmp_path / "state.json")
    assert json.loads(path.read_text(encoding="utf-8"))["text"] == state.describe()

    back = BoardState.load(path)
    assert len(back) == len(state)
    assert back.by_name("Akachi Onyele") is not None
    assert back.by_name("Akachi Onyele").space == "Istanbul"
    # ids keep counting from where they stopped, so a later piece cannot reuse one
    before = back._next_id
    back.apply(FakeMove(arrived=[Sighting("Tokyo", x=900, y=0)]))
    assert back.at("Tokyo")[0].id == before


def test_tracked_piece_history_records_where_it_has_been():
    state = BoardState()
    state.seed([Sighting("Rome", x=100, y=100)])
    state.apply(
        FakeMove(departed=[Sighting("Rome", x=100, y=100)], arrived=[Sighting("Istanbul", x=160, y=110)])
    )
    spaces = [h["space"] for h in state.on_board[0].history]
    assert spaces == ["Rome", "Istanbul"]


def test_restore_ignores_fields_it_does_not_know():
    piece = TrackedPiece.restore({"id": 3, "space": "Rome", "something_new": 1})
    assert piece.id == 3 and piece.space == "Rome"


def test_a_gallery_match_is_a_guess_and_never_a_name():
    """Measured at 2 of 13: a confident wrong label is worse than no label.

    It also has to stay out of the way. If the guess were stored as the name, the spoken setup
    could no longer claim the piece, and "Lily Chen is in Shanghai" would be silently ignored
    because the gallery had already called that piece Akachi Onyele.
    """
    state = BoardState()
    state.seed([Sighting("Shanghai", x=900, y=100, name="investigator:Akachi Onyele")])
    piece = state.on_board[0]
    assert piece.name is None, "a gallery match must not become the name"
    assert piece.guess == "investigator:Akachi Onyele"
    assert "maybe Akachi Onyele" in piece.label and f"#{piece.id}" in piece.label
    assert state.by_name("Akachi Onyele") is None, "a guess must not answer a name lookup"

    state.name(piece.id, "investigator:Lily Chen")
    assert piece.name == "investigator:Lily Chen"
    assert piece.label == "Lily Chen", "a real name replaces the guess in what is spoken"
