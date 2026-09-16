"""The game state: the spoken setup, and the moment it meets what the camera saw."""

from dataclasses import dataclass

import pytest

from src.strategy.game import ROBOT, GameState, InvestigatorState
from src.strategy.reference import (
    ANCIENT_ONES,
    INVESTIGATORS,
    ancient_one,
    closest_investigators,
    investigator,
)
from src.strategy.state import BoardState


@dataclass
class Sighting:
    space: str | None = None
    near: str | None = None
    x: float = 0.0
    y: float = 0.0
    kind: str = "piece"
    name: str | None = None
    value: int | None = None


# ---------------------------------------------------------------- the reference tables


def test_the_base_game_has_twelve_investigators_and_four_ancient_ones():
    assert len(INVESTIGATORS) == 12
    assert len(ANCIENT_ONES) == 4
    assert {a.name for a in ANCIENT_ONES} == {"Azathoth", "Cthulhu", "Shub-Niggurath", "Yog-Sothoth"}


def test_every_investigator_has_twelve_health_plus_sanity():
    """The rule the reference document states for every base investigator."""
    for sheet in INVESTIGATORS:
        assert sheet.health + sheet.sanity == 12, sheet.name


def test_every_starting_space_exists_on_the_board():
    """Setup claims pieces by starting space, so a typo here would silently never match."""
    from src.vision.spaces import BY_NAME

    for sheet in INVESTIGATORS:
        assert sheet.starting_space in BY_NAME, f"{sheet.name} starts at unknown {sheet.starting_space!r}"


def test_names_are_found_the_way_people_say_them():
    assert investigator("Lily Chen").name == "Lily Chen"
    assert investigator("lily").name == "Lily Chen"
    assert investigator("CHEN").name == "Lily Chen"
    assert investigator("  akachi  ").name == "Akachi Onyele"
    assert ancient_one("yog-sothoth").name == "Yog-Sothoth"
    assert ancient_one("Yog Sothoth").name == "Yog-Sothoth"
    assert ancient_one("shub").name == "Shub-Niggurath"


def test_expansion_content_is_not_recognised():
    """Hard rule: base game only. An unknown name must come back as unknown, not as a guess."""
    assert investigator("Roland Banks") is None
    assert investigator("Joe Diamond") is None
    assert ancient_one("Nyarlathotep") is None


def test_a_mangled_name_gets_suggestions():
    assert "Lily Chen" in closest_investigators("lily chan")
    assert "Akachi Onyele" in closest_investigators("akashi onielle")


# ---------------------------------------------------------------- setup


def test_setting_the_ancient_one_fills_in_the_doom_track():
    game = GameState()
    assert game.set_ancient_one("Azathoth").starting_doom == 15
    assert game.doom == 15
    assert "Azathoth" in game.briefing() and "15" in game.briefing()


def test_an_unrecognised_ancient_one_is_recorded_not_invented():
    game = GameState()
    assert game.set_ancient_one("Nyarlathotep") is None
    assert game.ancient_one is None
    assert any("Nyarlathotep" in u for u in game.unresolved)
    assert any("did not recognise" in m for m in game.missing())


def test_an_investigator_starts_from_its_sheet():
    game = GameState()
    lily = game.add_investigator("Lily Chen", controller=ROBOT)
    assert lily.space == "Shanghai"
    assert lily.health == 6 and lily.sanity == 6
    assert "Protective Amulet" in lily.possessions
    assert lily.is_robot and game.robot_investigator is lily


def test_an_investigator_with_starting_clues_gets_them():
    game = GameState()
    akachi = game.add_investigator("Akachi Onyele", controller="Alessandro")
    assert akachi.clues == 1 and akachi.space == "15"
    assert not akachi.is_robot


def test_adding_the_same_investigator_twice_does_not_duplicate_it():
    game = GameState()
    first = game.add_investigator("Lily Chen", controller=ROBOT)
    again = game.add_investigator("lily", controller="someone else")
    assert again is first and game.players == 1


def test_the_briefing_reads_back_what_it_understood():
    game = GameState()
    game.set_ancient_one("Cthulhu")
    game.add_investigator("Lily Chen", controller=ROBOT)
    game.add_investigator("Leo Anderson", controller="Alessandro")
    game.reserve = ["Bank Loan"]
    spoken = game.briefing()
    assert "Cthulhu" in spoken
    assert "I am playing Lily Chen" in spoken and "Shanghai" in spoken
    assert "Leo Anderson with Alessandro" in spoken
    assert "Bank Loan" in spoken


def test_missing_lists_what_still_has_to_be_asked():
    game = GameState()
    assert any("Ancient One" in m for m in game.missing())
    game.set_ancient_one("Azathoth")
    assert any("investigators" in m for m in game.missing())
    game.add_investigator("Leo Anderson", controller="Alessandro")
    assert any("which investigator I am playing" in m for m in game.missing())
    assert not game.ready
    game.add_investigator("Lily Chen", controller=ROBOT)
    assert game.ready


# ---------------------------------------------------------------- board meets briefing


def test_the_spoken_setup_names_the_pieces_the_camera_found():
    """The point of the whole design: no recognition, just what the person said plus position."""
    board = BoardState()
    board.seed([Sighting("Shanghai", x=900, y=100), Sighting("Buenos Aires", x=100, y=500)])
    game = GameState(board)
    game.set_ancient_one("Azathoth")
    game.add_investigator("Lily Chen", controller=ROBOT)
    game.add_investigator("Leo Anderson", controller="Alessandro")

    claimed = game.claim_pieces()
    assert len(claimed) == 2, claimed
    assert board.at("Shanghai")[0].name == "investigator:Lily Chen"
    assert board.at("Buenos Aires")[0].name == "investigator:Leo Anderson"
    assert game.by_name("Lily Chen").piece_id == board.at("Shanghai")[0].id
    assert not any("which piece" in m for m in game.missing())


def test_a_piece_is_not_claimed_when_two_sit_on_the_same_space():
    board = BoardState()
    board.seed([Sighting("Shanghai", x=900, y=100), Sighting("Shanghai", x=910, y=105)])
    game = GameState(board)
    game.add_investigator("Lily Chen", controller=ROBOT)
    assert game.claim_pieces() == [], "an ambiguous space must be asked about, not guessed"
    assert any("which piece" in m for m in game.missing())


def test_a_claimed_name_follows_the_piece_when_it_moves():
    board = BoardState()
    board.seed([Sighting("Shanghai", x=900, y=100)])
    game = GameState(board)
    game.add_investigator("Lily Chen", controller=ROBOT)
    game.claim_pieces()

    @dataclass
    class FakeMove:
        departed: list
        arrived: list

    board.apply(FakeMove([Sighting("Shanghai", x=900, y=100)], [Sighting("Tokyo", x=960, y=90)]))
    moved = game.sync_from_board()
    assert moved and "Lily Chen is now at Tokyo" in moved[0]
    assert game.by_name("Lily Chen").space == "Tokyo"
    assert board.at("Tokyo")[0].name == "investigator:Lily Chen"


def test_claiming_is_idempotent():
    board = BoardState()
    board.seed([Sighting("Shanghai", x=900, y=100)])
    game = GameState(board)
    game.add_investigator("Lily Chen", controller=ROBOT)
    assert len(game.claim_pieces()) == 1
    assert game.claim_pieces() == [], "a second call must not re-claim what it already has"


# ---------------------------------------------------------------- storage


def test_the_game_round_trips_through_a_file(tmp_path):
    board = BoardState()
    board.seed([Sighting("Shanghai", x=900, y=100)])
    game = GameState(board)
    game.set_ancient_one("Yog-Sothoth")
    game.add_investigator("Lily Chen", controller=ROBOT)
    game.add_investigator("Trish Scarborough", controller="Alessandro")
    game.claim_pieces()
    game.mystery = "Find the Key"
    game.doom = 11
    game.round = 3

    back = GameState.load(game.save(tmp_path / "game.json"))
    assert back.ancient_one.name == "Yog-Sothoth"
    assert back.doom == 11 and back.round == 3 and back.mystery == "Find the Key"
    assert back.robot_investigator.name == "Lily Chen"
    assert back.by_name("Trish Scarborough").controller == "Alessandro"
    assert back.by_name("Lily Chen").piece_id == game.by_name("Lily Chen").piece_id


def test_a_defeated_investigator_is_flagged():
    state = InvestigatorState(sheet=investigator("Lily Chen"), controller=ROBOT)
    assert not state.defeated
    state.health = 0
    assert state.defeated


@pytest.mark.parametrize("sheet", INVESTIGATORS, ids=lambda s: s.name)
def test_every_sheet_describes_itself(sheet):
    text = sheet.describe()
    assert sheet.name in text and sheet.starting_space in text
