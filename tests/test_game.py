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


def test_the_spoken_setup_wins_over_a_gallery_guess():
    """The live failure: the gallery called the Shanghai piece Akachi Onyele; it was Lily Chen."""
    board = BoardState()
    board.seed([Sighting("Shanghai", x=900, y=100, name="investigator:Akachi Onyele")])
    game = GameState(board)
    game.add_investigator("Lily Chen", controller=ROBOT)

    claimed = game.claim_pieces()
    assert claimed, "the guess blocked the setup from claiming the piece"
    piece = board.at("Shanghai")[0]
    assert piece.name == "investigator:Lily Chen"
    assert piece.guess == "investigator:Akachi Onyele", "the guess is kept, just not believed"
    assert game.by_name("Lily Chen").piece_id == piece.id


# --------------------------------------------------------------------------- the round


def _table():
    from src.strategy.game import ROBOT, GameState

    game = GameState()
    game.set_ancient_one("Azathoth")
    game.add_investigator("Lily Chen", controller=ROBOT)
    game.add_investigator("Jacqueline Fine", controller="Alessandro")
    return game


def test_a_game_being_set_up_has_not_started():
    game = _table()
    assert game.started is False and game.round == 0
    assert "has not started" in game.where_we_are()
    assert game.to_act() is None


def test_the_round_runs_action_encounter_mythos_and_opens_the_next():
    """GAME_REFERENCE.md: every round is Action Phase -> Encounter Phase -> Mythos Phase."""
    game = _table()
    assert game.advance_phase() == "action" and game.round == 1
    assert game.advance_phase() == "encounter"
    assert game.advance_phase() == "mythos"
    assert game.advance_phase() == "action" and game.round == 2


def test_two_distinct_actions_each_at_most_once():
    game = _table()
    game.begin_round()
    assert game.record_action("Lily Chen", "travel") == (True, "")
    assert game.record_action("Lily Chen", "travel")[0] is False  # the same action twice
    assert game.record_action("Lily Chen", "rest") == (True, "")
    allowed, why = game.record_action("Lily Chen", "acquire_assets")
    assert allowed is False and "already taken 2 actions" in why
    assert game.actions_left("Lily Chen") == 0


def test_a_component_is_limited_by_component_not_by_action():
    """The reference gives Component Action its own rule, "each component once per round",
    which would say nothing if the action itself could only be taken once."""
    game = _table()
    game.begin_round()
    assert game.record_action("Lily Chen", "component", component="Lucky Rabbit's Foot")[0] is True
    assert game.record_action("Lily Chen", "component", component="Protective Amulet")[0] is True
    game.begin_round()
    assert game.record_action("Lily Chen", "component", component="Protective Amulet")[0] is True
    assert game.record_action("Lily Chen", "component", component="Protective Amulet")[0] is False


def test_the_lead_investigator_acts_first_and_then_round_the_table():
    game = _table()
    game.lead = "Jacqueline Fine"
    game.begin_round()
    assert [i.name for i in game.order()] == ["Jacqueline Fine", "Lily Chen"]
    assert game.to_act().name == "Jacqueline Fine"
    game.record_action("Jacqueline Fine", "travel")
    game.record_action("Jacqueline Fine", "rest")
    assert game.to_act().name == "Lily Chen"


def test_no_action_outside_the_action_phase():
    game = _table()
    game.begin_round()
    game.advance_phase()
    allowed, why = game.record_action("Lily Chen", "travel")
    assert allowed is False and "Encounter Phase" in why
    assert game.record_encounter("Lily Chen") == (True, "")
    assert game.record_encounter("Lily Chen")[0] is False  # one encounter each


def test_a_new_round_forgets_what_everyone_did():
    game = _table()
    game.begin_round()
    game.record_action("Lily Chen", "travel")
    game.advance_phase()
    game.record_encounter("Lily Chen")
    game.advance_phase()
    game.advance_phase()  # past the Mythos Phase, into round 2
    assert game.round == 2 and game.actions_left("Lily Chen") == 2
    assert game.by_name("Lily Chen").encountered is False


def test_the_round_survives_the_file(tmp_path):
    from src.strategy.game import GameState

    game = _table()
    game.begin_round()
    game.record_action("Lily Chen", "travel")
    game.advance_phase()
    game.record_encounter("Lily Chen")
    path = game.save(tmp_path / "game_state.json")

    again = GameState.load(path)
    assert again.round == 1 and again.phase == "encounter"
    assert again.by_name("Lily Chen").actions == ["travel"]
    assert again.by_name("Lily Chen").encountered is True
    assert again.where_we_are() == game.where_we_are()


def test_two_fours_are_not_two_successes():
    """2026-09-17, live: the robot quoted "a 5 or 6 is a success" and then called two 4s "dois
    sucessos". A wrong success count changes the game in silence, and counting is arithmetic."""
    game = _table()
    assert game.test_result("Lily Chen", [4, 4])["successes"] == 0
    assert game.test_result("Lily Chen", [4, 4])["passed"] is False
    assert game.test_result("Lily Chen", [4, 5, 6])["successes"] == 2
    assert game.test_result("Lily Chen", [])["passed"] is False


def test_blessed_counts_fours_and_cursed_only_sixes():
    game = _table()
    who = game.by_name("Lily Chen")
    who.conditions.append("Blessed")
    assert game.test_result("Lily Chen", [4, 4])["successes"] == 2
    who.conditions = ["Cursed"]
    assert game.test_result("Lily Chen", [5, 5, 6])["successes"] == 1


def test_an_effect_moves_the_sheet_and_is_bounded():
    """The robot said "vou atualizar isso aqui" and the sheet did not move, because no tool
    could move it. Deltas, because the table says "perde 1 de Sanity", never "fica com 5"."""
    game = _table()
    before = game.by_name("Lily Chen").sanity
    out = game.apply_effect("Lily Chen", sanity=-1, clues=+1)
    assert out["applied"] and out["sanity"] == before - 1 and out["clues"] == 1

    game.apply_effect("Lily Chen", sanity=-99)
    assert game.by_name("Lily Chen").sanity == 0  # never below zero
    game.apply_effect("Lily Chen", sanity=+99)
    assert game.by_name("Lily Chen").sanity == game.by_name("Lily Chen").sheet.sanity  # nor above the sheet


def test_an_effect_carries_cards_and_conditions():
    game = _table()
    game.apply_effect("Lily Chen", gain=["Bull Whip"], conditions_gained=["Blessed"])
    who = game.by_name("Lily Chen")
    assert "Bull Whip" in who.possessions and "Blessed" in who.conditions
    game.apply_effect("Lily Chen", lose=["Bull Whip"], conditions_lost=["Blessed"])
    assert "Bull Whip" not in who.possessions and "Blessed" not in who.conditions
    assert game.apply_effect("Nobody At All", sanity=-1)["applied"] is False


def test_the_mythos_line_names_who_draws():
    """It read "the Lead Investigator draws the Mythos card" and the robot, which was the Lead,
    asked the table whose turn it was."""
    game = _table()
    game.lead = "Lily Chen"  # the robot's own
    game.begin_round()
    game.advance_phase()
    game.advance_phase()
    assert "I draw the Mythos card" in game.where_we_are()

    game.lead = "Jacqueline Fine"
    assert "Jacqueline Fine draws the Mythos card" in game.where_we_are()


def test_the_session_id_is_the_game_it_belongs_to_and_survives_the_file():
    """Rounds are filed under it, so it must not change when the game is read back."""
    import time as _time

    from src.strategy.game import GameState

    game = GameState()
    game.started_at = _time.mktime((2026, 9, 16, 13, 39, 0, 0, 0, -1))
    assert game.session_id == "20260916-1339"


def test_the_session_id_comes_back_with_the_game(tmp_path):
    from src.strategy.game import ROBOT, GameState

    game = GameState()
    game.set_ancient_one("Azathoth")
    game.add_investigator("Lily Chen", controller=ROBOT)
    was = game.session_id
    game.save(tmp_path / "game.json")
    assert GameState.load(tmp_path / "game.json").session_id == was


def test_a_round_summary_says_what_the_round_did_not_only_what_is_true():
    from src.strategy.game import ROBOT, GameState

    game = GameState()
    game.set_ancient_one("Azathoth")
    game.mystery = "The Deep One's Attack"
    mine = game.add_investigator("Lily Chen", controller=ROBOT, space="Shanghai")
    game.add_investigator("Jacqueline Fine", controller="Alessandro", space="San Francisco")
    game.begin_round()

    mine.space = "Tokyo"
    mine.health -= 2
    mine.clues += 1
    mine.possessions.append("Lucky Cigarette Case")
    game.record_action("Lily Chen", "travel")
    game.phase = "encounter"
    game.record_encounter("Lily Chen")
    game.note("O Gate em Roma foi fechado.")

    summary = game.round_summary()
    assert "Round 1 against Azathoth" in summary
    assert "moved from Shanghai to Tokyo" in summary
    assert "took Travel" in summary and "had an encounter at Tokyo" in summary
    assert "gained Lucky Cigarette Case" in summary
    assert "lost 2 Health" in summary and "gained 1 Clue;" in summary
    assert "The table said: O Gate em Roma foi fechado." in summary
    # The robot is named, never "I": this text is what gets embedded and searched later.
    assert "Lily Chen (the robot)" in summary and " I (" not in summary


def test_a_round_whose_opening_was_never_recorded_does_not_pretend_to_know_what_changed():
    """An older saved game has no round_start. Saying what is true now is honest; calling every
    note something the table said this round would not be."""
    from src.strategy.game import ROBOT, GameState

    game = GameState()
    game.set_ancient_one("Azathoth")
    game.add_investigator("Lily Chen", controller=ROBOT, space="Shanghai")
    game.round, game.phase = 1, "mythos"
    game.note("Em Roma tem um portal aberto.")

    summary = game.round_summary()
    assert "How this round opened was not recorded" in summary
    assert "The table said" not in summary
