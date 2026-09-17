"""Candidate generation: only what the rules allow, and honest about what it cannot see."""

from __future__ import annotations

import pytest

from src.strategy import moves
from src.strategy.game import ROBOT, GameState


def _game(space: str = "Shanghai") -> GameState:
    game = GameState()
    game.set_ancient_one("Azathoth")
    game.add_investigator("Lily Chen", controller=ROBOT, space=space)
    game.reserve = ["Bull Whip", "Arcane Scholar"]
    return game


def _situation(game: GameState) -> moves.Situation:
    return moves.situation_of(game, game.robot_investigator)


def _keys(cands) -> set[str]:
    return {c.key for c in cands}


def test_the_action_table_is_the_six_printed_actions():
    assert len(moves.ACTIONS) == 6
    assert {a.key for a in moves.ACTIONS} == {
        moves.TRAVEL,
        moves.PREPARE,
        moves.ACQUIRE,
        moves.REST,
        moves.TRADE,
        moves.COMPONENT,
    }
    assert all(a.source for a in moves.ACTIONS)


def test_a_city_offers_travel_prepare_and_acquire():
    game = _game("Shanghai")
    found = candidates = moves.candidates(game, _situation(game))
    assert moves.TRAVEL in _keys(found)
    assert moves.PREPARE in _keys(found)
    assert moves.ACQUIRE in _keys(found)
    # Every Travel candidate goes to a space the board actually touches.
    destinations = {c.destination for c in candidates if c.key == moves.TRAVEL}
    assert destinations == set(moves.map_graph.adjacent("Shanghai"))


def test_a_sea_space_offers_neither_acquire_nor_prepare():
    game = _game("11")  # a Sea space
    found = moves.candidates(game, _situation(game))
    assert moves.ACQUIRE not in _keys(found)
    assert moves.PREPARE not in _keys(found)
    assert moves.TRAVEL in _keys(found)
    assert any("not a City space" in r for r in moves.refusals(game, _situation(game)))


def test_prepare_for_travel_offers_only_the_tickets_the_space_touches():
    game = _game("Istanbul")  # rail hub, no water
    tickets = {c.detail for c in moves.candidates(game, _situation(game)) if c.key == moves.PREPARE}
    assert tickets == {"Train ticket"}


def test_two_tickets_is_the_maximum():
    game = _game("Shanghai")
    state = game.robot_investigator
    state.train_tickets, state.ship_tickets = 1, 1
    found = moves.candidates(game, _situation(game))
    assert moves.PREPARE not in _keys(found)
    assert any("maximum" in r for r in moves.refusals(game, _situation(game)))


def test_a_ticket_extends_travel_along_its_own_kind_of_path():
    game = _game("Rome")
    state = game.robot_investigator
    state.train_tickets = 1
    destinations = {c.destination for c in moves.candidates(game, _situation(game)) if c.key == moves.TRAVEL}
    assert "17" in destinations  # Rome -> Istanbul -> 17, both Train paths
    assert "13" not in destinations  # London -> 13 is a Ship path and no Ship ticket is held


def test_a_named_monster_removes_rest_and_acquire():
    game = _game("Shanghai")
    game.robot_investigator.health = 3
    game.board.seed([_sighting("Shanghai", 1)])
    game.board.name(1, "monster:Cultist")
    found = moves.candidates(game, _situation(game))
    assert moves.REST not in _keys(found)
    assert moves.ACQUIRE not in _keys(found)
    assert any("Monster is at Shanghai" in r for r in moves.refusals(game, _situation(game)))


def test_an_unnamed_piece_is_a_question_not_a_monster():
    game = _game("Shanghai")
    game.robot_investigator.health = 3
    game.board.seed([_sighting("Shanghai", 1)])  # seen but never named
    found = {c.key: c for c in moves.candidates(game, _situation(game))}
    assert moves.REST in found  # not refused: nothing says it is a Monster
    assert any("nobody has named" in u for u in found[moves.REST].unknowns)
    assert any("nobody has named" in u for u in found[moves.ACQUIRE].unknowns)


def test_rest_is_not_offered_at_full_health_and_sanity():
    game = _game("Shanghai")
    found = moves.candidates(game, _situation(game))
    assert moves.REST not in _keys(found)
    assert any("full Health and Sanity" in r for r in moves.refusals(game, _situation(game)))


def test_acquire_admits_it_does_not_know_the_card_values():
    game = _game("Shanghai")
    acquire = next(c for c in moves.candidates(game, _situation(game)) if c.key == moves.ACQUIRE)
    assert any("the number printed on them" in u for u in acquire.unknowns)
    assert any("Bull Whip" in u for u in acquire.unknowns)


def test_trade_needs_somebody_else_on_the_space():
    game = _game("Shanghai")
    assert moves.TRADE not in _keys(moves.candidates(game, _situation(game)))
    game.add_investigator("Lola Hayes", controller="Alessandro", space="Shanghai")
    trade = [c for c in moves.candidates(game, _situation(game)) if c.key == moves.TRADE]
    assert [c.partner for c in trade] == ["Lola Hayes"]


def test_a_component_action_comes_from_the_printed_card_text():
    game = _game("Shanghai")
    game.robot_investigator.possessions = ["Flute of the Outer Gods", "Bull Whip"]
    components = {c.detail: c for c in moves.candidates(game, _situation(game)) if c.key == moves.COMPONENT}
    assert "Flute of the Outer Gods" in components  # its text starts with "Action:"
    assert components["Flute of the Outer Gods"].effect.startswith("Action:")
    assert "Bull Whip" not in components  # a passive bonus is not an action


def test_a_card_the_base_game_does_not_have_becomes_an_unknown():
    game = _game("Shanghai")
    game.robot_investigator.possessions = ["Nonexistent Relic"]
    component = next(c for c in moves.candidates(game, _situation(game)) if c.key == moves.COMPONENT)
    assert component.unknowns and "not in the base-game card list" in component.unknowns[0]


def test_a_delayed_investigator_does_nothing_and_says_why():
    game = _game("Shanghai")
    game.robot_investigator.conditions = ["Delayed"]
    assert moves.candidates(game, _situation(game)) == []
    assert any("Delayed" in r for r in moves.refusals(game, _situation(game)))


def test_a_turn_is_at_most_two_distinct_actions():
    game = _game("Shanghai")
    game.robot_investigator.health = 3
    for plan in moves.plans(game, _situation(game)):
        assert 1 <= len(plan.actions) <= moves.MAX_ACTIONS
        keys = [a.key for a in plan.actions]
        if len(keys) == 2 and keys[0] == keys[1]:
            assert keys[0] == moves.COMPONENT
            assert plan.actions[0].detail != plan.actions[1].detail


def test_the_second_action_is_judged_from_where_the_first_one_ends():
    game = _game("11")  # a Sea space: no Acquire here
    plans = moves.plans(game, _situation(game))
    travel_then_buy = [
        p
        for p in plans
        if len(p.actions) == 2 and p.actions[0].key == moves.TRAVEL and p.actions[1].key == moves.ACQUIRE
    ]
    assert travel_then_buy, "travelling to a city should make Acquire Assets possible"
    plan = travel_then_buy[0]
    assert plan.ends_at == plan.actions[0].destination
    assert plan.actions[1].detail == f"at {plan.ends_at}"


def test_a_plan_describes_itself_the_way_it_would_be_spoken():
    game = _game("Shanghai")
    plan = next(p for p in moves.plans(game, _situation(game)) if p.actions[0].destination == "Tokyo")
    text = plan.describe()
    assert text.startswith("Travel to Tokyo")
    assert "Shanghai to Tokyo by ship" in text


def test_plans_carry_the_unknowns_of_the_actions_in_them():
    game = _game("Shanghai")
    plan = next(p for p in moves.plans(game, _situation(game)) if p.actions[0].key == moves.ACQUIRE)
    assert plan.unknowns == plan.actions[0].unknowns


def test_component_action_text_takes_only_the_action_clause():
    effect = "Gain +1 Lore. | Action: Test Lore. | Reckoning: lose 1 Sanity."
    assert moves.component_action_text(effect) == "Action: Test Lore."
    assert moves.component_action_text("Gain +1 Strength during Combat Encounters.") == ""


def test_card_effects_come_from_the_versioned_base_game_file():
    assert "Strength" in moves.card_effect("Bull Whip")
    assert moves.card_effect("Elder Sign") == ""  # Arkham Horror, not this box


def _sighting(space: str, piece_id: int):
    """A minimal scan sighting, shaped like the ones src/vision/detect.py returns."""

    class _S:
        def __init__(self) -> None:
            self.space = space
            self.near = None
            self.x = 0.0
            self.y = 0.0
            self.kind = "piece"
            self.id = piece_id

    return _S()


@pytest.fixture(autouse=True)
def _no_cached_cards(monkeypatch):
    # The versioned card list is part of the repository; what the owner has dictated on this
    # machine is not, so it is stubbed out rather than read.
    moves._cards.cache_clear()
    monkeypatch.setattr(moves, "_dictated", lambda: {})
    yield
    moves._cards.cache_clear()


def test_a_value_read_from_the_box_stops_being_a_question(monkeypatch):
    game = _game("Shanghai")
    game.reserve = ["Bull Whip", "Arcane Scholar"]
    monkeypatch.setattr(moves, "_dictated", lambda: {"bull whip": {"name": "Bull Whip", "value": 2}})
    acquire = next(c for c in moves.candidates(game, _situation(game)) if c.key == moves.ACQUIRE)
    asked = " ".join(acquire.unknowns)
    assert "Arcane Scholar" in asked  # nobody has read its value
    assert "Bull Whip" not in asked  # this one has been read
    assert moves.card_value("Bull Whip") == 2


def test_the_planner_does_not_offer_an_action_already_spent_this_round():
    """The other half of the loop: once take_turn writes its actions into the game, replanning
    has to stop offering the action it just spent, or every call returns the same move."""
    from src.strategy.game import ROBOT, GameState
    from src.strategy.moves import plans, situation_of

    game = GameState()
    game.set_ancient_one("Azathoth")
    game.add_investigator("Lily Chen", controller=ROBOT)
    game.begin_round()
    who = game.by_name("Lily Chen")

    before = plans(game, situation_of(game, who))
    assert any(len(p.actions) == 2 for p in before)  # two actions while both are free

    game.record_action("Lily Chen", "rest")
    after = plans(game, situation_of(game, who))
    assert after, "one action is still left, so there are still plans"
    assert all(len(p.actions) == 1 for p in after)  # only one action left to spend
    assert all(a.key != "rest" for p in after for a in p.actions)

    game.record_action("Lily Chen", "travel")
    assert plans(game, situation_of(game, who)) == []  # nothing left this round
