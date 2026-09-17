"""What the thinking between rounds is allowed to hand the score, and what the code refuses."""

import pytest

from src.strategy import intent as intent_module
from src.strategy.game import ROBOT, GameState
from src.strategy.intent import Intent, bonus, reach_spaces, verify


@pytest.fixture
def game():
    state = GameState()
    state.set_ancient_one("Azathoth")
    state.add_investigator("Lily Chen", controller=ROBOT, space="Shanghai")
    return state


def test_an_invented_space_never_becomes_a_move(game):
    """The measured failure is a real fact welded to a link that does not hold. A link that
    names a place the board does not have is thrown out before it can score anything."""
    kept, refused = verify(
        [
            {"kind": "reach", "target": "Innsmouth", "why": "there is a Gate there"},
            {"kind": "reach", "target": "Tokyo", "why": "the Mystery wants somebody in the east"},
        ],
        game,
    )
    assert [i.target for i in kept] == ["Tokyo"]
    assert any("not a space on this board" in r for r in refused)


def test_an_action_that_is_not_printed_on_the_reference_card_is_refused(game):
    kept, refused = verify(
        [
            {"kind": "prefer", "target": "Cast Spell", "why": "the Amulet helps"},
            {"kind": "prefer", "target": "Acquire Assets", "why": "the Reserve is full"},
        ],
        game,
    )
    assert [i.target for i in kept] == ["acquire_assets"]
    assert any("not an action on the reference card" in r for r in refused)


def test_a_priority_with_no_reason_cannot_be_argued_with_so_it_is_refused(game):
    kept, refused = verify([{"kind": "reach", "target": "Tokyo", "why": "   "}], game)
    assert kept == ()
    assert any("no reason given" in r for r in refused)


def test_the_names_people_and_models_actually_write_are_understood(game):
    kept, _ = verify(
        [
            {"kind": "reach", "target": "space 5", "why": "a Clue is there"},
            {"kind": "prefer", "target": "prepare_for_travel", "why": "the goal is far"},
        ],
        game,
    )
    assert [i.target for i in kept] == ["5", "prepare_for_travel"]


def test_avoiding_the_space_it_is_already_on_is_not_a_priority(game):
    """It would take the same amount off every turn that stays, which is not a preference."""
    kept, refused = verify([{"kind": "avoid", "target": "Shanghai", "why": "monsters"}], game)
    assert kept == ()
    assert any("standing there now" in r for r in refused)


def test_a_round_cannot_have_more_priorities_than_it_has_priorities(game):
    proposed = [
        {"kind": "prefer", "target": t, "why": "because"}
        for t in ("Rest", "Trade", "Acquire Assets", "Prepare for Travel", "Travel")
    ]
    kept, refused = verify(proposed, game)
    assert len(kept) == intent_module.MAX_INTENTS
    assert any("more than" in r for r in refused)


def test_the_weight_is_clamped_so_one_priority_cannot_run_the_turn(game):
    kept, _ = verify([{"kind": "prefer", "target": "Rest", "why": "hurt", "weight": 99}], game)
    assert kept[0].weight == intent_module.MAX_WEIGHT


def test_reach_is_handed_to_the_distance_term_not_scored_here():
    """One idea, one number: walking towards a chosen space is scored like walking towards a Gate."""
    intents = (Intent("reach", "Tokyo", "the Mystery"), Intent("prefer", "rest", "hurt"))
    assert reach_spaces(intents) == ["Tokyo"]

    plan = type("P", (), {"actions": [type("A", (), {"key": "rest"})()]})()
    value, said = bonus(intents, plan, "Shanghai", 1.0)
    assert value == 1.0 and any("meant to Rest" in s for s in said)


def test_avoiding_a_space_only_costs_the_turns_that_end_there():
    intents = (Intent("avoid", "Rome", "a Serpent People is on it", 2.0),)
    plan = type("P", (), {"actions": []})()
    assert bonus(intents, plan, "Rome", 1.0)[0] == -2.0
    assert bonus(intents, plan, "Tokyo", 1.0)[0] == 0.0
