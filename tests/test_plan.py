"""The plan for the whole game: written once, revised when the game argues with it."""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.llm.ollama_client import LLMError
from src.strategy import decide, moves
from src.strategy import plan as planner
from src.strategy.game import ROBOT, GameState


@dataclass
class _Reply:
    text: str
    model: str = "test"
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0


def _answers(payload: dict):
    calls: list[list[dict]] = []

    def chat_fn(messages, **kwargs):
        calls.append(messages)
        return _Reply(json.dumps(payload))

    chat_fn.calls = calls  # type: ignore[attr-defined]
    return chat_fn


PLAN = {
    "aim": "Vamos resolver tres Mysteries antes do Doom chegar a zero.",
    "my_role": "Lily Chen fecha Gates na Asia com Strength 4.",
    "priorities": ["Fechar o Gate de Rome", "Juntar Clues", "Nao deixar ninguem cair"],
    "watch_for": ["Um Monster Epic aparecer", "Doom abaixo de 5"],
}


def _game() -> GameState:
    game = GameState()
    game.set_ancient_one("Azathoth")
    game.add_investigator("Lily Chen", controller=ROBOT)
    game.add_investigator("Jacqueline Fine", controller="Alessandro")
    game.round = 1
    return game


def test_the_plan_is_written_from_the_setup():
    game = _game()
    chat_fn = _answers(PLAN)
    made = planner.make_plan(game, language="pt-BR", chat_fn=chat_fn)
    assert made.aim.startswith("Vamos resolver")
    assert len(made.priorities) == 3
    assert made.round_written == 1
    sent = chat_fn.calls[0][1]["content"]
    assert "Lily Chen" in sent and "Azathoth" in sent and "Strength 4" in sent


def test_the_plan_is_spoken_without_glueing_english_words_in_front():
    made = planner.GamePlan(**{**PLAN, "priorities": tuple(PLAN["priorities"]), "watch_for": ()})
    spoken = made.speak()
    assert spoken.startswith("Vamos resolver")
    assert "First," not in spoken and "Primeiro" not in spoken


def test_more_than_four_priorities_are_cut():
    game = _game()
    payload = {**PLAN, "priorities": [f"p{i}" for i in range(9)], "watch_for": [f"w{i}" for i in range(7)]}
    made = planner.make_plan(game, chat_fn=_answers(payload))
    assert len(made.priorities) == planner.MAX_PRIORITIES
    assert len(made.watch_for) == planner.MAX_WATCH


def test_without_the_model_the_robot_plays_without_a_plan():
    def chat_fn(messages, **kwargs):
        raise LLMError("ollama is down")

    made = planner.make_plan(_game(), chat_fn=chat_fn)
    assert made.empty
    assert made.as_text() == ""


def test_a_round_that_changed_nothing_costs_no_model_call():
    game = _game()
    made = planner.make_plan(game, chat_fn=_answers(PLAN))
    chat_fn = _answers(PLAN)
    again = planner.revise(made, game, [], chat_fn=chat_fn)
    assert again is made
    assert chat_fn.calls == []


def test_a_revision_keeps_what_it_dropped_in_the_history():
    game = _game()
    made = planner.make_plan(game, chat_fn=_answers(PLAN))
    game.round = 3
    revised = planner.revise(
        made,
        game,
        ["doom moved from 15 to 9"],
        chat_fn=_answers({**PLAN, "changed": "deixei de juntar Clues, o Doom esta rapido demais"}),
    )
    assert revised.history == ("round 3: deixei de juntar Clues, o Doom esta rapido demais",)
    assert revised.round_written == 3


def test_a_failed_revision_leaves_the_plan_in_force():
    game = _game()
    made = planner.make_plan(game, chat_fn=_answers(PLAN))

    def chat_fn(messages, **kwargs):
        raise LLMError("ollama is down")

    assert planner.revise(made, game, ["doom fell"], chat_fn=chat_fn) is made


def test_changes_since_reports_what_a_round_did():
    game = _game()
    before = game.record()
    game.doom = 9
    game.investigators[0].health = 2
    game.investigators[0].space = "Tokyo"
    changes = planner.changes_since(planner.GamePlan(aim="x"), game, before)
    assert any("doom moved from 15 to 9" in c for c in changes)
    assert any("Lily Chen moved from Shanghai to Tokyo" in c for c in changes)
    assert any("down to 2 Health" in c for c in changes)


def test_a_plan_survives_being_saved_and_read_back():
    made = planner.GamePlan(
        **{**PLAN, "priorities": tuple(PLAN["priorities"]), "watch_for": tuple(PLAN["watch_for"])}
    )
    back = planner.GamePlan.restore(json.loads(json.dumps(made.record())))
    assert back == made


def test_the_turn_decision_carries_the_plan_and_calls_it_an_intention():
    game = _game()
    made = planner.make_plan(game, chat_fn=_answers(PLAN))
    situation = moves.situation_of(game, game.robot_investigator)
    text = decide.brief(game, situation, plan=made)
    assert "my own intention" in text
    assert "not a rule" in text
    assert "Fechar o Gate de Rome" in text


def test_an_empty_plan_adds_nothing_to_the_brief():
    game = _game()
    situation = moves.situation_of(game, game.robot_investigator)
    assert "intention" not in decide.brief(game, situation, plan=planner.GamePlan())
