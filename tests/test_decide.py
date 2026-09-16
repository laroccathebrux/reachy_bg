"""Choosing a turn: the score prunes, the model picks, and the robot always plays."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.llm.ollama_client import LLMError
from src.strategy import decide, map_graph, moves
from src.strategy.game import ROBOT, GameState


@dataclass
class _Reply:
    text: str
    model: str = "test"
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0


def _answers(text: str):
    """A stand-in for the model that always answers the same thing, and counts the calls."""
    calls: list[list[dict]] = []

    def chat_fn(messages, **kwargs):
        calls.append(messages)
        return _Reply(text)

    chat_fn.calls = calls  # type: ignore[attr-defined]
    return chat_fn


def _game(space: str = "Shanghai") -> GameState:
    game = GameState()
    game.set_ancient_one("Azathoth")
    game.add_investigator("Lily Chen", controller=ROBOT, space=space)
    game.mystery = "Find the way in"
    game.reserve = ["Bull Whip", "Arcane Scholar"]
    return game


def _situation(game: GameState) -> moves.Situation:
    return moves.situation_of(game, game.robot_investigator)


class _Sighting:
    def __init__(self, space: str, piece_id: int) -> None:
        self.space, self.near, self.x, self.y, self.kind, self.id = space, None, 0.0, 0.0, "piece", piece_id


@pytest.fixture
def verified_map(monkeypatch):
    """Pretend the path table has been checked, so turns that move are allowed."""
    monkeypatch.setattr(map_graph, "VERIFIED", True)


def test_resting_beats_everything_when_the_investigator_is_nearly_dead(verified_map):
    game = _game()
    game.robot_investigator.health = 1
    situation = _situation(game)
    plans = moves.plans(game, situation)
    best = max((decide.score(game, situation, p) for p in plans), key=lambda s: s.score)
    assert any(a.key == moves.REST for a in best.plan.actions)
    assert any("kills it" in r for r in best.reasons)


def test_walking_towards_a_gate_scores_better_than_walking_away(verified_map):
    game = _game()
    game.board.seed([_Sighting("Tokyo", 1)])
    game.board.name(1, "gate:Tokyo")
    situation = _situation(game)
    towards = next(p for p in moves.plans(game, situation) if p.ends_at == "Tokyo")
    away = next(p for p in moves.plans(game, situation) if p.ends_at == "17")
    assert decide.score(game, situation, towards).score > decide.score(game, situation, away).score


def test_a_named_monster_on_the_destination_costs_the_plan(verified_map):
    game = _game()
    game.board.seed([_Sighting("Tokyo", 1)])
    game.board.name(1, "monster:Cultist")
    situation = _situation(game)
    onto = next(p for p in moves.plans(game, situation) if p.ends_at == "Tokyo")
    scored = decide.score(game, situation, onto)
    assert scored.score < 0
    assert any("means a fight" in r for r in scored.reasons)


def test_the_shortlist_keeps_one_plan_of_every_kind(verified_map):
    game = _game()
    game.robot_investigator.health = 4
    situation = _situation(game)
    scored = [decide.score(game, situation, p) for p in moves.plans(game, situation)]
    kept = decide.shortlist(scored, limit=6)
    assert len(kept) == 6
    first_keys = {s.plan.actions[0].key for s in kept}
    assert {moves.TRAVEL, moves.REST, moves.ACQUIRE, moves.PREPARE} <= first_keys
    assert kept == sorted(kept, key=lambda s: -s.score)


def test_the_model_choice_is_the_decision(verified_map):
    game = _game()
    chat_fn = _answers('{"choice": 2, "reason": "Vou para Tokyo.", "ask": ""}')
    decision = decide.decide(game, language="pt-BR", chat_fn=chat_fn, think=False)
    assert decision.chosen_by == "model"
    assert decision.reason == "Vou para Tokyo."
    assert decision.plan is not None
    assert decision.plan is decision.considered[0].plan or decision.plan is not None


def test_a_choice_outside_the_list_is_asked_again_once_then_the_score_decides(verified_map):
    game = _game()
    chat_fn = _answers('{"choice": 99, "reason": "Nope."}')
    decision = decide.decide(game, chat_fn=chat_fn, think=False)
    assert len(chat_fn.calls) == 2  # asked again exactly once
    assert decision.chosen_by == "score"
    assert decision.plan is not None
    assert decision.reason


def test_when_the_model_is_not_there_the_robot_still_plays(verified_map):
    def chat_fn(messages, **kwargs):
        raise LLMError("ollama is down")

    decision = decide.decide(game := _game(), chat_fn=chat_fn, think=False)
    assert decision.acted
    assert decision.chosen_by == "score"
    assert decision.reason
    assert game.robot_investigator is not None


def test_prose_around_the_json_is_still_read(verified_map):
    game = _game()
    chat_fn = _answers('Sure thing:\n{"choice": 1, "reason": "Fico e compro."}\nHope that helps.')
    decision = decide.decide(game, chat_fn=chat_fn, think=False)
    assert decision.chosen_by == "model"
    assert decision.reason == "Fico e compro."


def test_an_unverified_map_forbids_moving_a_piece():
    game = _game()
    chat_fn = _answers('{"choice": 1, "reason": "ok"}')
    decision = decide.decide(game, chat_fn=chat_fn, think=False)
    assert decision.plan is not None
    assert all(a.key != moves.TRAVEL for a in decision.plan.actions)
    assert any("checked against the board" in r for r in decision.refusals)


def test_a_delayed_investigator_is_told_so_and_nothing_is_chosen(verified_map):
    game = _game()
    game.robot_investigator.conditions = ["Delayed"]
    decision = decide.decide(game, chat_fn=_answers("{}"), think=False)
    assert not decision.acted
    assert decision.chosen_by == "rules"
    assert any("Delayed" in r for r in decision.refusals)


def test_the_questions_reach_the_table_instead_of_becoming_assumptions(verified_map):
    game = _game()
    game.board.seed([_Sighting("Shanghai", 1)])  # a piece nobody has named
    chat_fn = _answers('{"choice": 1, "reason": "ok", "ask": "Which Monster is that?"}')
    decision = decide.decide(game, chat_fn=chat_fn, think=False)
    assert decision.questions[0] == "Which Monster is that?"
    assert any("nobody has named" in q for q in decision.questions)


def test_the_brief_holds_the_state_and_the_printed_card_text(verified_map):
    game = _game()
    text = decide.brief(game, _situation(game))
    assert "Azathoth" in text and "Doom is at 15" in text
    assert "moves towards 0" in text  # the direction, or the model reads 15 of 15 as "almost over"
    assert "Find the way in" in text
    assert "Lily Chen" in text and "Shanghai" in text
    assert "Bull Whip" in text and "Strength" in text  # the printed effect, not a retelling


def test_advice_is_labelled_as_opinion_in_the_brief(verified_map):
    game = _game()
    text = decide.brief(
        game,
        _situation(game),
        advice=[{"text": "Close Gates early.", "source_author": "a forum guide"}],
    )
    assert "opinion" in text and "not a rule" in text
    assert "a forum guide" in text


def test_every_plan_considered_is_kept_for_the_record(verified_map):
    game = _game()
    decision = decide.decide(game, chat_fn=_answers('{"choice": 1, "reason": "ok"}'), think=False)
    record = decision.record()
    assert len(record["considered"]) == len(moves.plans(game, _situation(game)))
    assert record["considered"][0]["score"] >= record["considered"][-1]["score"]
    assert record["plan"]["text"]


def test_the_shortlist_shown_to_the_model_carries_what_each_plan_depends_on(verified_map):
    game = _game()
    situation = _situation(game)
    scored = [decide.score(game, situation, p) for p in moves.plans(game, situation)]
    text = decide.options_text(decide.shortlist(scored))
    assert text.splitlines()[0].startswith("1) ")
    assert "depends on" in text  # Acquire Assets never knows the card values


def test_a_reply_without_a_reason_falls_back_to_the_score_words(verified_map):
    game = _game()
    decision = decide.decide(game, chat_fn=_answers('{"choice": 1, "reason": ""}'), think=False)
    assert decision.reason
    assert decision.plan is not None


def test_the_printed_rule_of_every_offered_action_goes_with_the_shortlist(verified_map):
    game = _game()
    situation = _situation(game)
    scored = [decide.score(game, situation, p) for p in moves.plans(game, situation)]
    text = decide.rules_of(decide.shortlist(scored))
    assert "Acquire Assets: test Influence" in text  # not "pay with Clues", which it invented live
    assert "Rulebook, Action Phase" in text
