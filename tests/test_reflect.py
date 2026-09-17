"""The loop that thinks between rounds, on a fake model and a fake lookup (no Ollama, no Qdrant)."""

import json

import pytest

from src.llm.ollama_client import LLMError
from src.strategy import map_graph
from src.strategy.game import ROBOT, GameState
from src.strategy.reflect import Reflection, Thinker


@pytest.fixture
def game():
    state = GameState()
    state.set_ancient_one("Azathoth")
    state.mystery = "The Deep One's Attack"
    state.reserve = ["Kerosene", "Bull Whip"]
    mine = state.add_investigator("Lily Chen", controller=ROBOT, space="Shanghai")
    mine.health = 3
    state.add_investigator("Jacqueline Fine", controller="Alessandro", space="San Francisco")
    state.begin_round()
    return state


@pytest.fixture
def verified_map():
    was = map_graph.VERIFIED
    map_graph.VERIFIED = True
    yield
    map_graph.VERIFIED = was


def _answers(*replies):
    """A model that says these things in order, then repeats the last one."""
    said = list(replies)

    def chat_fn(messages, **kwargs):
        text = said.pop(0) if len(said) > 1 else said[0]
        chat_fn.calls.append(messages)
        return type("R", (), {"text": text})()

    chat_fn.calls = []
    return chat_fn


def test_it_looks_something_up_before_it_commits(game, verified_map):
    chat_fn = _answers(
        json.dumps({"look_up": "what does Rest recover?"}),
        json.dumps(
            {
                "intents": [
                    {
                        "kind": "prefer",
                        "target": "Rest",
                        "why": "Lily Chen is at 3 Health",
                        "grounded_in": "Rest recovers 1 Health and 1 Sanity",
                    }
                ]
            }
        ),
    )
    looked: list[str] = []

    thinking = Reflection(
        game, chat_fn=chat_fn, look_up=lambda q: looked.append(q) or "Rest recovers 1 Health and 1 Sanity."
    ).run()

    assert looked == ["what does Rest recover?"]
    assert thinking.stopped == "committed"
    assert [i.target for i in thinking.intents] == ["rest"]
    assert thinking.intents[0].grounded_in == "Rest recovers 1 Health and 1 Sanity"
    # What it was told is in the conversation the second call saw.
    assert "Rest recovers 1 Health" in chat_fn.calls[-1][-1]["content"]


def test_a_loop_that_never_commits_still_ends(game, verified_map):
    chat_fn = _answers(json.dumps({"look_up": "and what about this?"}))
    thinking = Reflection(game, chat_fn=chat_fn, look_up=lambda q: "nothing", max_steps=3).run()
    assert thinking.stopped == "out of steps"
    assert thinking.steps == 3 and thinking.intents == ()


def test_a_clock_ends_it_even_mid_thought(game, verified_map):
    ticks = iter([0.0, 0.0, 500.0, 500.0, 500.0])
    chat_fn = _answers(json.dumps({"look_up": "?"}))
    thinking = Reflection(
        game, chat_fn=chat_fn, look_up=lambda q: "x", clock=lambda: next(ticks), max_seconds=10.0
    ).run()
    assert thinking.stopped == "out of time"


def test_the_round_starting_stops_it(game, verified_map):
    reflection = Reflection(game, chat_fn=_answers(json.dumps({"look_up": "?"})), look_up=lambda q: "x")
    reflection.cancel()
    thinking = reflection.run()
    assert thinking.stopped == "cancelled" and thinking.steps == 0


def test_a_model_that_is_not_there_costs_the_round_its_priorities_and_nothing_else(game, verified_map):
    def chat_fn(messages, **kwargs):
        raise LLMError("ollama is down")

    thinking = Reflection(game, chat_fn=chat_fn).run()
    assert thinking.stopped == "no model" and thinking.intents == ()


def test_what_it_invents_is_refused_and_what_holds_is_kept(game, verified_map):
    chat_fn = _answers(
        json.dumps({"look_up": "what does Rest do?"}),
        json.dumps(
            {
                "intents": [
                    {
                        "kind": "reach",
                        "target": "Innsmouth",
                        "why": "a Gate is there",
                        "grounded_in": "Rest recovers Health and Sanity",
                    },
                    {
                        "kind": "prefer",
                        "target": "Rest",
                        "why": "Lily Chen is at 3 Health",
                        "grounded_in": "Rest recovers Health and Sanity",
                    },
                ]
            }
        ),
    )
    thinking = Reflection(
        game, chat_fn=chat_fn, look_up=lambda q: "Rest recovers 1 Health and 1 Sanity."
    ).run()
    assert [i.target for i in thinking.intents] == ["rest"]
    assert any("Innsmouth" in r for r in thinking.refused)


def test_a_priority_that_cites_nothing_it_read_is_thrown_away(game, verified_map):
    """The failure the loop exists for: a real-sounding reason resting on nothing. On the first
    live run the model explained a Rest by "o combate iminente no espaço do Mar" - there is no
    such space, there was no combat, and it had read neither."""
    chat_fn = _answers(
        json.dumps({"look_up": "what does Rest do?"}),
        json.dumps(
            {
                "intents": [
                    {
                        "kind": "prefer",
                        "target": "Rest",
                        "why": "preciso me preparar",
                        "grounded_in": "o combate iminente no espaço do Mar",
                    }
                ]
            }
        ),
    )
    thinking = Reflection(
        game, chat_fn=chat_fn, look_up=lambda q: "Rest recovers 1 Health and 1 Sanity."
    ).run()
    assert thinking.intents == ()
    assert any("is not in anything it read" in r for r in thinking.refused)


def test_it_is_not_allowed_to_commit_before_it_has_read_anything(game, verified_map):
    """Having tools is not the same as using them, and a prompt asking nicely is not a
    mechanism: on the first live run it committed on step 1, every time."""
    chat_fn = _answers(
        json.dumps({"intents": [{"kind": "prefer", "target": "Rest", "why": "hurt"}]}),
        json.dumps({"look_up": "what does Rest do?"}),
        json.dumps(
            {
                "intents": [
                    {
                        "kind": "prefer",
                        "target": "Rest",
                        "why": "Lily Chen is at 3 Health",
                        "grounded_in": "Rest recovers 1 Health and 1 Sanity",
                    }
                ]
            }
        ),
    )
    thinking = Reflection(
        game, chat_fn=chat_fn, look_up=lambda q: "Rest recovers 1 Health and 1 Sanity."
    ).run()
    assert thinking.lookups == ("what does Rest do?",)
    assert [i.target for i in thinking.intents] == ["rest"]


def test_it_is_told_the_turns_the_score_will_be_choosing_between(game, verified_map):
    chat_fn = _answers(json.dumps({"intents": []}))
    Reflection(game, chat_fn=chat_fn, look_up=lambda q: "").run()
    asked = chat_fn.calls[0][-1]["content"]
    assert "The kind of turn the score will be choosing between" in asked
    assert "Acquire Assets" in asked


def _commits_to_rest():
    return _answers(
        json.dumps({"look_up": "what does Rest do?"}),
        json.dumps(
            {
                "intents": [
                    {
                        "kind": "prefer",
                        "target": "Rest",
                        "why": "at 3 Health",
                        "grounded_in": "Rest recovers 1 Health and 1 Sanity",
                    }
                ]
            }
        ),
    )


def test_the_thinker_hands_the_turn_what_it_settled_on(game, verified_map, monkeypatch):
    monkeypatch.setattr(
        "src.strategy.reflect._rules_lookup", lambda q: "Rest recovers 1 Health and 1 Sanity."
    )
    thinker = Thinker(game, chat_fn=_commits_to_rest(), background=False)
    assert thinker.take() == ()  # nothing decided yet: the turn happens anyway

    thinker.start()
    assert [i.target for i in thinker.take()] == ["rest"]


def test_a_thinker_with_no_investigator_of_its_own_has_nothing_to_think_about(verified_map):
    empty = GameState()
    called: list[int] = []
    thinker = Thinker(empty, chat_fn=lambda *a, **k: called.append(1), background=False)
    thinker.start()
    assert called == [] and thinker.take() == ()


def test_starting_a_new_round_throws_away_the_last_rounds_thinking(game, verified_map, monkeypatch):
    monkeypatch.setattr(
        "src.strategy.reflect._rules_lookup", lambda q: "Rest recovers 1 Health and 1 Sanity."
    )
    thinker = Thinker(game, chat_fn=_commits_to_rest(), background=False)
    thinker.start()
    assert thinker.take()
    thinker.cancel()
    assert thinker._running is None
