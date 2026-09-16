"""The spoken turn: the robot hears its investigator called and answers with a move."""

from __future__ import annotations

import time

from src.integration.turn_taking import CANNOT_ACT, NO_INVESTIGATOR, TurnTaker
from src.speech.addressee import Decision as Addressed
from src.speech.addressee import TurnLogger
from src.speech.addressee import decide as judge
from src.speech.gatekeeper import Gatekeeper
from src.strategy.decide import Decision
from src.strategy.game import ROBOT, GameState
from src.strategy.moves import Candidate, TurnPlan


def _game(with_robot: bool = True) -> GameState:
    game = GameState()
    game.set_ancient_one("Azathoth")
    if with_robot:
        game.add_investigator("Lily Chen", controller=ROBOT)
    game.add_investigator("Jacqueline Fine", controller="Alessandro")
    return game


def _decision(reason: str = "Vou ficar em Shanghai.", ask: str = "") -> Decision:
    plan = TurnPlan((Candidate("rest", "at Shanghai"),), "Shanghai")
    return Decision(plan, reason, ask=ask, questions=(ask,) if ask else (), chosen_by="model")


class _Said:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def __call__(self, text: str, language: str) -> None:
        self.lines.append((text, language))


def _taker(game: GameState, said: _Said, decision: Decision | None = None, **kwargs) -> TurnTaker:
    return TurnTaker(
        game,
        say=said,
        decide_fn=lambda *a, **k: decision or _decision(),
        background=False,
        **kwargs,
    )


def test_a_turn_call_naming_my_investigator_is_mine():
    said = _Said()
    taker = _taker(_game(), said)
    assert taker.handle("é a vez da Lily Chen", "pt-BR") is True
    assert said.lines == [("Vou ficar em Shanghai.", "pt-BR")]
    assert taker.turns == 1


def test_a_turn_call_for_somebody_else_is_not_mine():
    said = _Said()
    taker = _taker(_game(), said)
    assert taker.handle("é a vez da Jacqueline", "pt-BR") is False
    assert said.lines == []


def test_naming_my_investigator_without_handing_the_turn_is_not_a_turn():
    said = _Said()
    taker = _taker(_game(), said)
    # A question about the investigator belongs to the conversation, not to the Action Phase.
    assert taker.handle("a Lily Chen tem quantos Health?", "pt-BR") is False
    assert said.lines == []


def test_english_works_the_same_way():
    said = _Said()
    taker = _taker(_game(), said, _decision("I will rest."))
    assert taker.handle("it is Lily Chen's turn", "en-US") is True
    assert said.lines == [("I will rest.", "en-US")]


def test_the_question_the_model_wrote_is_spoken_and_the_english_notes_are_not():
    said = _Said()
    decision = Decision(
        _decision().plan,
        "Vou fazer Acquire Assets.",
        ask="Quanto vale o Bull Whip?",
        questions=("Quanto vale o Bull Whip?", "the printed value of each Reserve card"),
    )
    _taker(_game(), said, decision).handle("vez da Lily", "pt-BR")
    assert said.lines[0][0] == "Vou fazer Acquire Assets. Quanto vale o Bull Whip?"
    assert "printed value" not in said.lines[0][0]


def test_nothing_to_do_is_said_in_the_language_of_the_table():
    said = _Said()
    nothing = Decision(None, "I cannot take an action this turn.", chosen_by="rules")
    _taker(_game(), said, nothing).handle("é a vez da Lily Chen", "pt-BR")
    assert said.lines == [(CANNOT_ACT["pt-BR"], "pt-BR")]


def test_without_an_investigator_there_is_no_turn_to_take():
    said = _Said()
    taker = _taker(_game(with_robot=False), said)
    assert taker.investigator == ""
    assert taker.handle("é a vez da Lily Chen", "pt-BR") is False
    assert said.lines == []


def test_a_decision_that_crashes_does_not_take_the_conversation_down():
    said = _Said()

    def boom(*a, **k):
        raise RuntimeError("ollama exploded")

    taker = TurnTaker(_game(), say=said, decide_fn=boom, background=False)
    assert taker.handle("é a vez da Lily Chen", "pt-BR") is True
    assert said.lines == []
    assert taker.last is None


def test_the_table_repeating_itself_does_not_produce_two_moves():
    said = _Said()
    game = _game()
    started = []

    def slow(*a, **k):
        started.append(time.monotonic())
        time.sleep(0.3)
        return _decision()

    taker = TurnTaker(game, say=said, decide_fn=slow, background=True)
    assert taker.handle("é a vez da Lily Chen", "pt-BR") is True
    time.sleep(0.05)
    assert taker.handle("Lily Chen, sua vez", "pt-BR") is True  # swallowed, not queued
    time.sleep(0.5)
    assert len(started) == 1
    assert len(said.lines) == 1


def test_the_addressee_rules_and_the_taker_agree_on_what_a_turn_call_is():
    taker = _taker(_game(), _Said())
    for line in ("é a vez da Lily Chen", "agora é a Lily Chen", "it is Lily Chen's turn"):
        language = "en-US" if "turn" in line else "pt-BR"
        verdict = judge(line, language, my_investigator="Lily Chen")
        assert isinstance(verdict, Addressed)
        assert verdict.addressed
        if verdict.reason == "my_turn":
            assert taker.is_for_me(line, language)


class _Audio:
    """Enough of RobotAudioInterface for the gatekeeper's routing."""

    def __init__(self) -> None:
        self.last_played_at = 0.0
        self.gate_tail_s = 0.3
        self.dropped = 0
        self.released = 0

    def discard_utterance(self, until: float) -> int:
        self.dropped += 1
        return 4

    def release_utterance(self, until: float) -> int:
        self.released += 1
        return 4


class _Utterance:
    def __init__(self) -> None:
        self.started_at = 10.0
        self.ended_at = 12.0
        self.speech_ended_at = 11.8
        self.speech_s = 1.8
        self.audio = b""
        self.sample_rate = 16000
        self.path = None


class _Transcriber:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, *a, **k):
        class _R:
            pass

        result = _R()
        result.text = self.text
        result.language = "pt-BR"
        result.language_confidence = 0.99
        return result


def test_the_gate_keeps_a_turn_call_away_from_the_agent(tmp_path):
    audio, taken = _Audio(), []
    keeper = Gatekeeper(
        audio,
        _Transcriber("é a vez da Lily Chen"),
        TurnLogger(tmp_path / "addressee.jsonl"),
        humans=2,
        my_investigator=lambda: "Lily Chen",
        on_addressed=lambda text, language, reason, speaker: (taken.append(text), "my_turn")[1],
        clock=lambda: 20.0,
    )
    verdict = keeper.judge(_Utterance())
    assert verdict.route == "my_turn"
    assert verdict.decision.reason == "my_turn"
    assert taken == ["é a vez da Lily Chen"]
    assert audio.released == 0 and audio.dropped == 1  # the agent never heard it


def test_a_turn_the_robot_cannot_take_still_reaches_the_agent(tmp_path):
    audio = _Audio()
    keeper = Gatekeeper(
        audio,
        _Transcriber("é a vez da Lily Chen"),
        TurnLogger(tmp_path / "addressee.jsonl"),
        humans=2,
        my_investigator=lambda: "Lily Chen",
        on_addressed=lambda text, language, reason, speaker: None,  # no game loaded, say
        clock=lambda: 20.0,
    )
    verdict = keeper.judge(_Utterance())
    assert verdict.route == "released"
    assert audio.released == 1


def test_no_investigator_line_exists_in_both_languages():
    assert set(NO_INVESTIGATOR) == {"pt-BR", "en-US"}
    assert set(CANNOT_ACT) == {"pt-BR", "en-US"}
