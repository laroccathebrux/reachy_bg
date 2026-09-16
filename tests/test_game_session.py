"""The game kept across restarts: heard, written down, read back."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.integration.game_session import GameSession, briefing, next_question
from src.strategy.game import ROBOT, GameState
from src.strategy.setup import SetupReading


class _Said:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def __call__(self, text: str, language: str) -> None:
        self.lines.append((text, language))

    @property
    def last(self) -> str:
        return self.lines[-1][0] if self.lines else ""


@dataclass
class _Reading:
    """A stand-in for what src.strategy.setup returns, without the model."""

    applied: list
    questions: list
    unknown: list
    unknown_items: list = field(default_factory=list)

    def record(self) -> dict:
        return {"applied": self.applied, "questions": self.questions, "unknown": self.unknown}


def _setup_fn(**changes):
    """A fake extractor: applies ``changes`` to the game the first time it is called."""

    def read(text, language, game, **kwargs):
        applied = []
        if "ancient_one" in changes:
            game.set_ancient_one(changes["ancient_one"])
            applied.append(f"facing {changes['ancient_one']}")
        for name, controller in changes.get("investigators", []):
            game.add_investigator(name, controller=controller)
            applied.append(f"{name} with {controller}")
        return _Reading(applied, [], [])

    return read


def _session(tmp_path, said, **kwargs) -> GameSession:
    return GameSession.open(
        tmp_path / "game_state.json", say=said, language="pt-BR", background=False, **kwargs
    )


def test_a_game_the_robot_learns_survives_the_process(tmp_path):
    said = _Said()
    session = _session(
        tmp_path,
        said,
        setup_fn=_setup_fn(ancient_one="Azathoth", investigators=[("Lily Chen", ROBOT)]),
    )
    assert session.handle("o ancião é Azathoth e tu joga com a Lily Chen", "pt-BR") == "setup"
    assert (tmp_path / "game_state.json").exists()

    # A new process, the same file.
    again = _session(tmp_path, _Said())
    assert again.game.ancient_one is not None and again.game.ancient_one.name == "Azathoth"
    assert again.game.robot_investigator is not None
    assert again.game.robot_investigator.name == "Lily Chen"


def test_reopening_says_what_it_remembers_in_the_table_language(tmp_path):
    said = _Said()
    session = _session(
        tmp_path,
        said,
        setup_fn=_setup_fn(ancient_one="Azathoth", investigators=[("Lily Chen", ROBOT)]),
    )
    session.handle("o ancião é Azathoth, tu joga com a Lily Chen", "pt-BR")
    greeting = _session(tmp_path, _Said()).greeting("pt-BR")
    assert "Continuando" in greeting
    assert "Azathoth" in greeting and "Lily Chen" in greeting
    assert "We are facing" not in greeting


def test_a_brand_new_game_has_nothing_to_say(tmp_path):
    assert _session(tmp_path, _Said()).greeting("pt-BR") == ""


def test_the_robot_says_back_what_it_wrote_down_and_asks_for_the_rest(tmp_path):
    said = _Said()
    session = _session(tmp_path, said, setup_fn=_setup_fn(ancient_one="Azathoth"))
    session.handle("vamos jogar contra o Azathoth", "pt-BR")
    assert "Anotado" in said.last
    assert "Azathoth" in said.last
    # the next missing thing, one at a time: with no investigator at all, that comes first
    assert "Quais investigadores estão em jogo" in said.last


def test_a_question_is_never_taken_as_a_briefing(tmp_path):
    said = _Said()
    session = _session(tmp_path, said, setup_fn=_setup_fn(ancient_one="Azathoth"))
    assert session.handle("Reachy, o que é um Gate?", "pt-BR") is None
    assert session.handle("quantos investigadores a gente usa?", "pt-BR") is None
    assert said.lines == []


def test_once_the_game_is_known_a_statement_goes_to_the_agent(tmp_path):
    said = _Said()
    session = _session(tmp_path, said, setup_fn=_setup_fn())
    session.game.set_ancient_one("Azathoth")
    session.game.add_investigator("Lily Chen", controller=ROBOT)
    assert session.wants_setup is False
    assert session.handle("eu comprei uma carta", "pt-BR") is None


def test_a_turn_call_still_wins_over_the_setup_ear(tmp_path):
    said = _Said()
    session = _session(tmp_path, said, setup_fn=_setup_fn())
    session.game.add_investigator("Lily Chen", controller=ROBOT)
    session.taker.decide_fn = lambda *a, **k: _decision()
    assert session.handle("é a vez da Lily Chen", "pt-BR") == "my_turn"


def _decision():
    from src.strategy.decide import Decision
    from src.strategy.moves import Candidate, TurnPlan

    return Decision(TurnPlan((Candidate("rest", "at Shanghai"),), "Shanghai"), "Vou descansar.")


def test_an_extractor_that_crashes_leaves_the_game_alone(tmp_path):
    said = _Said()

    def boom(*a, **k):
        raise RuntimeError("ollama exploded")

    session = _session(tmp_path, said, setup_fn=boom)
    assert session.handle("o ancião é Azathoth", "pt-BR") == "setup"
    assert session.game.ancient_one is None
    assert said.lines == []


def test_a_corrupt_file_starts_a_new_game_instead_of_failing(tmp_path):
    (tmp_path / "game_state.json").write_text("{not json", encoding="utf-8")
    session = _session(tmp_path, _Said())
    assert session.game.ancient_one is None
    assert session.greeting() == ""


def test_nothing_extracted_still_answers_with_what_it_needs(tmp_path):
    said = _Said()
    session = _session(tmp_path, said, setup_fn=lambda *a, **k: _Reading([], [], []))
    session.handle("acho que vou tomar um café", "pt-BR")
    assert "Qual Ancient One" in said.last  # it repeats the question rather than going quiet


def test_the_briefing_and_the_questions_exist_in_both_languages():
    game = GameState()
    game.set_ancient_one("Azathoth")
    game.add_investigator("Lily Chen", controller=ROBOT)
    assert "We are facing Azathoth" in briefing(game, "en-US")
    assert "Vamos enfrentar Azathoth" in briefing(game, "pt-BR")
    assert next_question(game, None, "pt-BR").startswith("O que o Mystery")
    assert next_question(game, None, "en-US").startswith("What does the current Mystery")


def test_an_unknown_name_is_asked_about_first(tmp_path):
    game = GameState()
    game.set_ancient_one("Azathoth")
    reading = SetupReading()
    reading.unknown.append('the investigator "Lily Chang" (did you mean Lily Chen?)')
    assert "Lily Chang" in next_question(game, reading, "pt-BR")
    assert next_question(game, reading, "pt-BR").startswith("Não conheço")


def test_an_unrecognised_name_is_asked_about_in_the_table_language_with_its_candidates(tmp_path):
    # Whisper wrote "Azatov" and "Lily Shane" in the first live session.
    game = GameState()
    reading = SetupReading()
    reading.unknown_items.append({"kind": "ancient_one", "said": "Azatov", "suggestions": ["Azathoth"]})
    question = next_question(game, reading, "pt-BR")
    assert question == 'Não conheço o Ancient One "Azatov". Era Azathoth? Qual é?'
    english = next_question(game, reading, "en-US")
    assert english == 'I do not know the Ancient One "Azatov". Did you mean Azathoth? Which one is it?'


def test_the_speaker_becomes_the_controller_of_the_investigator_they_claim(tmp_path):
    said = _Said()
    seen = {}

    def read(text, language, game, speaker="", **kwargs):
        seen["speaker"] = speaker
        game.add_investigator("Jacqueline Fine", controller=speaker or "unknown")
        return _Reading(["Jacqueline Fine"], [], [])

    session = _session(tmp_path, said, setup_fn=read)
    session.handle("eu vou controlar a Jacqueline Fine", "pt-BR", speaker="Alessandro")
    assert seen["speaker"] == "Alessandro"
    assert "Jacqueline Fine (Alessandro)" in said.last
