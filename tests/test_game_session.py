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
    fields: list = field(default_factory=list)

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
    session.game.mystery = "Find the Way In"
    assert session.wants_setup is False
    assert session.handle("eu comprei uma carta", "pt-BR") is None
    assert session.handle("o ancião continua sendo o Azathoth", "pt-BR") is None


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
    assert session.handle("o ancião é Azathoth", "pt-BR", reason="name") == "setup"
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
    # The person named the robot, so it answers even when the model got nothing out of it.
    session.handle("o ancião é aquele lá, o da caixa", "pt-BR", reason="name")
    assert "Qual Ancient One" in said.last  # it asks again rather than going quiet


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


def test_the_setup_ear_hears_the_mystery_after_the_rest_is_known(tmp_path):
    # The live session of 17:54: the saved game had the Ancient One and the investigator, so
    # "O mistério atual diz o seguinte" was table talk and was dropped.
    said = _Said()

    def read(text, language, game, **kwargs):
        game.mystery = "The Deep Ones Attack"
        reading = _Reading(["mystery: The Deep Ones Attack"], [], [])
        reading.fields = ["mystery"]
        return reading

    session = _session(tmp_path, said, setup_fn=read)
    session.game.set_ancient_one("Azathoth")
    session.game.add_investigator("Lily Chen", controller=ROBOT)
    assert session.game.ready is True  # and yet the Mystery was still missing
    assert session.wants_setup is True
    route = session.handle("O mistério atual diz o seguinte, ele é o The Deep Ones Attack", "pt-BR")
    assert route == "setup"
    assert session.game.mystery == "The Deep Ones Attack"
    assert "The Deep Ones Attack" in said.last


def test_a_correction_is_heard_even_when_nothing_is_missing(tmp_path):
    # A wrong Mystery used to be stuck: the ear only listened while something was missing.
    said = _Said()

    def read(text, language, game, **kwargs):
        game.mystery = "The Deep Ones Attack"
        reading = _Reading(["mystery: The Deep Ones Attack"], [], [])
        reading.fields = ["mystery"]
        return reading

    session = _session(tmp_path, said, setup_fn=read)
    session.game.set_ancient_one("Azathoth")
    session.game.add_investigator("Lily Chen", controller=ROBOT)
    session.game.mystery = "Q-Mystery"  # what Whisper left behind in the live session
    assert session.wants_setup is False
    assert session.handle("não, o Mystery é The Deep Ones Attack", "pt-BR") == "setup"
    assert session.game.mystery == "The Deep Ones Attack"


def test_a_statement_about_anything_else_is_not_hijacked_by_the_setup_ear(tmp_path):
    said = _Said()
    session = _session(tmp_path, said, setup_fn=_setup_fn())
    assert session.wants_setup is True  # nothing is known yet
    assert session.handle("eu comprei uma carta", "pt-BR") is None
    assert session.handle("vou pegar um café", "pt-BR") is None
    assert said.lines == []


def test_a_name_from_the_reference_is_enough_to_look_like_setup():
    from src.integration.game_session import looks_like_setup

    assert looks_like_setup("É a Lily Chen mesmo.")
    assert looks_like_setup("era o Azathoth sim")
    assert looks_like_setup("O mistério atual diz o seguinte")
    assert looks_like_setup("the current Mystery is Find the Way In")
    assert not looks_like_setup("coloquei um token no espaço 18")


def test_a_sentence_that_only_sounded_like_setup_goes_on_to_the_agent(tmp_path):
    # "Não, tu não entendeu. Esse foi o mistério que eu comprei." carries the word and is not a
    # briefing: the guess must cost the table nothing, not even a question read back.
    said = _Said()
    session = _session(tmp_path, said, setup_fn=lambda *a, **k: _Reading([], [], []))
    route = session.handle(
        "não, tu não entendeu, esse foi o mistério que eu comprei", "pt-BR", reason="setup_talk"
    )
    assert route is None  # the audio is released, the agent answers it
    assert said.lines == []


def test_a_real_briefing_heard_on_the_guess_is_kept(tmp_path):
    said = _Said()
    session = _session(tmp_path, said, setup_fn=_setup_fn(ancient_one="Azathoth"))
    route = session.handle("o ancião é o Azathoth", "pt-BR", reason="setup_talk")
    assert route == "setup"
    assert "Anotado" in said.last
    assert (tmp_path / "game_state.json").exists()


def test_the_same_sentence_after_the_robot_was_named_still_answers(tmp_path):
    # Named or holding the floor, the person is talking to the robot: it answers even when the
    # model got nothing out of the sentence.
    said = _Said()
    session = _session(tmp_path, said, setup_fn=lambda *a, **k: _Reading([], [], []))
    route = session.handle("o ancião é aquele lá", "pt-BR", reason="name")
    assert route == "setup"
    assert "Qual Ancient One" in said.last


def test_only_what_changed_is_read_back(tmp_path):
    said = _Said()

    def read(text, language, game, **kwargs):
        game.mystery = "The Deep Ones Attack"
        r = _Reading(["mystery: The Deep Ones Attack"], [], [])
        r.fields = ["mystery"]
        return r

    session = _session(tmp_path, said, setup_fn=read)
    session.game.set_ancient_one("Azathoth")
    session.game.add_investigator("Lily Chen", controller=ROBOT)
    session.game.reserve = ["Bull Whip", "Kerosene", "Dynamite", "Old Journal"]
    session.handle("o mistério atual é The Deep Ones Attack", "pt-BR", reason="setup_talk")
    assert "The Deep Ones Attack" in said.last
    assert "Bull Whip" not in said.last  # the Reserve did not change: it is not read back
    assert "Azathoth" not in said.last


def test_the_turn_comes_back_as_text_for_the_agent_to_say(tmp_path):
    from src.strategy.decide import Decision
    from src.strategy.moves import Candidate, TurnPlan

    session = GameSession.open(tmp_path / "game.json", language="pt-BR", background=False)
    session.game.add_investigator("Lily Chen", controller=ROBOT)
    plan = TurnPlan((Candidate("rest", "at Shanghai"),), "Shanghai")
    session.taker.decide_fn = lambda *a, **k: Decision(plan, "Vou descansar.", ask="Tem Monster aqui?")
    report = session.turn_report()
    assert report["took_a_turn"] is True
    assert report["move"] == "Rest at Shanghai"
    assert report["say"] == "Vou descansar. Tem Monster aqui?"
    assert "as it is" in report["note"]


def test_without_an_investigator_the_turn_tool_says_so_instead_of_inventing(tmp_path):
    session = GameSession.open(tmp_path / "game.json", language="pt-BR", background=False)
    report = session.turn_report()
    assert report["took_a_turn"] is False and report["say"] == ""
    assert "which investigator" in report["note"]


def test_the_setup_tool_writes_it_down_and_hands_back_one_sentence(tmp_path):
    said = _Said()
    session = _session(tmp_path, said, setup_fn=_setup_fn(ancient_one="Azathoth"))
    report = session.setup_report("o ancião é Azathoth")
    assert report["noted"] == ["facing Azathoth"]
    assert "Azathoth" in report["say"]
    assert any("investigator" in m for m in report["still_missing"])


def test_a_session_without_a_voice_never_speaks(tmp_path):
    # The agent is the only mouth at the table; the session is built without a say callable.
    session = GameSession.open(tmp_path / "game.json", language="pt-BR", background=False)
    session.game.add_investigator("Lily Chen", controller=ROBOT)
    session.setup_report("o ancião é Azathoth")  # would have spoken before
    session.turn_report()
    assert session.say("anything", "pt-BR") is None
