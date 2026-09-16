"""Reading a spoken briefing into the game state, with the model's answer faked."""

import json
from dataclasses import dataclass

from src.strategy.game import ROBOT, GameState
from src.strategy.setup import (
    SCHEMA,
    apply_reading,
    extract,
    questions_for,
    read_setup,
)


@dataclass
class FakeReply:
    text: str


def replying(payload) -> callable:
    """A chat function that answers with this payload, whatever it is asked."""

    def chat_fn(messages, **kwargs):
        chat_fn.calls.append((messages, kwargs))
        return FakeReply(payload if isinstance(payload, str) else json.dumps(payload))

    chat_fn.calls = []
    return chat_fn


# ---------------------------------------------------------------- extraction


def test_extract_asks_for_the_schema_and_no_creativity():
    chat_fn = replying({"ancient_one": "Azathoth", "investigators": [], "reserve": [], "mystery": ""})
    extract("o ancião é Azathoth", "pt-BR", chat_fn=chat_fn)
    _, kwargs = chat_fn.calls[0]
    assert kwargs["format"] is SCHEMA, "the answer must be constrained to the schema"
    assert kwargs["temperature"] == 0.0, "extraction must not be creative"


def test_extract_survives_a_model_that_does_not_answer_json():
    assert extract("qualquer coisa", "pt-BR", chat_fn=replying("I think the Ancient One is...")) == {}


def test_extract_survives_a_model_that_answers_a_list():
    assert extract("x", "pt-BR", chat_fn=replying([1, 2, 3])) == {}


# ---------------------------------------------------------------- applying


def test_a_full_briefing_fills_the_game():
    game = GameState()
    reading = read_setup(
        "o ancião é Azathoth, eu jogo com o Leo Anderson e você joga com a Lily Chen",
        "pt-BR",
        game,
        speaker="Alessandro",
        chat_fn=replying(
            {
                "ancient_one": "Azathoth",
                "doom": None,
                "mystery": "",
                "investigators": [
                    {"name": "Leo Anderson", "controller": "Alessandro"},
                    {"name": "Lily Chen", "controller": "reachy"},
                ],
                "reserve": [],
            }
        ),
    )
    assert game.ancient_one.name == "Azathoth" and game.doom == 15
    assert game.robot_investigator.name == "Lily Chen"
    assert game.robot_investigator.space == "Shanghai", "the sheet supplies the starting space"
    assert game.by_name("Leo Anderson").controller == "Alessandro"
    assert reading.understood_anything
    assert any("Lily Chen with you" in a for a in reading.applied)
    assert game.ready


def test_the_speaker_becomes_the_controller_when_they_say_only_me():
    game = GameState()
    read_setup(
        "eu jogo com o Leo",
        "pt-BR",
        game,
        speaker="Alessandro",
        chat_fn=replying(
            {
                "ancient_one": "",
                "investigators": [{"name": "Leo", "controller": "eu"}],
                "reserve": [],
                "mystery": "",
            }
        ),
    )
    assert game.by_name("Leo Anderson").controller == "Alessandro"


def test_you_in_any_wording_means_the_robot():
    for said in ("you", "reachy", "robot", "robô"):
        game = GameState()
        apply_reading({"investigators": [{"name": "Lily Chen", "controller": said}]}, game)
        assert game.robot_investigator is not None, f"controller {said!r} did not mean the robot"
        assert game.robot_investigator.controller == ROBOT


def test_an_unknown_investigator_becomes_a_question_not_a_guess():
    """Hard rule: base game only. An expansion name must be asked about."""
    game = GameState()
    reading = apply_reading({"investigators": [{"name": "Roland Banks", "controller": "reachy"}]}, game)
    assert game.investigators == [], "an unknown name was added anyway"
    assert any("Roland Banks" in u for u in reading.unknown)
    assert reading.questions and "Roland Banks" in reading.questions[0]


def test_a_misheard_name_comes_back_with_suggestions():
    game = GameState()
    reading = apply_reading({"investigators": [{"name": "lily chan", "controller": "reachy"}]}, game)
    assert reading.unknown and "Lily Chen" in reading.unknown[0]


def test_an_unknown_ancient_one_is_refused():
    game = GameState()
    reading = apply_reading({"ancient_one": "Nyarlathotep"}, game)
    assert game.ancient_one is None
    assert any("Nyarlathotep" in u for u in reading.unknown)


def test_a_briefing_can_arrive_in_pieces():
    """People do not say it all at once; a second call must add, not replace."""
    game = GameState()
    apply_reading({"ancient_one": "Cthulhu"}, game)
    assert game.ancient_one.name == "Cthulhu" and game.robot_investigator is None

    apply_reading({"investigators": [{"name": "Lily Chen", "controller": "reachy"}]}, game)
    assert game.ancient_one.name == "Cthulhu", "the second call wiped the first"
    assert game.robot_investigator.name == "Lily Chen"

    apply_reading({"reserve": ["Bank Loan"], "mystery": "Find the Key"}, game)
    assert game.reserve == ["Bank Loan"] and game.mystery == "Find the Key"
    assert game.ancient_one.name == "Cthulhu" and game.robot_investigator is not None


def test_a_stated_doom_overrides_the_starting_value():
    game = GameState()
    apply_reading({"ancient_one": "Azathoth", "doom": 11}, game)
    assert game.doom == 11, "a doom read off the board must win over the sheet default"


def test_nothing_understood_says_so():
    game = GameState()
    reading = apply_reading({}, game)
    assert not reading.understood_anything
    assert "did not catch" in reading.describe()


# ---------------------------------------------------------------- questions


def test_questions_come_one_at_a_time_and_in_order():
    game = GameState()
    asks = questions_for(game)
    assert "Ancient One" in asks[0]

    game.set_ancient_one("Azathoth")
    assert "investigators are in play" in questions_for(game)[0]

    game.add_investigator("Leo Anderson", controller="Alessandro")
    assert "Which investigator am I playing?" in questions_for(game)[0]

    game.add_investigator("Lily Chen", controller=ROBOT)
    assert "Mystery" in questions_for(game)[0]


def test_the_reading_reads_back_what_it_understood():
    game = GameState()
    reading = apply_reading(
        {"ancient_one": "Azathoth", "investigators": [{"name": "Lily Chen", "controller": "reachy"}]},
        game,
    )
    spoken = reading.describe()
    assert spoken.startswith("Got it:")
    assert "Azathoth" in spoken and "Lily Chen" in spoken
    assert reading.record()["text"] == spoken


def test_the_board_question_only_appears_once_there_are_pieces():
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

    game = GameState()
    game.set_ancient_one("Azathoth")
    game.add_investigator("Lily Chen", controller=ROBOT)
    game.mystery = "Find the Key"
    assert questions_for(game) == [], "nothing to ask before the board has been scanned"

    game.board = BoardState()
    game.board.seed([Sighting("Tokyo", x=10, y=10)])  # a piece, but not where Lily starts
    assert any("which piece" in q for q in questions_for(game))
